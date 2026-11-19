// Where the world, the player outboxes and the data files live.
//
// Postgres (Neon) when DATABASE_URL is set, otherwise an in-process memory store (local
// play and tests; data are also mirrored to data/<session>/ on disk so nothing is lost).
// Both expose the same async interface:
//   poll(id, token, since, now)  -> { msgs: [{s, m}], nextDue, token, hot }   (cheap, no world load)
//   load()                       -> { version, world, seen }
//   commit(version, res, now)    -> true if saved, false if someone else saved first
//   readFile(session, name), listFiles(session), listSessions()

const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------- Postgres (Neon over HTTP)

function pgStore(url) {
  const { neon } = require('@neondatabase/serverless');
  const sql = neon(url);
  const q = (text, params = []) => sql.query(text, params);
  let ready = null;
  const init = () => (ready ||= (async () => {
    await q(`CREATE TABLE IF NOT EXISTS tcg_world (id int PRIMARY KEY, version bigint NOT NULL, data text NOT NULL, meta jsonb NOT NULL, next_due bigint NOT NULL DEFAULT 0)`);
    await q(`INSERT INTO tcg_world (id, version, data, meta) VALUES (1, 0, 'null', '{}') ON CONFLICT DO NOTHING`);
    await q(`CREATE TABLE IF NOT EXISTS tcg_seen (client text PRIMARY KEY, at bigint NOT NULL)`);
    await q(`CREATE TABLE IF NOT EXISTS tcg_outbox (client text NOT NULL, seq bigint NOT NULL, msg jsonb NOT NULL, PRIMARY KEY (client, seq))`);
    await q(`CREATE TABLE IF NOT EXISTS tcg_files (id bigserial PRIMARY KEY, session text NOT NULL, name text NOT NULL, content text NOT NULL)`);
    await q(`CREATE INDEX IF NOT EXISTS tcg_files_sn ON tcg_files (session, name, id)`);
    await q(`CREATE TABLE IF NOT EXISTS tcg_docs (session text NOT NULL, name text NOT NULL, content text NOT NULL, PRIMARY KEY (session, name))`);
  })().catch((e) => { ready = null; throw e; }));

  return {
    kind: 'postgres',
    async poll(id, token, since, now) {
      await init();
      const [r] = await q(`
        WITH s AS (
          INSERT INTO tcg_seen (client, at) SELECT $1, $3 FROM tcg_world WHERE id = 1 AND meta->'conns'->>$1 = $4
          ON CONFLICT (client) DO UPDATE SET at = EXCLUDED.at RETURNING 1),
        d AS (DELETE FROM tcg_outbox WHERE client = $1 AND seq <= $2 RETURNING 1)
        SELECT (SELECT coalesce(json_agg(json_build_object('s', seq, 'm', msg) ORDER BY seq), '[]'::json) FROM tcg_outbox WHERE client = $1 AND seq > $2) AS msgs,
               w.next_due, w.meta->'conns'->>$1 AS token, (w.meta->'hot'->>$1) IS NOT NULL AS hot, (w.meta->'kicked'->>$4) IS NOT NULL AS kicked
        FROM tcg_world w WHERE w.id = 1`, [id, since, now, token]);
      return { msgs: r.msgs, nextDue: Number(r.next_due), token: r.token, hot: r.hot, kicked: r.kicked };
    },
    async load() {
      await init();
      const [r] = await q(`SELECT version, data, (SELECT coalesce(json_object_agg(client, at), '{}'::json) FROM tcg_seen) AS seen FROM tcg_world WHERE id = 1`);
      const seen = {};
      for (const [k, v] of Object.entries(r.seen || {})) seen[k] = Number(v);
      return { version: Number(r.version), world: JSON.parse(r.data), seen };
    },
    async commit(version, res, now) {
      const [r] = await q(`
        WITH u AS (UPDATE tcg_world SET version = version + 1, data = $2, meta = $3::jsonb, next_due = $4 WHERE id = 1 AND version = $1 RETURNING version),
        o AS (INSERT INTO tcg_outbox (client, seq, msg)
              SELECT x.client, x.seq, x.msg FROM u, jsonb_to_recordset($5::jsonb) AS x(client text, seq bigint, msg jsonb) RETURNING 1),
        r AS (DELETE FROM tcg_outbox t USING u, jsonb_to_recordset($8::jsonb) AS x(client text, upto bigint)
              WHERE t.client = x.client AND t.seq <= x.upto RETURNING 1),
        f AS (INSERT INTO tcg_files (session, name, content)
              SELECT e.j->>'session', e.j->>'name', e.j->>'content' FROM u, jsonb_array_elements($6::jsonb) WITH ORDINALITY AS e(j, ord)
              ORDER BY e.ord RETURNING 1),
        dd AS (INSERT INTO tcg_docs (session, name, content)
               SELECT x.session, x.name, x.content FROM u, jsonb_to_recordset($7::jsonb) AS x(session text, name text, content text)
               ON CONFLICT (session, name) DO UPDATE SET content = EXCLUDED.content RETURNING 1),
        s AS (INSERT INTO tcg_seen (client, at) SELECT $9, $10 FROM u WHERE $9::text IS NOT NULL
              ON CONFLICT (client) DO UPDATE SET at = EXCLUDED.at RETURNING 1)
        SELECT (SELECT version FROM u) AS v`,
      [version, JSON.stringify(res.world), JSON.stringify(res.meta), Math.round(res.nextDue || 0),
        JSON.stringify(res.out), JSON.stringify(res.appends), JSON.stringify(res.puts),
        JSON.stringify(res.resets || []), res.selfSeen, now]);
      return r.v !== null && r.v !== undefined;
    },
    async readFile(session, name) {
      await init();
      const [d] = await q(`SELECT content FROM tcg_docs WHERE session = $1 AND name = $2`, [session, name]);
      if (d) return d.content;
      const rows = await q(`SELECT content FROM tcg_files WHERE session = $1 AND name = $2 ORDER BY id`, [session, name]);
      return rows.length ? rows.map((x) => x.content).join('') : null;
    },
    async listFiles(session) {
      await init();
      const rows = await q(`SELECT name FROM tcg_docs WHERE session = $1 UNION SELECT DISTINCT name FROM tcg_files WHERE session = $1 ORDER BY name`, [session]);
      return rows.map((x) => x.name);
    },
    async listSessions() {
      await init();
      const rows = await q(`SELECT session FROM tcg_docs GROUP BY session UNION SELECT session FROM tcg_files GROUP BY session ORDER BY session DESC`);
      return rows.map((x) => x.session);
    },
  };
}

// ---------------------------------------------------------------- memory (local server, tests)

function memoryStore(dataDir) {
  const st = { version: 0, world: null, meta: { conns: {}, hot: {} }, nextDue: 0, seen: {}, outbox: {}, files: {}, docs: {} };
  const key = (s, n) => s + '\u0000' + n;
  // mirror to disk in the original layout, and restore the world on restart
  const worldFile = dataDir && path.join(dataDir, 'world.json');
  if (worldFile && fs.existsSync(worldFile)) {
    try { Object.assign(st, JSON.parse(fs.readFileSync(worldFile, 'utf8'))); } catch { /* start fresh */ }
  }
  const disk = (session, name, content, append) => {
    if (!dataDir) return;
    const f = path.join(dataDir, session, name);
    fs.mkdirSync(path.dirname(f), { recursive: true });
    if (append) fs.appendFileSync(f, content); else fs.writeFileSync(f, content);
  };
  return {
    kind: 'memory',
    async poll(id, token, since, now) {
      if (st.meta.conns?.[id] === token) st.seen[id] = now;
      const box = (st.outbox[id] ||= []);
      while (box.length && box[0].s <= since) box.shift();
      return { msgs: box.filter((x) => x.s > since), nextDue: st.nextDue, token: st.meta.conns?.[id] || null, hot: !!st.meta.hot?.[id], kicked: !!st.meta.kicked?.[token] };
    },
    async load() {
      return { version: st.version, world: st.world ? JSON.parse(JSON.stringify(st.world)) : null, seen: { ...st.seen } };
    },
    async commit(version, res, now) {
      if (version !== st.version) return false;
      st.version++; st.world = res.world; st.meta = res.meta; st.nextDue = res.nextDue;
      for (const r of res.resets || []) st.outbox[r.client] = (st.outbox[r.client] || []).filter((x) => x.s > r.upto);
      for (const o of res.out) (st.outbox[o.client] ||= []).push({ s: o.seq, m: o.msg });
      for (const a of res.appends) { (st.files[key(a.session, a.name)] ||= []).push(a.content); disk(a.session, a.name, a.content, true); }
      for (const p of res.puts) { st.docs[key(p.session, p.name)] = p.content; disk(p.session, p.name, p.content, false); }
      if (res.selfSeen) st.seen[res.selfSeen] = now;
      if (worldFile) { fs.mkdirSync(path.dirname(worldFile), { recursive: true }); fs.writeFileSync(worldFile, JSON.stringify({ version: st.version, world: st.world, meta: st.meta, nextDue: st.nextDue })); }
      return true;
    },
    async readFile(session, name) {
      const k = key(session, name);
      if (k in st.docs) return st.docs[k];
      if (st.files[k]) return st.files[k].join('');
      const f = dataDir && path.join(dataDir, session, name); // written before a restart
      return f && fs.existsSync(f) ? fs.readFileSync(f, 'utf8') : null;
    },
    async listFiles(session) {
      return [...new Set([...Object.keys(st.docs), ...Object.keys(st.files)].filter((k) => k.startsWith(session + '\u0000')).map((k) => k.split('\u0000')[1]))].sort();
    },
    async listSessions() {
      return [...new Set([...Object.keys(st.docs), ...Object.keys(st.files)].map((k) => k.split('\u0000')[0]))].sort().reverse();
    },
  };
}

// defaultDataDir: where the memory store mirrors data (the local server passes ./data)
function makeStore(defaultDataDir) {
  const url = process.env.DATABASE_URL || process.env.POSTGRES_URL;
  return url ? pgStore(url) : memoryStore(process.env.DATA_DIR === '' ? null : (process.env.DATA_DIR || defaultDataDir || null));
}

module.exports = { makeStore, pgStore, memoryStore };
