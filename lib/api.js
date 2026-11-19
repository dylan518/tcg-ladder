// HTTP API shared by the local server (dev-server.js) and the Vercel function (api/index.js).
//
// Players talk to the game by polling instead of a socket, so it runs on stateless hosts:
//   GET  /api/c?id=&token=&since=     new messages for this player (several times a second)
//   POST /api/c {id, token, since, msgs}  join / ready / key presses; replies with new messages
// Experimenter (needs ?key=ADMIN_KEY):
//   GET /api/status, POST /api/session, /api/preview, /api/tournament/{dryrun,setup,start,pause,resume}
//   GET /api/data/<file>[?session=]   download a data file of the active (or named) session
//   GET /api/sessions                 list sessions;  GET /api/export?session=  every file as JSON

const crypto = require('crypto');
const { step, KICKED_MSG, STALE_MS } = require('../engine');
const { makeStore } = require('../store');

const store = makeStore(process.env.TCG_LOCAL_DATA_DIR);
const ADMIN_KEY = process.env.ADMIN_KEY || (process.env.VERCEL ? null : crypto.randomBytes(4).toString('hex'));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Load the world, run one request, save it back; retried if another request saved first.
async function runStep(req) {
  for (let attempt = 0; attempt < 40; attempt++) {
    const { version, world, seen } = await store.load();
    const now = Date.now();
    const res = step(world, seen, req, now);
    if (!res.changed) return res;
    if (await store.commit(version, res, now)) {
      for (const l of res.logs) console.log(l);
      return res;
    }
    await sleep(5 + Math.random() * 20 * Math.min(attempt + 1, 5));
  }
  throw new Error('busy, please retry');
}

async function poll(id, token, since) {
  let r = await store.poll(id, token, since, Date.now());
  if (r.nextDue && r.nextDue <= Date.now()) {
    // a timer (turn timeout, next phase, disconnect check) is due: advance the world first
    await runStep({ kind: 'tick' });
    r = await store.poll(id, token, since, Date.now());
  }
  if (r.kicked) return { msgs: [], direct: [KICKED_MSG], kicked: true };
  return { msgs: r.msgs, rejoin: !!id && r.token !== token, hot: r.hot };
}

function readBody(req) {
  if (req.body !== undefined && req.body !== null) {
    if (typeof req.body === 'object' && !Buffer.isBuffer(req.body)) return Promise.resolve(req.body);
    const s = Buffer.isBuffer(req.body) ? req.body.toString('utf8') : String(req.body);
    return Promise.resolve(s ? JSON.parse(s) : {});
  }
  return new Promise((resolve, reject) => {
    let b = '';
    req.on('data', (c) => { b += c; if (b.length > 1e6) req.destroy(); });
    req.on('end', () => { try { resolve(b ? JSON.parse(b) : {}); } catch (e) { reject(e); } });
    req.on('error', reject);
  });
}

// returns true if it handled the request
async function handleApi(req, res) {
  const url = new URL(req.url, 'http://x');
  let route = url.pathname.replace(/^\/api\/?/, '');
  if (route === 'index' || route === 'index.js') route = url.searchParams.get('path') || ''; // Vercel rewrite
  const json = (code, obj) => {
    res.statusCode = code;
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'no-store');
    res.end(JSON.stringify(obj));
  };
  try {
    if (route === 'c') {
      if (req.method === 'POST') {
        const b = await readBody(req);
        const id = String(b.id || '').trim(), token = String(b.token || ''), since = Number(b.since) || 0;
        if (!id || !token) return json(400, { error: 'id and token required' });
        const r = await runStep({ kind: 'client', id, token, msgs: Array.isArray(b.msgs) ? b.msgs.slice(0, 50) : [] });
        if (r.reply?.direct) return json(200, { msgs: [], direct: r.reply.direct });
        return json(200, await poll(id, token, since));
      }
      const id = String(url.searchParams.get('id') || '').trim(), token = String(url.searchParams.get('token') || '');
      return json(200, await poll(id, token, Number(url.searchParams.get('since')) || 0));
    }

    if (route === 'slots') {
      // public: the player names of the active session, so one shared link works for everyone
      const { world, seen } = await store.load();
      const s = world?.session, now = Date.now();
      const names = (s?.participants || []).map((id) => ({ id, taken: !!(world.online?.[id] && seen[id] && now - seen[id] < STALE_MS) }));
      return json(200, { mode: s ? (s.tournament ? s.tournament.mode : 'ladder') : null, names });
    }

    if (route === 'assign' && req.method === 'POST') {
      // public: give this device a player slot of the active session (one link for everyone)
      const b = await readBody(req);
      const r = await runStep({ kind: 'assign', device: String(b.device || '').slice(0, 100), name: String(b.name || '').slice(0, 100), leave: !!b.leave });
      return json(r.reply.code, r.reply.body);
    }

    if (!ADMIN_KEY) return json(500, { error: 'ADMIN_KEY is not set on the server' });
    if (url.searchParams.get('key') !== ADMIN_KEY) return json(403, { error: 'bad admin key' });

    const dm = route.match(/^data\/(.+)$/);
    if (dm || route === 'export' || route === 'sessions') {
      if (route === 'sessions') return json(200, { sessions: await store.listSessions() });
      const session = url.searchParams.get('session') || (await store.load()).world?.session?.id;
      if (!session) return json(404, { error: 'no session' });
      if (route === 'export') {
        const files = {};
        for (const n of await store.listFiles(session)) files[n] = await store.readFile(session, n);
        return json(200, { session, files });
      }
      const name = dm[1];
      if (name.includes('..')) return json(400, { error: 'bad name' });
      const content = await store.readFile(session, name);
      if (content === null) return json(404, { error: 'no data yet' });
      res.statusCode = 200;
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Content-Disposition', `attachment; filename="${session}_${name.replace(/\//g, '_')}"`);
      return res.end(content);
    }

    const body = req.method === 'POST' ? await readBody(req) : {};
    if (route === 'tournament/results' && !body.log) {
      // rebuild the per-game log from rounds.csv (for games that ended before the results screen existed)
      const sid = (await store.load()).world?.session?.id;
      const csv = sid && await store.readFile(sid, 'rounds.csv');
      if (csv) body.log = require('../results').logFromRoundsCsv(csv);
    }
    const r = await runStep({ kind: 'admin', route, body });
    return json(r.reply.code, r.reply.body);
  } catch (e) {
    console.error(e);
    return json(500, { error: e.message });
  }
}

module.exports = { handleApi, ADMIN_KEY, store };
