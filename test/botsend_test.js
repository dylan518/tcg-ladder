// 'botsend' mode: a bot sends to each human every round; the humans decode Dylan's strategy.
// Usage: URL=http://localhost:8096 ADMIN_KEY=test node test/botsend_test.js
const WebSocket = require('./pollclient');
const URL_ = process.env.URL || 'http://localhost:8096', KEY = process.env.ADMIN_KEY || 'test';
const api = async (p, b) => (await fetch(`${URL_}/api/${p}?key=${KEY}`, b ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) } : {})).json();
const errs = [];
const STEP = { up: [0, 1], down: [0, -1], left: [-1, 0], right: [1, 0] };
function human(id) {
  const ws = new WebSocket(URL_.replace('http', 'ws'));
  let trail = [], acted = null;
  ws.on('open', () => ws.send(JSON.stringify({ type: 'join', id })));
  ws.on('message', (raw) => {
    const m = JSON.parse(raw);
    if (m.type === 'wait' && m.canReady) ws.send(JSON.stringify({ type: 'ready' }));
    if (m.type === 'paired') { trail = []; acted = null; }
    if (m.type !== 'state') return;
    if (m.yourRole !== 'receiver') errs.push(`${id} was not the receiver`);
    const other = m.tokens[3 - m.you];
    if (m.phase === 'sender') { const l = trail[trail.length - 1]; if (!l || l[0] !== other.xPos || l[1] !== other.yPos) trail.push([other.xPos, other.yPos]); }
    if (m.phase === 'receiver' && acted !== m.round + ':' + m.trial) {
      acted = m.round + ':' + m.trial;
      // decode: excursions out of the centre; first one's farthest tile = goal tile, last one's = direction
      const d = ([x, y]) => Math.abs(x) + Math.abs(y);
      const trips = []; let cur = null;
      for (const p of trail.slice(1)) { if (d(p) === 0) { if (cur) trips.push(cur); cur = null; } else if (!cur || d(p) > d(cur)) cur = p; }
      if (cur) trips.push(cur);
      const dt = trips.length > 1 ? trips[trips.length - 1] : null;
      const k = dt ? (((Math.round(Math.atan2(dt[0], dt[1]) * 180 / Math.PI / 45)) % 8) + 8) % 8 : 0;
      const pos = [trips[0] || [0, 0]], far = 0;
      const [tx, ty] = pos[far] || [0, 0], acts = [];
      for (let y = 0; y < ty; y++) acts.push('up'); for (let y = 0; y > ty; y--) acts.push('down');
      for (let x = 0; x < tx; x++) acts.push('right'); for (let x = 0; x > tx; x--) acts.push('left');
      acts.push(...(k <= 4 ? Array(k).fill('rotateright') : Array(8 - k).fill('rotateleft')), 'stop');
      acts.forEach((a, k) => setTimeout(() => ws.send(JSON.stringify({ type: 'action', action: a })), 450 + 30 * k));
    }
  });
}
(async () => {
  const s = await api('tournament/setup', { mode: 'botsend', playerA: 'Dylan', playerB: 'Pav', rounds: 12, goalStart: 1, turnSeconds: 10, showSolution: true, name: 'botsend' });
  if (!s.session) throw new Error(JSON.stringify(s));
  human('Dylan'); human('Pav');
  const t0 = Date.now();
  for (;;) { await new Promise((r) => setTimeout(r, 200)); const st = await api('status'); if (st.status === 'done') break; if (Date.now() - t0 > 240000) { errs.push('timeout ' + st.status + ' ' + st.round); break; } }
  const csv = await (await fetch(`${URL_}/api/data/rounds.csv?key=${KEY}&session=${s.session.id}`)).text();
  const head = csv.split('\n')[0].split(',');
  const rows = csv.trim().split('\n').slice(1).map((l) => Object.fromEntries(l.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map((c) => c.replace(/,$/, '').replace(/^"|"$/g, '')).map((c, i) => [head[i], c])));
  if (rows.length !== 24) errs.push(`rows ${rows.length}, want 12`);
  for (const r of rows) if (r.sender_id !== 'Bot-' + r.receiver_id) errs.push('wrong sender ' + r.sender_id);
  const wins = rows.filter((r) => r.success === '1').length;
  console.log(rows.slice(0, 2).map((r) => `${r.receiver_id} goal ${r.goal_tile}/${r.goal_direction}: ${r.sender_moves} -> ${r.receiver_tile}/${r.receiver_direction} ${r.success}`).join('\n'));
  console.log(`${wins}/${rows.length} wins`);
  if (wins !== rows.length) errs.push('the decoder should win every round');
  console.log(errs.length ? 'FAILED\n  ' + errs.slice(0, 10).join('\n  ') : 'ALL CHECKS PASSED');
  process.exit(errs.length ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
