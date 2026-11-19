// End-to-end check: creates a ladder session and plays it with scripted bots.
// Usage: ADMIN_KEY=test TCG_SPEED=20 node dev-server.js & node test/bots.js [design] [n]
const WebSocket = require('./pollclient');
const URL_ = process.env.URL || 'http://localhost:8080';
const KEY = process.env.ADMIN_KEY || 'test';
const design = process.argv[2] || 'chain';
const n = Number(process.argv[3] || 4);
const ids = Array.from({ length: n }, (_, i) => 'B' + String(i + 1).padStart(2, '0'));

function stepsTo(tok, gx, gy) {
  const s = [];
  let { xPos: x, yPos: y } = tok;
  while (x < gx) { s.push('right'); x++; } while (x > gx) { s.push('left'); x--; }
  while (y < gy) { s.push('up'); y++; } while (y > gy) { s.push('down'); y--; }
  return s;
}
function rotTo(shape, from, to) {
  if (shape === 2) return [];
  const s = []; let a = from;
  while (a !== to && !(shape === 1 && Math.abs(a - to) === 180)) { s.push('rotateright'); a = (a + 90) % 360; }
  return s;
}

function bot(id) {
  const ws = new WebSocket(URL_.replace('http', 'ws'));
  let acted = null, seen = [];
  const act = (list) => list.forEach((a, i) => setTimeout(() => ws.send(JSON.stringify({ type: 'action', action: a })), 30 * (i + 1)));
  ws.on('open', () => ws.send(JSON.stringify({ type: 'join', id })));
  ws.on('message', (raw) => {
    const m = JSON.parse(raw);
    if (m.type === 'readyCheck' && !m.ready.includes(id)) return ws.send(JSON.stringify({ type: 'ready' }));
    if (m.type !== 'state') return;
    const key = m.trial + m.phase;
    const me = m.tokens[m.you], other = m.tokens[3 - m.you];
    if (m.phase === 'sender' && m.yourRole === 'receiver') seen.push([other.xPos, other.yPos]);
    if (acted === key) return;
    if (m.phase === 'sender' && m.yourRole === 'sender') {
      acted = key;
      // convention: visit partner's target, dwell, then go home
      const rg = other.goal, sg = me.goal;
      const toR = stepsTo(me, rg.xPos, rg.yPos);
      const toS = stepsTo({ xPos: rg.xPos, yPos: rg.yPos }, sg.xPos, sg.yPos);
      act(['stop', ...toR, ...rotTo(me.shape, 0, rg.angle), ...toS, ...rotTo(me.shape, me.shape === 2 ? 0 : rg.angle, sg.angle), 'stop']);
    } else if (m.phase === 'receiver' && m.yourRole === 'receiver') {
      acted = key;
      // guess: the cell the sender visited just before heading home (crude)
      const tgt = seen.length > 2 ? seen[Math.floor(seen.length / 2)] : [0, 0];
      seen = [];
      act(['stop', ...stepsTo(me, tgt[0], tgt[1]), 'stop']);
    } else if (m.phase === 'receiver') seen = [];
  });
  return ws;
}

(async () => {
  const body = { name: 'bots', participants: ids, design, numRounds: 3, trialsPerPairing: 6, turnSeconds: 5, seed: 42, trialTypes: [3] };
  const r = await fetch(`${URL_}/api/session?key=${KEY}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error);
  console.log('session', j.session.id, 'rounds', JSON.stringify(j.session.rounds));
  ids.forEach(bot);
  const t0 = Date.now();
  const iv = setInterval(async () => {
    const s = await (await fetch(`${URL_}/api/status?key=${KEY}`)).json();
    const ps = Object.values(s.pairings);
    const done = ps.filter((p) => p.status === 'done').length;
    if (done === ps.length) {
      console.log('all pairings done in', (Date.now() - t0) / 1000, 's');
      ps.forEach((p) => console.log(` round ${p.round} ${p.p1}-${p.p2} ${p.room} ${p.successes}/${p.trials}`));
      process.exit(0);
    }
    if (Date.now() - t0 > 120000) { console.log('TIMEOUT', JSON.stringify(s.pairings)); process.exit(1); }
  }, 500);
})();
