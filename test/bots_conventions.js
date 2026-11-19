// Simulated participants with signalling conventions, to validate analysis/label_strategies.py.
// Each bot has a convention for showing the triangle's direction with the circle:
//   F     step from the target toward the pointing direction and back (x2)
//   B     step opposite to the pointing direction and back (x2)
//   pause sit on the target (no direction cue)
// As receiver a bot decodes with its own convention. After each pairing it adopts the
// partner's convention with probability ADOPT, so conventions spread along the ladder.
// Usage: ADMIN_KEY=test TCG_SPEED=20 PORT=8081 node dev-server.js &
//        URL=http://localhost:8081 node test/bots_conventions.js [design] [n]
const WebSocket = require('./pollclient');
const URL_ = process.env.URL || 'http://localhost:8081';
const KEY = process.env.ADMIN_KEY || 'test';
const design = process.argv[2] || 'chain';
const n = Number(process.argv[3] || 6);
const ADOPT = 0.8;
const ids = Array.from({ length: n }, (_, i) => 'S' + String(i + 1).padStart(2, '0'));
const convention = Object.fromEntries(ids.map((id, i) => [id, i === 0 ? 'F' : i === 1 ? 'B' : 'pause']));
let rng = 7;
const rand = () => ((rng = (rng * 16807) % 2147483647) / 2147483647);

const DIRS = { 0: [0, 1, 'up', 'down'], 90: [1, 0, 'right', 'left'], 180: [0, -1, 'down', 'up'], 270: [-1, 0, 'left', 'right'] };
const ANGLE_OF = { '0,1': 0, '1,0': 90, '0,-1': 180, '-1,0': 270 };

function stepsTo(x, y, gx, gy) {
  const s = [];
  while (x < gx) { s.push('right'); x++; } while (x > gx) { s.push('left'); x--; }
  while (y < gy) { s.push('up'); y++; } while (y > gy) { s.push('down'); y--; }
  return s;
}

function bot(id) {
  const ws = new WebSocket(URL_.replace('http', 'ws'));
  let acted = null, seen = [], partnerConv = null;
  const run = (list) => {
    let delay = 0;
    for (const a of list) {
      if (typeof a === 'number') { delay += a; continue; }
      delay += 60;
      setTimeout(() => ws.send(JSON.stringify({ type: 'action', action: a })), delay);
    }
  };
  ws.on('open', () => ws.send(JSON.stringify({ type: 'join', id })));
  ws.on('message', (raw) => {
    const m = JSON.parse(raw);
    if (m.type === 'readyCheck' && !m.ready.includes(id)) return ws.send(JSON.stringify({ type: 'ready' }));
    if (m.type === 'paired') partnerConv = convention[m.partner];
    if (m.type === 'pairingDone' && partnerConv !== convention[id] && rand() < ADOPT) {
      console.log(`  ${id} adopts ${partnerConv} (was ${convention[id]}) after pairing with ${m.partner}`);
      convention[id] = partnerConv;
    }
    if (m.type !== 'state') return;
    const key = m.trial + m.phase;
    const me = m.tokens[m.you], other = m.tokens[3 - m.you];
    if (m.phase === 'sender' && m.yourRole === 'receiver' && other.state === 'in-use') seen.push([other.xPos, other.yPos, Date.now()]);
    if (acted === key) return;
    if (m.phase === 'sender' && m.yourRole === 'sender') {
      acted = key;
      const T = other.goal, G = me.goal, conv = convention[id];
      const [dx, dy, fwd, back] = DIRS[T.angle];
      const plan = ['stop', ...stepsTo(0, 0, T.xPos, T.yPos)];
      const sign = conv === 'B' ? -1 : 1;
      const nx = T.xPos + sign * dx, ny = T.yPos + sign * dy;
      if (conv !== 'pause' && Math.abs(nx) <= 1 && Math.abs(ny) <= 1) {
        const out = conv === 'B' ? back : fwd, ret = conv === 'B' ? fwd : back;
        plan.push(out, ret, out, ret);
      } else plan.push(1200);
      plan.push(...stepsTo(T.xPos, T.yPos, G.xPos, G.yPos), 'stop');
      run(plan);
    } else if (m.phase === 'receiver' && m.yourRole === 'receiver') {
      acted = key;
      // target: the cell the sender returned to most / stayed on longest
      const score = {};
      for (let i = 0; i < seen.length; i++) {
        const k = seen[i][0] + ',' + seen[i][1];
        const dur = (seen[i + 1] ? seen[i + 1][2] : seen[i][2]) - seen[i][2];
        score[k] = (score[k] || 0) + 1 + dur / 500;
      }
      delete score['0,0'];
      const best = Object.entries(score).sort((a, b) => b[1] - a[1])[0];
      const [tx, ty] = best ? best[0].split(',').map(Number) : [0, 0];
      // direction: first step off the target that returns to it
      let dir = null;
      for (let i = 0; i + 2 < seen.length; i++) {
        if (seen[i][0] === tx && seen[i][1] === ty && seen[i + 2][0] === tx && seen[i + 2][1] === ty) {
          dir = ANGLE_OF[`${seen[i + 1][0] - tx},${seen[i + 1][1] - ty}`];
          break;
        }
      }
      const conv = convention[id];
      let angle = 0;
      if (dir !== undefined && dir !== null) angle = conv === 'B' ? (dir + 180) % 360 : dir;
      const plan = ['stop', ...stepsTo(0, 0, tx, ty)];
      for (let a = 0; a < angle; a += 90) plan.push('rotateright');
      plan.push('stop');
      seen = [];
      run(plan);
    } else if (m.phase === 'receiver') seen = [];
  });
}

(async () => {
  const body = { name: 'conventions', participants: ids, design, numRounds: 4, trialsPerPairing: 12, turnSeconds: 6, seed: 11, trialTypes: [3] };
  const r = await fetch(`${URL_}/api/session?key=${KEY}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error);
  console.log('session', j.session.id);
  console.log('starting conventions', JSON.stringify(convention));
  ids.forEach(bot);
  const t0 = Date.now();
  const iv = setInterval(async () => {
    const s = await (await fetch(`${URL_}/api/status?key=${KEY}`)).json();
    const ps = Object.values(s.pairings);
    if (ps.every((p) => p.status === 'done')) {
      ps.forEach((p) => console.log(` round ${p.round} ${p.p1}-${p.p2} ${p.successes}/${p.trials}`));
      console.log('final conventions', JSON.stringify(convention));
      process.exit(0);
    }
    if (Date.now() - t0 > 300000) { console.log('TIMEOUT'); process.exit(1); }
  }, 500);
})();
