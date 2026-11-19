// Fill some tournament slots with simple bots so a few real people can try the game.
// Usage: URL=https://tcg-ladder.vercel.app node test/fill_players.js P1 P2 P3 P4 P5 P6
// Each bot presses Ready, and plays like a slow person: as Sender it walks to the goal tile and
// stops; as Receiver it moves to a random tile, turns a random amount and stops. Ctrl+C to stop.
const WebSocket = require('./pollclient');
const URL_ = process.env.URL || 'http://localhost:8080';
const ids = process.argv.slice(2);
if (!ids.length) { console.error('give player IDs, e.g. P1 P2 P3'); process.exit(1); }

const steps = (gx, gy) => {
  const s = [];
  for (let y = 0; y < gy; y++) s.push('up'); for (let y = 0; y > gy; y--) s.push('down');
  for (let x = 0; x < gx; x++) s.push('right'); for (let x = 0; x > gx; x--) s.push('left');
  return s;
};

function bot(id) {
  const ws = new WebSocket(URL_.replace('http', 'ws'));
  let acted = null;
  const act = (list) => list.forEach((a, i) => setTimeout(() => ws.readyState === 1 && ws.send(JSON.stringify({ type: 'action', action: a })), 1200 + 500 * i));
  ws.on('open', () => ws.send(JSON.stringify({ type: 'join', id })));
  ws.on('message', (raw) => {
    const m = JSON.parse(raw);
    if (m.type === 'wait' && m.canReady) ws.send(JSON.stringify({ type: 'ready' }));
    if (m.type === 'wait' && m.title && !/Press Ready|Waiting for next/.test(m.title)) console.log(`${id}: ${m.title}`);
    if (m.type === 'error') { console.log(`${id}: ${m.message}`); if (m.kicked) setTimeout(() => ws.send(JSON.stringify({ type: 'join', id })), 1000); }
    if (m.type !== 'state') return;
    const k = `${m.round}:${m.trial}:${m.phase}`;
    if (acted === k) return;
    const me = m.tokens[m.you], other = m.tokens[3 - m.you];
    if (m.phase === 'sender' && m.yourRole === 'sender') {
      acted = k;
      act([...steps(other.goal.xPos, other.goal.yPos), 'stop']);
    } else if (m.phase === 'receiver' && m.yourRole === 'receiver') {
      acted = k;
      const tx = Math.floor(Math.random() * 3) - 1, ty = Math.floor(Math.random() * 3) - 1;
      act([...steps(tx, ty), ...Array(Math.floor(Math.random() * 8)).fill('rotateright'), 'stop']);
    } else if (m.phase === 'feedback' && m.yourRole === 'receiver') {
      acted = k;
      console.log(`${id} round ${m.round}: ${m.success ? 'success' : 'miss'}`);
    }
  });
  ws.on('close', () => {});
}

ids.forEach(bot);
console.log(`bots playing ${ids.join(', ')} on ${URL_}`);
