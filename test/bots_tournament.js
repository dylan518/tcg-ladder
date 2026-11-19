// End-to-end check of the tournament / pilot / one-on-one modes with scripted bots.
// Usage: ADMIN_KEY=test TCG_SPEED=20 TCG_STALE_MS=600 PORT=8081 DATA_DIR=/tmp/x node dev-server.js &
//        URL=http://localhost:8081 node test/bots_tournament.js [tournament|pilot|oneonone]   (URL may be the Vercel deployment)
//
// Bots: the sender walks to the goal tile and stops; the receiver goes to the sender's last
// tile and points a random way (so roughly 1 in 4 rounds succeed). Checks, and exits 1 on failure:
//  - rounds.csv: one row per round per group, matching schedule.json, success = tile and direction match
//  - nobody gets a board in a round they sit out; partner IDs are never sent; receivers never see goals
//  - tokens have a colour; everyone sees the partner-change message; rounds.csv has the degree of separation
//  - a disconnect mid-round replays that group's round (attempt 2) without duplicating rows
//  - pause takes effect after the round and resume continues
const fs = require('fs');
const path = require('path');
const WebSocket = require('./pollclient');
const URL_ = process.env.URL || 'http://localhost:8081';
const KEY = process.env.ADMIN_KEY || 'test';
const DATA_DIR = process.env.DATA_DIR;
const mode = process.argv[2] || 'tournament';
const AUTO = process.env.AUTO === '1'; // AUTO=1: no Start call, the session must start itself
const COLOR = { A: 'red', B: 'blue', C: 'green' };
const ORIG = { P1: 'A', P2: 'A', P3: 'A', P4: 'B', P5: 'B', P6: 'B', P7: 'C', P8: 'C', P9: 'C' };

const errors = [];
const fail = (m) => { if (errors.length < 50) errors.push(m); };
const api = async (p, body) => {
  const r = await fetch(`${URL_}/api/${p}?key=${KEY}`, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const j = await r.json();
  if (!r.ok) throw new Error(`${p}: ${j.error}`);
  return j;
};

function stepsTo(x, y, gx, gy) {
  const s = [];
  while (x < gx) { s.push('right'); x++; } while (x > gx) { s.push('left'); x--; }
  while (y < gy) { s.push('up'); y++; } while (y > gy) { s.push('down'); y--; }
  return s;
}

const bots = {};
let SWAP = Infinity; // first round after the swap message (tournament)
function bot(id, schedule) {
  const b = bots[id] ||= { id, rounds: [], sawSwap: false, lastSender: [0, 0], acted: null };
  const ws = b.ws = new WebSocket(URL_.replace('http', 'ws'));
  const act = (list) => list.forEach((a, i) => setTimeout(() => a !== 'wait' && ws.readyState === 1 && ws.send(JSON.stringify({ type: 'action', action: a })), 25 * (i + 1)));
  ws.on('open', () => ws.send(JSON.stringify({ type: 'join', id })));
  ws.on('message', (raw) => {
    const m = JSON.parse(raw);
    if (m.type === 'wait' && m.announce) b.sawSwap = true;
    if (m.type === 'wait' && m.canReady && AUTO) ws.send(JSON.stringify({ type: 'ready' }));
    if (m.type === 'paired') {
      b.round = m.round;
      b.acted = null; // a replayed round has the same number, so reset on every new board
      b.rounds.push(m.round);
      const rd = schedule[m.round - 1];
      const x = Object.values(rd.groups).find((g) => g.sender === id || g.receiver === id);
      if (!x) fail(`${id} got a board in round ${m.round} but sits out`);
      if (m.partner) fail(`partner id sent to ${id}`);
      if (mode === 'tournament' && m.round >= SWAP && !b.sawSwap) fail(`${id} reached round ${m.round} without the swap message`);
    }
    if (m.type !== 'state') return;
    if (m.partner) fail(`partner id in state for ${id}`);
    const me = m.tokens[m.you], other = m.tokens[3 - m.you];
    if (m.yourRole === 'receiver' && m.phase !== 'feedback' && (me.goal || other.goal)) fail(`receiver ${id} can see a goal`);
    if (m.phase === 'feedback' && m.yourRole === 'receiver') b.sawSolution ||= !!(me.goal || other.goal);
    if (m.yourRole === 'sender' && me.goal) fail(`sender ${id} has a goal of their own`);
    if (mode === 'tournament') {
      for (const t of [me, other]) if (!t.colorName) fail('token without colour');
    }
    if (m.phase === 'sender' && m.yourRole === 'receiver' && other.state === 'in-use') b.lastSender = [other.xPos, other.yPos];
    const k = `${m.trial}:${m.phase}`;
    if (b.acted === k) return;
    if (m.phase === 'sender' && m.yourRole === 'sender') {
      b.acted = k;
      const g = other.goal;
      act([...stepsTo(0, 0, g.xPos, g.yPos), ...Array(16).fill('wait'), 'stop']); // key is auto-placed; Space before any move in the first 0.4 s is ignored
    } else if (m.phase === 'receiver' && m.yourRole === 'receiver') {
      b.acted = k;
      const [tx, ty] = b.lastSender;
      const rot = Array(Math.floor(Math.random() * 4)).fill('rotateright');
      act([...stepsTo(0, 0, tx, ty), ...rot, ...Array(16).fill('wait'), 'stop']);
      b.lastSender = [0, 0];
    }
  });
  ws.on('close', () => {});
}

(async () => {
  const cfg = mode === 'tournament' ? { mode, players: Number(process.env.PLAYERS) || 8, turnSeconds: 3, name: 'bots', autoStart: AUTO, autoSend: process.env.AUTOSEND }
    : { mode, playerA: mode === 'pilot' ? 'P7' : 'Dylan', playerB: mode === 'pilot' ? 'P1' : 'Pav', rounds: 8, turnSeconds: 3, name: 'bots', autoStart: AUTO, autoSend: process.env.AUTOSEND };
  const dry = await api('tournament/dryrun', cfg);
  if (!dry.ok) throw new Error('dry run checks failed');
  const setup = await api('tournament/setup', cfg);
  const sid = setup.session.id;
  SWAP = (dry.schedule.find((rd) => rd.messageBefore) || {}).round || Infinity;
  const ids = setup.players.map((p) => p.id);
  console.log(`session ${sid}, ${ids.length} players, ${dry.schedule.length} rounds`);

  // Start must refuse while players are missing
  try { await api('tournament/start', {}); fail('start allowed with players offline'); } catch (e) { if (!/Not connected/.test(e.message)) fail(e.message); }
  ids.forEach((id) => bot(id, dry.schedule));
  await new Promise((r) => setTimeout(r, 800));
  if (AUTO) { const st = await api('status'); if (st.status === 'lobby') fail('auto start did not start with all players connected'); }
  else await api('tournament/start', {});

  const t0 = Date.now();
  let disconnected = false, pausedAt = null, resumed = false;
  for (;;) {
    await new Promise((r) => setTimeout(r, 150));
    const s = await api('status');
    // disconnect one sender mid-turn, reconnect a second later
    if (!disconnected && s.round === (mode === 'tournament' ? 10 : 3)) {
      const g = s.groups[0];
      if (g.state === 'sender' || g.state === 'receiver') {
        disconnected = true;
        const who = g.sender;
        console.log(`disconnecting ${who} in round ${s.round} (${g.state} phase)`);
        bots[who].ws.close();
        setTimeout(() => bot(who, dry.schedule), 1000);
      }
    }
    if (!pausedAt && s.round === (mode === 'tournament' ? 5 : 5) && s.status === 'running') {
      await api('tournament/pause', {});
      pausedAt = s.round;
      console.log(`pause requested in round ${s.round}`);
    }
    if (pausedAt && !resumed && s.status === 'paused') {
      // pause lands at the next round boundary: before round pausedAt (if requested between rounds) or after it
      if (s.round !== pausedAt && s.round !== pausedAt + 1) fail(`paused at round ${s.round}, expected ${pausedAt} or ${pausedAt + 1}`);
      if (s.groups.some((g) => g.state !== 'waiting')) fail('a group is still playing while paused');
      await new Promise((r) => setTimeout(r, 500));
      const again = await api('status');
      if (again.round !== s.round || again.status !== 'paused') fail('round advanced while paused');
      await api('tournament/resume', {});
      resumed = true;
      console.log(`paused cleanly before round ${s.round}, resumed`);
    }
    if (s.status === 'done') break;
    if (Date.now() - t0 > 480000) { fail('TIMEOUT ' + JSON.stringify({ status: s.status, round: s.round, groups: s.groups })); break; }
  }
  console.log(`finished in ${((Date.now() - t0) / 1000).toFixed(1)} s`);

  // ---- rounds.csv
  const lines = (await (await fetch(`${URL_}/api/data/rounds.csv?key=${KEY}&session=${encodeURIComponent(sid)}`)).text()).trim().split('\n');
  const head = lines[0].split(',');
  const rows = lines.slice(1).map((l) => {
    const cells = l.match(/("([^"]|"")*"|[^,]*)(,|$)/g).map((c) => c.replace(/,$/, '').replace(/^"|"$/g, '').replace(/""/g, '"'));
    return Object.fromEntries(head.map((h, i) => [h, cells[i]]));
  });
  const nRows = dry.schedule.reduce((a, rd) => a + Object.keys(rd.groups).length, 0);
  if (rows.length !== nRows) fail(`rounds.csv has ${rows.length} rows, want ${nRows}`);
  const seen = new Set();
  let wins = 0, replays = 0;
  for (const r of rows) {
    const k = `${r.round}:${r.group}`;
    if (seen.has(k)) fail(`duplicate row ${k}`);
    seen.add(k);
    const x = dry.schedule[r.round - 1].groups[r.group], goal = dry.goals[r.round - 1];
    if (r.sender_id !== x.sender || r.receiver_id !== x.receiver || r.sitting_out_id !== (x.sitout || dry.schedule[r.round - 1].sitout || '')) fail(`row ${k} roles differ from schedule`);
    if (r.goal_tile !== goal.tile || r.goal_direction !== goal.direction) fail(`row ${k} goal differs`);
    if (mode === 'tournament') {
      const want = r.phase === '1' ? '' : (x.separation === null ? 'none' : String(x.separation));
      if (r.degrees_of_separation !== want) fail(`row ${k} degrees_of_separation=${r.degrees_of_separation}, want ${want}`);
    }
    const ok = r.receiver_tile === r.goal_tile && r.receiver_direction === r.goal_direction ? '1' : '0';
    if (r.success !== ok) fail(`row ${k} success=${r.success} but placement says ${ok}`);
    if (!/^start\(/.test(r.sender_moves)) fail(`row ${k} sender moves missing: ${r.sender_moves}`);
    if (!(Number(r.sender_phase_ms) > 0) || !(Number(r.receiver_phase_ms) > 0)) fail(`row ${k} phase durations missing`);
    if (Number(r.attempt) > 1) replays++;
    wins += Number(r.success);
  }
  if (!replays) fail('no replayed round after the disconnect');
  const sawSolution = ids.some((id) => bots[id].sawSolution);
  if (sawSolution !== (mode === 'oneonone')) fail(`solution shown after rounds: ${sawSolution}, expected only in one-on-one`);
  if (mode === 'tournament') for (const id of ids) if (!bots[id].sawSwap) fail(`${id} never saw the swap message`);
  console.log(`rounds.csv: ${rows.length} rows, ${wins} successes, ${replays} replayed after disconnect`);
  console.log('sample row:', JSON.stringify(rows[0]));
  if (errors.length) { console.log('FAILED:\n  ' + errors.join('\n  ')); process.exit(1); }
  console.log('ALL CHECKS PASSED');
  process.exit(0);
})().catch((e) => { console.error(e); process.exit(1); });
