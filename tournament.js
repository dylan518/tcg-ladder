// Tournament / pilot / one-on-one schedules, the fixed goal list, and schedule checks.
// Pure functions only (no server state), so `node tournament.js` prints the dry run.
//
// A "round" is one trial: in each group one Sender sends to one Receiver and (in 3-player
// groups) the third player sits out.

const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------- goals

const GOALS_FILE = path.join(__dirname, 'goals.json');
const GOAL_SEED = 63063;
// 8 compass directions (45° steps). The 4-direction goal set used before 2026-11-09 is kept in goals_4dir.json.
const DIRECTIONS = ['up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left'];
const DIR_ANGLE = Object.fromEntries(DIRECTIONS.map((d, i) => [d, i * 45]));
const ANGLE_DIR = Object.fromEntries(DIRECTIONS.map((d, i) => [i * 45, d]));
const TILE_NAME = {
  '-1,1': 'top-left', '0,1': 'top', '1,1': 'top-right',
  '-1,0': 'left', '0,0': 'center', '1,0': 'right',
  '-1,-1': 'bottom-left', '0,-1': 'bottom', '1,-1': 'bottom-right',
};
const tileName = (x, y) => TILE_NAME[`${x},${y}`];

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle(arr, rand) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// One goal per tournament round (30): never the centre tile; directions and tiles as even as possible (each of the
// 8 directions and each of the 8 outer tiles 4 or 5 times).
function generateGoals(seed = GOAL_SEED, n = TOTAL_ROUNDS) {
  const rand = mulberry32(seed);
  const outer = Object.keys(TILE_NAME).filter((k) => k !== '0,0');
  let tiles = [];
  while (tiles.length + outer.length <= n) tiles.push(...outer);
  tiles.push(...shuffle(outer, rand).slice(0, n - tiles.length));
  tiles = shuffle(tiles, rand);
  let dirs = [];
  while (dirs.length + DIRECTIONS.length <= n) dirs.push(...DIRECTIONS);
  dirs.push(...shuffle(DIRECTIONS, rand).slice(0, n - dirs.length));
  dirs = shuffle(dirs, rand);
  return tiles.map((k, i) => {
    const [x, y] = k.split(',').map(Number);
    return { round: i + 1, x, y, tile: tileName(x, y), direction: dirs[i] };
  });
}

// Fixed across sessions: created once, then always read back from goals.json.
function loadGoals() {
  if (!fs.existsSync(GOALS_FILE)) {
    fs.writeFileSync(GOALS_FILE, JSON.stringify({ seed: GOAL_SEED, goals: generateGoals() }, null, 2));
  }
  return JSON.parse(fs.readFileSync(GOALS_FILE, 'utf8')).goals;
}

// Tournament: a new random goal order for every session (from the session seed), drawn from the same balanced
// set of 30 goals in goals.json (so sessions pool), stratified so each phase gets the same number of diagonal
// directions (±1). Every pair plays the same goal in a given round.
function sessionGoals(seed) {
  const rand = mulberry32(((Number(seed) || 1) * 2654435761) >>> 0 ^ 0x5bd1e995);
  const all = loadGoals().slice(0, TOTAL_ROUNDS).map(({ round, ...g }) => g);
  const diag = shuffle(all.filter((g) => g.direction.includes('-')), rand);
  const card = shuffle(all.filter((g) => !g.direction.includes('-')), rand);
  const phases = Array.from({ length: PHASES }, () => []);
  diag.forEach((g, i) => phases[i % PHASES].push(g));
  // fill each phase up to ROUNDS_PER_PHASE with straight directions
  for (const ph of shuffle(phases, rand)) while (ph.length < ROUNDS_PER_PHASE && card.length) ph.push(card.pop());
  return phases.flatMap((ph) => shuffle(ph, rand)).map((g, i) => ({ round: i + 1, ...g }));
}

// ---------------------------------------------------------------- tournament schedule
//
// 3 phases × 10 rounds. In each phase everyone has ONE partner and the two alternate roles every round
// (each sends 5, receives 5). Nobody sits out; no bot. A neutral message before rounds 11 and 21.
// The partner for each phase is a fresh random pairing (seeded), so you may get the same partner again.
// Degree of separation of a pair in phase k = shortest path between the two in the network of who
// played with whom in the earlier phases (1 = played together before, 2 = partner's partner, …;
// null = not connected). Stored with every round in rounds.csv for the analysis.
const ROUNDS_PER_PHASE = 10, PHASES = 3, TOTAL_ROUNDS = ROUNDS_PER_PHASE * PHASES;
// shown full-screen before rounds 11 and 21 (title + detail)
const SWAP_MESSAGE = 'New partner';
const SWAP_DETAIL = 'Partners have been randomly reassigned. You will play the next 10 rounds with your new partner. Same rules as before.';
const SITOUT_DETAIL = 'Partners have been randomly reassigned. With an odd number of players one person waits each part, and this time it is you. Please keep this page open; you will play again later.';
const SWAP_ROUND = ROUNDS_PER_PHASE + 1; // first message (the second is before round 21)
const DEFAULT_PLAYERS = 8;
const ORIGINAL_GROUP = {}; // (no groups in this design)
const CONFEDERATE = null;

function tournamentPlayers(n) { return Array.from({ length: n }, (_, i) => 'P' + (i + 1)); }

function randomPairing(players, rand) {
  const o = shuffle(players, rand), m = [];
  for (let i = 0; i + 1 < o.length; i += 2) m.push([o[i], o[i + 1]]);
  return m;
}

// shortest path between a and b using the partner edges of the given phases
function separation(phases, a, b) {
  const adj = {};
  for (const ph of phases) for (const [x, y] of ph) { (adj[x] ||= new Set()).add(y); (adj[y] ||= new Set()).add(x); }
  const dist = { [a]: 0 }, q = [a];
  while (q.length) {
    const u = q.shift();
    if (u === b) return dist[u];
    for (const v of adj[u] || []) if (!(v in dist)) { dist[v] = dist[u] + 1; q.push(v); }
  }
  return null;
}

// Each phase: a fresh random pairing in which nobody gets a partner they have already played with
// (drawn by rejection sampling; if no such pairing exists, e.g. 2 players, the draw with the fewest repeats is used).
// Odd number of players: each phase one random person sits out all its rounds, a different person each phase
// (until everyone has sat out once).
const PAIR_KEY = (a, b) => (a < b ? a + '|' + b : b + '|' + a);
function tournamentPairs(players, seed) {
  const rand = mulberry32(Number(seed) || 1);
  const phases = [], sitters = [], sat = new Set(), used = new Set();
  for (let k = 0; k < PHASES; k++) {
    let best = null;
    for (let attempt = 0; attempt < 5000; attempt++) {
      let pool = players, sitter = null;
      if (players.length % 2) {
        let cand = players.filter((p) => !sat.has(p));
        if (!cand.length) cand = players;
        sitter = cand[Math.floor(rand() * cand.length)];
        pool = players.filter((p) => p !== sitter);
      }
      const pairs = randomPairing(pool, rand);
      const repeats = pairs.filter(([a, b]) => used.has(PAIR_KEY(a, b))).length;
      if (!best || repeats < best.repeats) best = { pairs, sitter, repeats };
      if (repeats === 0) break;
    }
    if (best.sitter) { if (players.filter((p) => !sat.has(p)).length === 0) sat.clear(); sat.add(best.sitter); }
    best.pairs.forEach(([a, b]) => used.add(PAIR_KEY(a, b)));
    phases.push(best.pairs);
    sitters.push(best.sitter);
  }
  const sep = phases.map((ph, k) => ph.map(([a, b]) => (k === 0 ? null : separation(phases.slice(0, k), a, b))));
  return { phases, sep, sitters };
}

function tournamentSchedule(players, seed) {
  const { phases, sep, sitters } = tournamentPairs(players, seed);
  const rounds = [];
  for (let r = 1; r <= TOTAL_ROUNDS; r++) {
    const phase = Math.ceil(r / ROUNDS_PER_PHASE), k = (r - 1) % ROUNDS_PER_PHASE;
    const groups = {};
    phases[phase - 1].forEach(([a, b], i) => {
      const aSends = k % 2 === 0;
      groups['pair' + (i + 1)] = { members: [a, b], sender: aSends ? a : b, receiver: aSends ? b : a, sitout: null,
        separation: sep[phase - 1][i] };
    });
    rounds.push({ round: r, phase, groups, ...(sitters[phase - 1] ? { sitout: sitters[phase - 1] } : {}),
      ...(k === 0 && phase > 1 ? { messageBefore: SWAP_MESSAGE } : {}) });
  }
  return rounds;
}

// Two players. 'pilot': playerA always sends. 'oneonone': roles alternate, playerA first.
function pairSchedule(mode, a, b, n) {
  const g = mode === 'pilot' ? 'pilot' : 'oneonone';
  return Array.from({ length: n }, (_, i) => {
    const aSends = mode === 'pilot' || i % 2 === 0;
    return { round: i + 1, phase: 1, groups: { [g]: { members: [a, b], sender: aSends ? a : b, receiver: aSends ? b : a, sitout: null } } };
  });
}

// ---------------------------------------------------------------- checks

function checkTournament(rounds, goals, players) {
  const checks = [];
  const add = (name, problems) => checks.push({ name, ok: problems.length === 0, problems });
  const n = players.length, R = ROUNDS_PER_PHASE;
  const partnerIn = (ph) => {
    const m = {};
    for (const rd of rounds.filter((x) => x.phase === ph)) for (const x of Object.values(rd.groups)) {
      for (const [a, b] of [[x.sender, x.receiver], [x.receiver, x.sender]]) (m[a] ||= new Set()).add(b);
    }
    return m;
  };

  const p1 = [];
  if (n < 2) p1.push(`needs at least 2 players, got ${n}`);
  if (rounds.length !== TOTAL_ROUNDS) p1.push(`expected ${TOTAL_ROUNDS} rounds, got ${rounds.length}`);
  for (const rd of rounds) {
    const seen = [];
    for (const x of Object.values(rd.groups)) seen.push(x.sender, x.receiver);
    const missing = players.filter((p) => !seen.includes(p) && p !== rd.sitout), twice = seen.filter((p, i) => seen.indexOf(p) !== i);
    if (missing.length) p1.push(`round ${rd.round}: ${missing.join(', ')} not playing`);
    if (twice.length) p1.push(`round ${rd.round}: ${twice.join(', ')} in two pairs`);
    if (rd.sitout && seen.includes(rd.sitout)) p1.push(`round ${rd.round}: ${rd.sitout} sits out but is in a pair`);
    if (!!rd.sitout !== (n % 2 === 1)) p1.push(`round ${rd.round}: ${n % 2 ? 'nobody sits out (odd number of players)' : 'someone sits out (even number)'}`);
  }
  const sitters = [...Array(PHASES)].map((_, i) => rounds.find((x) => x.phase === i + 1)?.sitout).filter(Boolean);
  for (let ph = 1; ph <= PHASES; ph++) if (new Set(rounds.filter((x) => x.phase === ph).map((x) => x.sitout || '')).size > 1) p1.push(`phase ${ph}: sitter changes within the phase`);
  if (n >= PHASES && new Set(sitters).size !== sitters.length) p1.push(`the same person sits out twice: ${sitters.join(', ')}`);
  add(n % 2
    ? `1. ${TOTAL_ROUNDS} rounds (${PHASES} phases × ${R}); odd number (${n}): one person sits out each whole phase, a different person each phase (${sitters.join(', ')}); everyone else is in exactly one pair`
    : `1. ${TOTAL_ROUNDS} rounds (${PHASES} phases × ${R}); every round all ${n} players are in exactly one pair; nobody sits out`, [...new Set(p1)]);

  const p2 = [];
  for (let ph = 1; ph <= PHASES; ph++) {
    const m = partnerIn(ph), c = {};
    for (const rd of rounds.filter((x) => x.phase === ph)) for (const x of Object.values(rd.groups)) {
      (c[x.sender] ||= { s: 0, r: 0 }).s++; (c[x.receiver] ||= { s: 0, r: 0 }).r++;
    }
    const sitter = rounds.find((x) => x.phase === ph)?.sitout;
    for (const p of players) {
      if (p === sitter) continue;
      if (!m[p] || m[p].size !== 1) p2.push(`phase ${ph}: ${p} has ${m[p] ? m[p].size : 0} partners (want 1)`);
      if (!c[p] || c[p].s !== R / 2 || c[p].r !== R / 2) p2.push(`phase ${ph}: ${p} sends ${c[p]?.s || 0}, receives ${c[p]?.r || 0} (want ${R / 2} each)`);
    }
    for (const rd of rounds.filter((x) => x.phase === ph).slice(1)) {
      const prev = rounds.find((x) => x.round === rd.round - 1);
      for (const [g, x] of Object.entries(rd.groups)) if (prev.groups[g] && prev.groups[g].sender === x.sender) p2.push(`round ${rd.round} ${g}: roles did not switch`);
    }
  }
  add(`2. Each phase: one partner per person for all ${R} rounds; roles alternate every round (each sends ${R / 2}, receives ${R / 2})`, p2);

  const fmt = (ph) => {
    const r = rounds.find((x) => x.phase === ph);
    return r ? Object.values(r.groups).map((x) => `${x.members.join('–')}: ${x.separation === null ? 'none' : x.separation}`).join(', ') + (r.sitout ? ` (${r.sitout} sits out)` : '') : '';
  };
  const p3 = [];
  for (let ph = 2; ph <= PHASES; ph++) {
    const r = rounds.find((x) => x.phase === ph);
    if (r && Object.values(r.groups).some((x) => x.separation === undefined)) p3.push(`phase ${ph}: separation missing`);
  }
  const seenPairs = new Map();
  for (let ph = 1; ph <= PHASES; ph++) {
    const r = rounds.find((x) => x.phase === ph);
    if (r) for (const x of Object.values(r.groups)) { const k = PAIR_KEY(...x.members); seenPairs.set(k, [...(seenPairs.get(k) || []), ph]); }
  }
  const possible = n >= 4 || n === 3; // 2 players can only ever play each other
  if (possible) for (const [k, phs] of seenPairs) if (phs.length > 1) p3.push(`${k.replace('|', ' & ')} paired again in phases ${phs.join(', ')}`);
  add(`3. New random partner each phase (nobody plays the same partner twice${possible ? '' : '; impossible with 2 players'}). Degrees of separation, phase 2: ${fmt(2)}`, p3);
  add(`4. Phase 3 degrees of separation (from who played together in phases 1–2; 1 = same partner again): ${fmt(3)}`, []);

  const p5 = [];
  for (const r of [R + 1, 2 * R + 1]) if (!rounds.find((x) => x.round === r)?.messageBefore) p5.push(`no message before round ${r}`);
  if (rounds.some((x) => x.messageBefore && ![R + 1, 2 * R + 1].includes(x.round))) p5.push('a message in the wrong place');
  add(`5. "${SWAP_MESSAGE}" screen ("${SWAP_DETAIL}") before rounds ${R + 1} and ${2 * R + 1}, nowhere else`, p5);

  const diagPer = [...Array(PHASES)].map((_, k) => goals.slice(k * R, (k + 1) * R).filter((g) => g.direction.includes('-')).length);
  const gp = goalProblems(goals, TOTAL_ROUNDS);
  if (Math.max(...diagPer) - Math.min(...diagPer) > 1) gp.push(`diagonal goals per phase ${diagPer.join('/')} (want within 1)`);
  add(`6. Goals: random order for this session; ${TOTAL_ROUNDS} goals, none on the centre tile, each of the 8 directions and each outer tile ${Math.floor(TOTAL_ROUNDS / 8)} or ${Math.ceil(TOTAL_ROUNDS / 8)} times; diagonal goals per phase ${diagPer.join('/')}`, gp);
  return checks;
}

function goalProblems(goals, n) {
  const p = [];
  if (goals.length < n) p.push(`only ${goals.length} goals`);
  const g = goals.slice(0, n);
  if (g.some((x) => x.x === 0 && x.y === 0)) p.push('a goal is on the centre tile');
  if (g.some((x) => !(x.direction in DIR_ANGLE))) p.push('a goal has an unknown direction');
  if (n === TOTAL_ROUNDS) {
    const lo = Math.floor(n / 8), hi = Math.ceil(n / 8);
    for (const d of DIRECTIONS) {
      const k = g.filter((x) => x.direction === d).length;
      if (k < lo || k > hi) p.push(`${d} appears ${k} times (want ${lo} or ${hi})`);
    }
    for (const t of Object.values(TILE_NAME).filter((t) => t !== 'center')) {
      const k = g.filter((x) => x.tile === t).length;
      if (k < lo || k > hi) p.push(`tile ${t} appears ${k} times (want ${lo} or ${hi})`);
    }
  }
  return p;
}

function checkPair(mode, rounds, a, b, goals) {
  const p = [];
  if (!a || !b) p.push('both player IDs are needed');
  if (a === b) p.push('the two player IDs must differ');
  if (rounds.length < 1 || rounds.length > goals.length) p.push(`rounds must be 1–${goals.length}`);
  for (const rd of rounds) {
    const x = Object.values(rd.groups)[0];
    if (x.sender === x.receiver) p.push(`round ${rd.round}: sender and receiver are the same`);
    if (mode === 'pilot' && x.sender !== a) p.push(`round ${rd.round}: ${a} is not the sender`);
  }
  return [
    { name: mode === 'pilot' ? `Pilot: ${a} sends every round, ${b} receives every round` : `One-on-one: ${a} and ${b} alternate roles`, ok: !p.length, problems: p },
    { name: `Goals: first ${rounds.length} goals from goals.json, none on the centre tile`, ok: !goalProblems(goals, rounds.length).length, problems: goalProblems(goals, rounds.length) },
  ];
}

// ---------------------------------------------------------------- plan

// 'botsend': each human receiver (playerA, playerB) gets their own bot Sender every round, in a
// separate game, all on the same goal. Bots are named "Bot-<receiver>" and run by the server.
function botSendPlan(cfg, goals) {
  const humans = [cfg.playerA, cfg.playerB].map((x) => String(x || '').trim()).filter(Boolean);
  const n = Math.round(Number(cfg.rounds) || 8);
  const off = ((Math.round(Number(cfg.goalStart) || 1) - 1) % goals.length + goals.length) % goals.length;
  const rotated = goals.slice(off).concat(goals.slice(0, off));
  const bots = humans.map((h) => 'Bot-' + h);
  const schedule = Array.from({ length: n }, (_, i) => ({ round: i + 1, phase: 1,
    groups: Object.fromEntries(humans.map((h, k) => [h, { members: [bots[k], h], sender: bots[k], receiver: h, sitout: null }])) }));
  const p = [];
  if (!humans.length) p.push('at least one player ID is needed');
  if (new Set(humans).size !== humans.length) p.push('the player IDs must differ');
  if (n < 1 || n > goals.length) p.push(`rounds must be 1–${goals.length}`);
  const gp = goalProblems(rotated, n);
  return { mode: 'botsend', players: [...bots, ...humans], bots, originalGroup: {}, schedule, goals: rotated.slice(0, n), checks: [
    { name: `Bot sends: ${humans.map((h) => 'Bot-' + h + ' → ' + h).join(', ')} every round, same goal for everyone`, ok: !p.length, problems: p },
    { name: `Goals: goals ${off + 1}–${off + n} from goals.json (wrapping after the last goal), none on the centre tile`, ok: !gp.length, problems: gp },
  ] };
}

// cfg: { mode: 'tournament' } | { mode: 'pilot'|'oneonone', playerA, playerB, rounds }
function buildPlan(cfg) {
  const goals = loadGoals();
  const mode = cfg.mode || 'tournament';
  if (mode === 'tournament') {
    const players = tournamentPlayers(Math.round(Number(cfg.players) || DEFAULT_PLAYERS));
    const seed = Math.round(Number(cfg.seed) || 1);
    const schedule = tournamentSchedule(players, seed);
    const sgoals = sessionGoals(seed);
    return { mode, players, seed, originalGroup: {}, schedule, goals: sgoals, checks: checkTournament(schedule, sgoals, players) };
  }
  if (mode === 'botsend') return botSendPlan(cfg, goals);
  if (mode !== 'pilot' && mode !== 'oneonone') throw new Error('Unknown mode ' + mode);
  const a = String(cfg.playerA || '').trim(), b = String(cfg.playerB || '').trim();
  const n = Math.round(Number(cfg.rounds) || 8);
  const schedule = pairSchedule(mode, a, b, n);
  const originalGroup = {};
  for (const id of [a, b]) if (ORIGINAL_GROUP[id]) originalGroup[id] = ORIGINAL_GROUP[id];
  // cfg.goalStart (1-based, default 1): use goals from there on, wrapping round after goal 36 (for extra sessions)
  const off = ((Math.round(Number(cfg.goalStart) || 1) - 1) % goals.length + goals.length) % goals.length;
  const rotated = goals.slice(off).concat(goals.slice(0, off));
  const checks = checkPair(mode, schedule, a, b, rotated);
  if (off) checks[1].name = checks[1].name.replace('first ' + n + ' goals', `goals ${off + 1}–${off + n} (wrapping after the last goal)`);
  return { mode, players: [a, b], originalGroup, schedule, goals: rotated.slice(0, n), checks };
}

function scheduleTable(plan) {
  const groups = [...new Set(plan.schedule.flatMap((rd) => Object.keys(rd.groups)))];
  const lines = [['Round', 'Ph', 'Goal', ...groups.map((g) => `Group ${g}: sender → receiver (sits out)`)].join(' | ')];
  for (const rd of plan.schedule) {
    if (rd.messageBefore) lines.push(`— swap: "${rd.messageBefore}" —`);
    if (rd.sitout && (rd.round - 1) % ROUNDS_PER_PHASE === 0) lines.push(`— ${rd.sitout} sits out phase ${rd.phase} —`);
    const goal = plan.goals[rd.round - 1];
    lines.push([String(rd.round).padStart(5), ` ${rd.phase}`, `${goal.tile} ${goal.direction}`.padEnd(18),
      ...groups.map((g) => { const x = rd.groups[g]; return (x ? `${x.sender}${x.botSender ? ' (bot)' : ''} → ${x.receiver}${x.sitout ? ` (${x.sitout})` : ''}` : '—').padEnd(22); })].join(' | '));
  }
  return lines.join('\n');
}

module.exports = {
  buildPlan, scheduleTable, loadGoals, generateGoals, tileName, DIR_ANGLE, ANGLE_DIR,
  ORIGINAL_GROUP, CONFEDERATE, SWAP_MESSAGE, SWAP_DETAIL, SITOUT_DETAIL, ROUNDS_PER_PHASE, PHASES, TOTAL_ROUNDS, SWAP_ROUND, DEFAULT_PLAYERS,
};

// `node tournament.js [pilot|oneonone A B rounds]` prints the dry run and exits 1 if a check fails
if (require.main === module) {
  const [mode = 'tournament', playerA, playerB, rounds] = process.argv.slice(2);
  const plan = buildPlan({ mode, playerA, playerB, rounds });
  console.log(scheduleTable(plan));
  console.log('');
  for (const c of plan.checks) console.log(`${c.ok ? 'PASS' : 'FAIL'}  ${c.name}${c.ok ? '' : '\n      ' + c.problems.join('\n      ')}`);
  process.exit(plan.checks.every((c) => c.ok) ? 0 : 1);
}
