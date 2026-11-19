// Tacit Communication Game (TCG) engine with experimenter-controlled partner ladders.
//
// Two players share a 3x3 board. The Sender sees both players' goal configurations and
// can only communicate by moving their own token; the Receiver must infer where (and in
// which orientation) to place their token. Pairings follow a "ladder" schedule so that
// strategies can be traced as participants carry them from partner to partner.
//
// This file holds the game logic only. It keeps no state of its own between calls: every
// request loads the whole world (one JSON object), runs `step()` synchronously, and the
// caller saves the world back. That lets it run on stateless hosts (Vercel functions) as
// well as in the local server, with the same code. Timers are deadlines stored in the
// world and fired by the next request after they fall due (clients poll several times a
// second). Messages to players go into an outbox; data files are emitted as appends/puts.
//
// Data keep the original layout: <session>/roomNNNNNN/game_trial_<n>_<epoch>.json (read by
// the course's read_json_tcg.py), plus trials.csv, trials.jsonl, keypresses.csv,
// pairings.csv, rounds.csv, ladder.json, schedule.json, session.json.

const TOUR = require('./tournament');
const RESULTS = require('./results');

const SHAPE = { RECT: 1, CIRCLE: 2, TRIANGLE: 3 };
const COLORS = { 1: [49, 113, 166, 1], 2: [223, 129, 51, 1] };
// TCG_SPEED > 1 shortens the fixed display phases (used by the bot test)
const SPEED = Number(process.env.TCG_SPEED || 1);
const PHASE_MS = { roleassignment: 2000 / SPEED, tokenassignment: 1000 / SPEED, feedback: 2500 / SPEED };
// tournament modes keep the dead time short (~3 s per round instead of ~6.5 s): 1 s role intro, 1.5 s result
const TOUR_PHASE_MS = { roleassignment: 600 / SPEED, tokenassignment: 400 / SPEED, feedback: 1500 / SPEED };
const CELLS = [];
for (const y of [1, 0, -1]) for (const x of [-1, 0, 1]) CELLS.push([x, y]);
// a player counts as disconnected when they have not polled for this long
const STALE_MS = Number(process.env.TCG_STALE_MS || 5000);
// the world is checked at least this often while anyone is connected (disconnect detection)
const HEARTBEAT_MS = 2000;

// ---------------------------------------------------------------- utilities

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

function pick(arr, rand) { return arr[Math.floor(rand() * arr.length)]; }

function csvCell(v) {
  if (v === null || v === undefined) return '';
  const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function clone(o) { return JSON.parse(JSON.stringify(o)); }

// ---------------------------------------------------------------- ladder designs

// Each design returns a list of rounds; each round is a list of [idA, idB] pairs.
const DESIGNS = {
  // Diffusion chain: P1-P2, then P2-P3, then P3-P4, ... (one pair per round).
  chain(ids) {
    const rounds = [];
    for (let i = 0; i < ids.length - 1; i++) rounds.push([[ids[i], ids[i + 1]]]);
    return rounds;
  },
  // Everyone meets everyone once (circle method); n-1 rounds.
  roundrobin(ids) {
    const list = ids.slice();
    if (list.length % 2) list.push(null);
    const n = list.length, rounds = [];
    for (let r = 0; r < n - 1; r++) {
      const round = [];
      for (let i = 0; i < n / 2; i++) {
        const a = list[i], b = list[n - 1 - i];
        if (a && b) round.push([a, b]);
      }
      rounds.push(round);
      list.splice(1, 0, list.pop());
    }
    return rounds;
  },
  // Random perfect matchings each round, avoiding repeat partners when possible.
  random(ids, opts) {
    const rand = mulberry32(opts.seed);
    const met = new Set();
    const key = (a, b) => [a, b].sort().join('|');
    const rounds = [];
    for (let r = 0; r < opts.numRounds; r++) {
      let best = null, bestRepeats = Infinity;
      for (let attempt = 0; attempt < 200 && bestRepeats > 0; attempt++) {
        const s = shuffle(ids, rand), round = [];
        let repeats = 0;
        for (let i = 0; i + 1 < s.length; i += 2) {
          round.push([s[i], s[i + 1]]);
          if (met.has(key(s[i], s[i + 1]))) repeats++;
        }
        if (repeats < bestRepeats) { best = round; bestRepeats = repeats; }
      }
      best.forEach(([a, b]) => met.add(key(a, b)));
      rounds.push(best);
    }
    return rounds;
  },
  // Two isolated groups play internally, then are crossed: tests spread between communities.
  groups(ids, opts) {
    const half = Math.ceil(ids.length / 2);
    const A = ids.slice(0, half), B = ids.slice(half);
    const inner = Math.max(1, opts.numRounds - 1);
    const ra = DESIGNS.roundrobin(A).slice(0, inner), rb = DESIGNS.roundrobin(B).slice(0, inner);
    const rounds = [];
    for (let r = 0; r < Math.max(ra.length, rb.length); r++) rounds.push([...(ra[r] || []), ...(rb[r] || [])]);
    const cross = [];
    for (let i = 0; i < Math.min(A.length, B.length); i++) cross.push([A[i], B[i]]);
    rounds.push(cross);
    return rounds;
  },
};

// "A-B, C-D" per line -> rounds
function parseCustomRounds(text) {
  return text.split('\n').map((l) => l.trim()).filter(Boolean).map((line) =>
    line.split(/[,;]/).map((p) => p.trim()).filter(Boolean).map((p) => {
      const ids = p.split(/\s*[-–:\/]\s*|\s+/).filter(Boolean);
      if (ids.length !== 2) throw new Error(`Cannot parse pair "${p}" (use A-B)`);
      return ids;
    }));
}

// ---------------------------------------------------------------- trial generation

// Trial types (stored as Level, matching the original's column):
//  1  receiver = circle      -> only location must be communicated
//  2  receiver = rect/tri, sender = rect/tri -> location + orientation, sender can rotate
//  3  receiver = triangle, sender = circle  -> direction must be signalled without rotating (default)
function generateTrials(n, seed, types) {
  const rand = mulberry32(seed);
  const trials = [];
  let block = [];
  for (let i = 0; i < n; i++) {
    if (!block.length) block = shuffle(types, rand);
    const type = block.pop();
    const sShape = type === 3 ? SHAPE.CIRCLE : pick([SHAPE.RECT, SHAPE.TRIANGLE], rand);
    const rShape = type === 1 ? SHAPE.CIRCLE : type === 3 ? SHAPE.TRIANGLE : pick([SHAPE.RECT, SHAPE.TRIANGLE], rand);
    const [rx, ry] = pick(CELLS, rand);
    let sx, sy;
    do { [sx, sy] = pick(CELLS, rand); } while (sx === rx && sy === ry);
    const angleFor = (shape) => shape === SHAPE.CIRCLE ? 0
      : shape === SHAPE.RECT ? pick([0, 90], rand) : pick([0, 90, 180, 270], rand);
    trials.push({
      type,
      sender: { shape: sShape, xPos: sx, yPos: sy, angle: angleFor(sShape) },
      receiver: { shape: rShape, xPos: rx, yPos: ry, angle: angleFor(rShape) },
    });
  }
  return trials;
}

function orientationOk(tok) {
  const g = tok.goal;
  if (tok.shape === SHAPE.CIRCLE) return true;
  if (tok.shape === SHAPE.RECT) return tok.angle === g.angle || Math.abs(tok.angle - g.angle) === 180;
  return tok.angle === g.angle;
}
function locationOk(tok) { return tok.xPos === tok.goal.xPos && tok.yPos === tok.goal.yPos; }

// ---------------------------------------------------------------- world + request context
//
// W   the world (saved between requests):
//     session, rooms {key: room}, memberRoom {id: key}, adhoc {id: partner},
//     tour {matches {g: key}, timer, message}, online {id: true}, conns {id: token},
//     seq {id: n}, timers [{at, kind}], fileSeen {name: true}, sessionIds []
// C   the request context: now, outbox rows, file appends, file puts, log lines

let W = null, C = null;

function emptyWorld() {
  return { session: null, rooms: {}, memberRoom: {}, adhoc: {}, tour: { matches: {}, timer: null, message: null },
    online: {}, conns: {}, seq: {}, timers: [], fileSeen: {}, sessionIds: [], kicked: [], waiting: {} };
}

const now = () => C.now;
const clients = { has: (id) => !!W.online[id], keys: () => Object.keys(W.online) };
const log = (s) => C.logs.push(s);

function send(id, msg) {
  if (!W.online[id] || isBot(id)) return;
  const seq = W.seq[id] = (W.seq[id] || 0) + 1;
  C.out.push({ client: id, seq, msg });
}

// data files of the active session
function putFile(name, content) { C.puts.push({ session: W.session.id, name, content }); }
function appendFile(name, content) { C.appends.push({ session: W.session.id, name, content }); }
function appendCsv(name, header, row) {
  const k = W.session.id + '/' + name;
  const line = row.map(csvCell).join(',') + '\n';
  appendFile(name, (W.fileSeen[k] ? '' : header.join(',') + '\n') + line);
  W.fileSeen[k] = true;
}
function appendLines(name, header, text) {
  const k = W.session.id + '/' + name;
  appendFile(name, (W.fileSeen[k] ? '' : header + '\n') + text);
  W.fileSeen[k] = true;
}

// ---------------------------------------------------------------- session state

function saveSession() {
  if (!W.session) return;
  putFile('session.json', JSON.stringify(W.session, null, 2));
}

function newSessionId(base) {
  let sid = base;
  for (let i = 2; W.sessionIds.includes(sid); i++) sid = sid.replace(/(_\d+)?$/, `_${i}`);
  W.sessionIds.push(sid);
  return sid;
}

function stampNow() { return new Date(now()).toISOString().replace(/[:T]/g, '-').slice(0, 19); }

function createSession(cfg) {
  const ids = cfg.participants.map((s) => s.trim()).filter(Boolean);
  if (new Set(ids).size !== ids.length) throw new Error('Participant IDs must be unique');
  const seed = Number(cfg.seed) || Math.floor(Math.random() * 1e9);
  const numRounds = Number(cfg.numRounds) || 4;
  let rounds;
  if (cfg.design === 'custom') rounds = parseCustomRounds(cfg.custom || '');
  else if (DESIGNS[cfg.design]) rounds = DESIGNS[cfg.design](ids, { seed, numRounds });
  else throw new Error('Unknown design ' + cfg.design);
  if (cfg.design === 'custom') {
    for (const r of rounds) for (const pr of r) for (const id of pr) if (!ids.includes(id)) ids.push(id);
  }
  if (!rounds.length) throw new Error('Ladder has no rounds — need at least 2 participants');
  for (const r of rounds) {
    const seen = new Set();
    for (const [a, b] of r) {
      if (a === b) throw new Error(`${a} cannot be paired with themselves`);
      for (const id of [a, b]) {
        if (seen.has(id)) throw new Error(`${id} appears twice in one round`);
        seen.add(id);
      }
    }
  }
  const types = (cfg.trialTypes || [3]).map(Number).filter((t) => [1, 2, 3].includes(t));
  const sid = newSessionId(`${stampNow()}_${(cfg.name || 'session').replace(/[^\w-]+/g, '_')}`);
  W.session = {
    id: sid,
    name: cfg.name || 'session',
    design: cfg.design,
    seed,
    participants: ids,
    rounds,
    trialsPerPairing: Number(cfg.trialsPerPairing) || 18,
    turnMs: (Number(cfg.turnSeconds) || 10) * 1000,
    trialTypes: types.length ? types : [3],
    sameProblemsEachRound: !!cfg.sameProblemsEachRound,
    showGoalOnFeedback: !!cfg.showGoalOnFeedback,
    progress: Object.fromEntries(ids.map((id) => [id, 0])), // next round index per participant
    pairings: {}, // key -> record
    roomCounter: 0,
    createdAt: now(),
  };
  rounds.forEach((r, ri) => r.forEach(([a, b]) => {
    W.session.pairings[pairKey(ri, a, b)] = { round: ri + 1, p1: a, p2: b, status: 'pending', room: null, trialIndex: 0, successes: 0, trials: 0 };
  }));
  putFile('ladder.json', JSON.stringify({ design: W.session.design, participants: ids, rounds }, null, 2));
  saveSession();
  return W.session;
}

function pairKey(roundIdx, a, b) { return `r${roundIdx + 1}:${a}|${b}`; }

// The next scheduled pairing for a participant, skipping rounds where they sit out.
function nextPairingFor(id) {
  const session = W.session;
  if (!session || !(id in session.progress)) return null;
  for (let r = session.progress[id]; r < session.rounds.length; r++) {
    const pr = session.rounds[r].find(([a, b]) => a === id || b === id);
    if (pr) return { roundIdx: r, key: pairKey(r, pr[0], pr[1]), p1: pr[0], p2: pr[1], partner: pr[0] === id ? pr[1] : pr[0] };
  }
  return null;
}

// ---------------------------------------------------------------- lobby

function lobbyStatus(id) {
  if (W.rooms[W.memberRoom[id]]) return;
  const session = W.session;
  const next = nextPairingFor(id);
  if (W.adhoc[id]) {
    send(id, { type: 'lobby', state: 'waiting', partner: W.adhoc[id], round: 'free play' });
  } else if (!session || !(id in (session?.progress || {}))) {
    send(id, { type: 'lobby', state: 'unscheduled', message: 'Your ID is not on the current ladder. Enter a partner ID to play a free pairing.' });
  } else if (!next) {
    send(id, { type: 'lobby', state: 'done', message: 'You have finished all your pairings. Thank you!' });
  } else {
    // explain why we are waiting: partner offline, or partner still finishing an earlier pairing
    const partnerNext = nextPairingFor(next.partner);
    let detail = 'Waiting for your partner to connect…';
    if (clients.has(next.partner) && partnerNext && partnerNext.key !== next.key) detail = 'Your partner is finishing another pairing…';
    else if (clients.has(next.partner)) detail = 'Starting…';
    send(id, { type: 'lobby', state: 'waiting', partner: next.partner, round: next.roundIdx + 1, totalRounds: session.rounds.length, detail });
  }
}

function tryStartPairings() {
  if (tourState()) return;
  const session = W.session;
  if (session) {
    for (const id of clients.keys()) {
      if (W.memberRoom[id]) continue;
      const next = nextPairingFor(id);
      if (!next) continue;
      const other = next.partner;
      if (!clients.has(other) || W.memberRoom[other]) continue;
      const otherNext = nextPairingFor(other);
      if (!otherNext || otherNext.key !== next.key) continue;
      startRoom(next.key, next.p1, next.p2, next.roundIdx + 1, false);
    }
  }
  for (const [id, partner] of Object.entries(W.adhoc)) {
    if (W.adhoc[partner] === id && clients.has(partner) && !W.memberRoom[id] && !W.memberRoom[partner]) {
      delete W.adhoc[id]; delete W.adhoc[partner];
      const [a, b] = [id, partner].sort();
      startRoom(`free:${a}|${b}:${now()}`, a, b, 'free', true);
    }
  }
  for (const id of clients.keys()) lobbyStatus(id);
}

function startRoom(key, p1, p2, round, adhoc) {
  if (adhoc) {
    if (!W.session) {
      // free play without a ladder still needs somewhere to write data
      createSession({ name: 'freeplay', design: 'custom', participants: [], custom: `${p1}-${p2}` });
      W.session.pairings = {}; W.session.rounds = []; W.session.progress = {};
    }
    W.session.pairings[key] = { round: 'free', p1, p2, status: 'pending', room: null, trialIndex: 0, successes: 0, trials: 0, adhoc: true };
  }
  const rec = W.session.pairings[key];
  if (!rec.room) {
    rec.room = 'room' + String(++W.session.roomCounter).padStart(6, '0');
    rec.startedAt = now();
  }
  rec.status = 'playing';
  saveSession();
  const room = new Room(key, { recKey: key }, round);
  W.rooms[key] = room;
  W.memberRoom[p1] = key; W.memberRoom[p2] = key;
  room.begin();
}

// ---------------------------------------------------------------- game room
//
// Rooms are plain data in W.rooms; their methods come from these classes and are
// re-attached after each load (see attach()). A ladder room's record lives in
// session.pairings[recKey]; a tournament match keeps its own record in _rec.

class Room {
  // opts (tournament matches): trials, hidePartner, tour (colour by tourColorOf), label {trial, total}
  constructor(key, recRef, round, opts = {}) {
    this.kind = 'room';
    this.key = key;
    if (recRef.recKey) this.recKey = recRef.recKey; else this._rec = recRef.rec;
    this.round = round;
    this.opts = opts;
    const rec = this.rec;
    this.players = { 1: rec.p1, 2: rec.p2 };
    const session = W.session;
    const seedBase = session.seed + (session.sameProblemsEachRound || rec.adhoc ? 0 : 7919 * (typeof round === 'number' ? round : 0));
    this.trials = opts.trials || generateTrials(session.trialsPerPairing, seedBase, session.trialTypes);
    this.timer = null;
    this.paused = false;
    this.ready = null;
    this.t = null;
  }

  get rec() { return this.recKey ? W.session.pairings[this.recKey] : this._rec; }

  colorOf(id) { return this.opts.tour ? tourColorOf(id) : null; }

  setTimer(ms, kind, extra) { this.timer = { at: now() + ms, kind, ...extra }; }
  clearTimer() { this.timer = null; }

  // called when this room's timer falls due
  fire(tm) {
    if (tm.kind === 'phase') this.phase(tm.name);
    else if (tm.kind === 'timeout') this.act(tm.id, 'timeout');
    else if (tm.kind === 'afterFeedback') {
      if (this.idleTrials >= 2) this.askReady('The last trials timed out with no moves. Press Start when you are both back.');
      else this.startTrial();
    }
  }

  numOf(id) { return this.players[1] === id ? 1 : 2; }

  begin() {
    this.paused = false;
    for (const n of [1, 2]) {
      send(this.players[n], {
        type: 'paired', you: n, partner: this.players[3 - n], round: this.round,
        totalRounds: W.session.rounds.length, totalTrials: this.trials.length, room: this.rec.room,
      });
    }
    this.askReady();
  }

  // nothing runs until both players confirm they are at the keyboard
  askReady(reason) {
    this.clearTimer();
    this.ready = [];
    this.t = null;
    this.idleTrials = 0;
    for (const n of [1, 2]) send(this.players[n], { type: 'readyCheck', ready: [], reason: reason || null, trial: this.rec.trialIndex + 1, totalTrials: this.trials.length });
  }

  setReady(id) {
    if (!this.ready || this.paused) return;
    if (!this.ready.includes(id)) this.ready.push(id);
    for (const n of [1, 2]) send(this.players[n], { type: 'readyCheck', ready: [...this.ready], trial: this.rec.trialIndex + 1, totalTrials: this.trials.length });
    if (this.ready.length === 2) { this.ready = null; this.startTrial(); }
  }

  startTrial() {
    this.clearTimer();
    const idx = this.rec.trialIndex;
    if (idx >= this.trials.length) return this.finish();
    const spec = this.trials[idx];
    const tn = idx + 1;
    const senderNum = tn % 2 === 1 ? 1 : 2; // roles alternate every trial
    const mk = (n, role, goal) => ({
      player: n, time: W.session.turnMs, color: this.opts.tour ? this.colorOf(this.players[n]).rgb : COLORS[n], role, shape: goal.shape,
      xPos: goal.xPos, yPos: goal.yPos, angle: goal.angle, state: 'outside', goal: clone(goal),
      onTarget: 0, numTarget: 0, numMove: 0, control: [], timeout: [],
    });
    this.t = {
      n: tn, spec, senderNum, receiverNum: 3 - senderNum,
      tok: {
        [senderNum]: mk(senderNum, 'sender', spec.sender),
        [3 - senderNum]: mk(3 - senderNum, 'receiver', spec.receiver),
      },
      logs: {}, phase: null, phaseOnset: 0, started: false, path: [],
    };
    this.phase('roleassignment');
  }

  stage(step) { return { stage: 3, step, round: this.t.n }; }

  // write the record for an epoch in the original file layout
  writeEpoch(epoch) {
    putFile(`${this.rec.room}/game_trial_${this.t.n}_${epoch}.json`, JSON.stringify(this.t.logs[epoch], null, 4));
  }

  log(epoch, entry) { (this.t.logs[epoch] ||= []).push(entry); }

  phase(name) {
    const t = this.t;
    if (t.phase && t.phase !== 'feedback') this.writeEpoch(t.phase);
    t.phase = name;
    t.phaseOnset = now();
    t.started = false;
    const step = { roleassignment: 1, tokenassignment: 2, sender: 3, receiver: 4, feedback: 5 }[name];
    const base = { epoch: name, trial: t.n };

    if (name === 'roleassignment' || name === 'tokenassignment') {
      const entry = { ...base, p1: clone(t.tok[1]), p2: clone(t.tok[2]), player: this.players[1], stage: this.stage(step), timestamp: t.phaseOnset };
      this.log(name, entry);
      if (name === 'roleassignment') {
        const date = new Date(now()).toISOString();
        for (const n of [1, 2]) this.log(name, { ...base, Iamplayer: n, player: this.players[n], date, stage: this.stage(step), timestamp: t.phaseOnset });
      }
      this.broadcastState();
      this.setTimer((this.opts.tour ? TOUR_PHASE_MS : PHASE_MS)[name], 'phase', { name: name === 'roleassignment' ? 'tokenassignment' : 'sender' });
    } else if (name === 'sender' || name === 'receiver') {
      const num = name === 'sender' ? t.senderNum : t.receiverNum;
      const tok = t.tok[num];
      // as in the original, the key waits below the board until the player presses space
      Object.assign(tok, { xPos: 0, yPos: 0, angle: 0, state: 'outside', control: 'keyboard' });
      this.log(name, { ...base, player: this.players[num], stage: this.stage(step), timestamp: t.phaseOnset });
      if (this.opts.tour) {
        // tournament modes: the key is on the board (centre, pointing up) as soon as the turn starts
        tok.state = 'in-use';
        t.started = true;
        this.log(name, { token: clone(tok), action: 'start', player: this.players[num], stage: this.stage(step), timestamp: t.phaseOnset });
        t.path.push({ role: name, a: 'start', x: 0, y: 0, angle: 0, t: 0 });
      }
      this.broadcastState();
      t.turnMs = this.turnLength(name, this.players[num]);
      this.setTimer(t.turnMs, 'timeout', { id: this.players[num] });
      if (this.opts.tour && name === 'sender' && (this.botSender || isBot(this.players[num]) || isAutoSend(this.players[num]))) botTurn(this, name, num); // bots only send
    } else if (name === 'feedback') {
      // a key that never appeared on the board cannot be on its goal
      const ok = (tok) => tok.state !== 'outside' && locationOk(tok) && orientationOk(tok);
      t.success = (t.tok[t.senderNum].goal.free || ok(t.tok[t.senderNum])) && ok(t.tok[t.receiverNum]);
      for (const n of [1, 2]) t.tok[n].state = 'locked';
      this.log(name, { ...base, success: t.success, level: t.spec.type, p1: clone(t.tok[1]), p2: clone(t.tok[2]), player: this.players[1], stage: this.stage(step), timestamp: t.phaseOnset });
      this.writeEpoch('feedback');
      this.recordTrial();
      this.rec.trialIndex++; // committed: a disconnect during feedback will not replay this trial
      saveSession();
      this.broadcastState();
      this.idleTrials = t.anyInput ? 0 : (this.idleTrials || 0) + 1;
      t.logs = {}; // written out; keep the saved world small
      this.setTimer(this.opts.tour ? (W.session.showGoalOnFeedback ? 5000 / SPEED : TOUR_PHASE_MS.feedback) : PHASE_MS.feedback, 'afterFeedback');
    }
  }

  // Tournament modes: a player's first turn as Sender and first turn as Receiver get firstTurnMs
  // (set at setup; e.g. 30 s) so they can work out the controls; every other turn gets turnMs.
  turnLength(role, id) {
    const first = W.session.firstTurnMs;
    if (!this.opts.tour || !first || !this.rd) return W.session.turnMs;
    const earlier = tourState().schedule.slice(0, this.rd.round - 1).some((rd) => Object.values(rd.groups).some((g) => g[role] === id));
    return earlier ? W.session.turnMs : first;
  }

  act(id, action, fromBot) {
    const t = this.t;
    if (!t || this.paused || (t.phase !== 'sender' && t.phase !== 'receiver')) return;
    if (t.phase === 'sender' && isAutoSend(id) && !fromBot) return; // the bot sends for this player; their own keys are ignored
    const num = t.phase === 'sender' ? t.senderNum : t.receiverNum;
    if (this.players[num] !== id) return;
    const tok = t.tok[num];
    const ts = now();
    if (action !== 'timeout') t.anyInput = true;
    const entry = (a) => ({ token: clone(tok), action: a, player: id, stage: this.stage(t.phase === 'sender' ? 3 : 4), timestamp: ts });
    const finishTurn = (a) => {
      this.clearTimer();
      tok.timeout = ts - t.phaseOnset;
      if (!t.started) { this.log(t.phase, entry('start')); t.started = true; }
      this.log(t.phase, entry(a));
      t.path.push({ role: t.phase, a, x: tok.xPos, y: tok.yPos, angle: tok.angle, t: ts - t.phaseOnset });
      this.phase(t.phase === 'sender' ? 'receiver' : 'feedback');
    };
    if (action === 'stop' && tok.state === 'outside') action = 'start';
    // auto-placed key: ignore a Space pressed in the first 0.4 s before any move (e.g. left over from the last turn)
    if (action === 'stop' && this.opts.tour && !tok.numMove && ts - t.phaseOnset < 400) return;
    if (action === 'start') {
      if (tok.state !== 'outside') return;
      tok.state = 'in-use';
      t.started = true;
      this.log(t.phase, entry('start'));
      t.path.push({ role: t.phase, a: 'start', x: tok.xPos, y: tok.yPos, angle: tok.angle, t: ts - t.phaseOnset });
      return this.broadcastState();
    }
    if (action === 'stop' || action === 'timeout') return finishTurn(action);
    if (tok.state === 'outside') {
      tok.state = 'in-use';
      t.started = true;
      this.log(t.phase, entry('start'));
      t.path.push({ role: t.phase, a: 'start', x: tok.xPos, y: tok.yPos, angle: tok.angle, t: ts - t.phaseOnset });
      return this.broadcastState();
    }
    const d = { up: [0, 1], down: [0, -1], left: [-1, 0], right: [1, 0] }[action];
    const step = this.opts.tour ? 45 : 90; // tournament modes: 8 directions; ladder: 4
    const rot = { rotateleft: -step, rotateright: step }[action];
    if (!d && !rot) return;
    if (d) {
      const nx = tok.xPos + d[0], ny = tok.yPos + d[1];
      if (Math.abs(nx) > 1 || Math.abs(ny) > 1) return;
      tok.xPos = nx; tok.yPos = ny;
    } else {
      if (tok.shape === SHAPE.CIRCLE) return; // circles cannot rotate
      tok.angle = (tok.angle + rot + 360) % 360;
    }
    tok.numMove++;
    if (t.phase === 'sender') {
      const rg = t.tok[t.receiverNum].goal;
      tok.onTarget = tok.xPos === rg.xPos && tok.yPos === rg.yPos ? 1 : 0;
      if (tok.onTarget) tok.numTarget++;
    }
    this.log(t.phase, entry(action));
    t.path.push({ role: t.phase, a: action, x: tok.xPos, y: tok.yPos, angle: tok.angle, t: ts - t.phaseOnset });
    this.broadcastState();
  }

  // What each player is allowed to see: the receiver never sees goals.
  broadcastState() {
    const t = this.t;
    for (const n of [1, 2]) {
      if (this.botSender && this.players[n] === this.rd.groups[this.g].sender) continue; // bot-run sender: no screen
      const isSender = n === t.senderNum;
      const view = (m) => {
        const tok = t.tok[m];
        const v = { player: m, role: tok.role, shape: tok.shape, color: tok.color, state: tok.state, xPos: tok.xPos, yPos: tok.yPos, angle: tok.angle };
        if (this.opts.tour) v.colorName = this.colorOf(this.players[m]).name;
        if (!tok.goal.free && (isSender || (t.phase === 'feedback' && W.session.showGoalOnFeedback))) v.goal = tok.goal;
        return v;
      };
      const turnMs = t.turnMs || W.session.turnMs;
      const remaining = (t.phase === 'sender' || t.phase === 'receiver') ? Math.max(0, turnMs - (now() - t.phaseOnset)) : null;
      send(this.players[n], {
        type: 'state', phase: t.phase, trial: this.opts.label?.trial ?? t.n, totalTrials: this.opts.label?.total ?? this.trials.length, you: n,
        yourRole: isSender ? 'sender' : 'receiver', level: t.spec.type,
        tokens: { 1: view(1), 2: view(2) }, success: t.phase === 'feedback' ? t.success : null,
        solution: t.phase === 'feedback' && !!W.session.showGoalOnFeedback,
        botSendsForYou: isSender && isAutoSend(this.players[n]), // only this player is told
        remaining, turnMs, partner: this.opts.hidePartner ? null : this.players[3 - n], round: this.round,
        successes: this.rec.successes,
      });
    }
  }

  recordTrial() {
    const t = this.t, session = W.session;
    this.rec.trials = t.n;
    if (t.success) this.rec.successes++;
    const sTok = t.tok[t.senderNum], rTok = t.tok[t.receiverNum];
    const row = {
      session: session.id, design: session.design, room: this.rec.room, round: this.round,
      trial: t.n, level: t.spec.type,
      sender: this.players[t.senderNum], receiver: this.players[t.receiverNum],
      senderShape: sTok.shape, receiverShape: rTok.shape,
      senderGoal: [sTok.goal.xPos, sTok.goal.yPos, sTok.goal.angle],
      receiverGoal: [rTok.goal.xPos, rTok.goal.yPos, rTok.goal.angle],
      senderFinal: [sTok.xPos, sTok.yPos, sTok.angle],
      receiverFinal: [rTok.xPos, rTok.yPos, rTok.angle],
      success: t.success ? 1 : 0,
      senderLoc: locationOk(sTok) ? 1 : 0, senderOri: orientationOk(sTok) ? 1 : 0,
      receiverLoc: locationOk(rTok) ? 1 : 0, receiverOri: orientationOk(rTok) ? 1 : 0,
      senderMoves: sTok.numMove, receiverMoves: rTok.numMove, targetVisits: sTok.numTarget,
      senderPath: t.path.filter((p) => p.role === 'sender').map((p) => `${p.a}@${p.x},${p.y},${p.angle}:${p.t}`).join(' '),
      receiverPath: t.path.filter((p) => p.role === 'receiver').map((p) => `${p.a}@${p.x},${p.y},${p.angle}:${p.t}`).join(' '),
      timestamp: now(),
    };
    appendCsv('trials.csv', Object.keys(row), Object.values(row));
    appendFile('trials.jsonl', JSON.stringify({ ...row, path: t.path }) + '\n');
    // one row per key press (plus start/stop/timeout), written once per trial
    const kHdr = ['session', 'room', 'round', 'trial', 'sender', 'receiver', 'role', 'player', 'action', 't', 'x', 'y', 'angle'];
    const lines = t.path.map((p) => [session.id, this.rec.room, this.round, t.n, this.players[t.senderNum], this.players[t.receiverNum],
      p.role, this.players[p.role === 'sender' ? t.senderNum : t.receiverNum], p.a, p.t, p.x, p.y, p.angle].map(csvCell).join(',') + '\n').join('');
    if (lines) appendLines('keypresses.csv', kHdr.join(','), lines);
  }

  finish() {
    this.clearTimer();
    const rec = this.rec, session = W.session;
    rec.status = 'done';
    rec.finishedAt = now();
    if (!rec.adhoc) {
      for (const id of [rec.p1, rec.p2]) session.progress[id] = Math.max(session.progress[id], rec.round);
    }
    appendCsv('pairings.csv',
      ['session', 'design', 'round', 'room', 'p1', 'p2', 'trials', 'successes', 'startedAt', 'finishedAt'],
      [session.id, session.design, rec.round, rec.room, rec.p1, rec.p2, rec.trials, rec.successes, new Date(rec.startedAt).toISOString(), new Date(rec.finishedAt).toISOString()]);
    saveSession();
    delete W.rooms[this.key];
    for (const id of [rec.p1, rec.p2]) {
      delete W.memberRoom[id];
      send(id, { type: 'pairingDone', partner: id === rec.p1 ? rec.p2 : rec.p1, successes: rec.successes, trials: rec.trials });
    }
    W.timers.push({ at: now() + 3000 / SPEED, kind: 'tryStart' }); // short break before the next partner
  }

  // a player dropped: abandon the current trial and replay it once both are back
  pause(id) {
    this.clearTimer();
    this.paused = true;
    send(this.players[1] === id ? this.players[2] : this.players[1], { type: 'paused', message: 'Your partner disconnected. Waiting for them to return…' });
  }

  resumeIfReady() {
    if (this.paused && clients.has(this.players[1]) && clients.has(this.players[2])) this.begin();
  }
}

// ---------------------------------------------------------------- tournament / pilot / one-on-one
//
// Rounds are synchronised: every group plays one trial (one Sender, one Receiver) and the
// next round starts only when every group has finished. Players sitting out, and players
// whose group is done, see a neutral waiting screen and never the other pairs' boards.
// Schedule, goals and checks live in tournament.js.

const GROUP_COLOR = { A: { rgb: [214, 69, 65, 1], name: 'red' }, B: { rgb: [49, 113, 166, 1], name: 'blue' }, C: { rgb: [67, 160, 71, 1], name: 'green' } };
const PAIR_COLORS = [{ rgb: COLORS[1], name: 'blue' }, { rgb: COLORS[2], name: 'orange' }];
const BETWEEN_ROUNDS_MS = 500 / SPEED;
const MESSAGE_MS = 8000 / SPEED;
const ROUNDS_HEADER = ['session_id', 'round', 'phase', 'group', 'sender_id', 'receiver_id', 'sitting_out_id',
  'sender_original_group', 'receiver_original_group', 'goal_tile', 'goal_direction', 'sender_moves',
  'receiver_tile', 'receiver_direction', 'success', 'sender_phase_ms', 'receiver_phase_ms', 'attempt', 'room', 'timestamp',
  'degrees_of_separation']; // tournament: shortest path between the partners in the network of earlier phases (1 = played together; none = not connected)

const tourState = () => W.session && W.session.tournament;
const currentRound = () => tourState()?.schedule[tourState().roundIdx];
const tourMatch = (g) => W.rooms[W.tour.matches[g]];
const tourMatchCount = () => Object.keys(W.tour.matches).length;

function createTournamentSession(cfg) {
  const plan = TOUR.buildPlan(cfg);
  const failed = plan.checks.filter((c) => !c.ok);
  if (failed.length) throw new Error('Schedule checks failed:\n' + failed.map((c) => `${c.name}\n  ${c.problems.join('\n  ')}`).join('\n'));
  const sid = newSessionId(`${stampNow()}_${plan.mode}${cfg.name ? '_' + String(cfg.name).replace(/[^\w-]+/g, '_') : ''}`);
  W.session = {
    id: sid, name: cfg.name || plan.mode, design: plan.mode, seed: 0,
    participants: plan.players, rounds: [], trialsPerPairing: 1,
    turnMs: (Number(cfg.turnSeconds) || 9) * 1000, trialTypes: [3],
    firstTurnMs: (cfg.firstTurnSeconds === undefined ? 20 : Number(cfg.firstTurnSeconds) || 0) * 1000, // 0 = same as other turns
    sameProblemsEachRound: true,
    // show the solution (goal outline + final keys) after each round; default on only for one-on-one practice
    showGoalOnFeedback: cfg.showSolution !== undefined ? !!cfg.showSolution : plan.mode === 'oneonone',
    progress: {}, pairings: {}, roomCounter: 0, createdAt: now(),
    tournament: {
      mode: plan.mode, schedule: plan.schedule, goals: plan.goals, originalGroup: plan.originalGroup, seed: plan.seed ?? null,
      showPartnerGroupColor: cfg.showPartnerGroupColor !== false,
      autoStart: cfg.autoStart !== false, // start by itself once every player is connected and has pressed Ready
      ready: {},
      autoSend: (Array.isArray(cfg.autoSend) ? cfg.autoSend : String(cfg.autoSend || '').split(/[\s,]+/)).map((x) => String(x).trim()).filter((x) => plan.players.includes(x)),
      bots: [...new Set([...(plan.bots || []), ...(Array.isArray(cfg.bots) ? cfg.bots : String(cfg.bots || '').split(/[\s,]+/)).map((x) => String(x).trim())])].filter((x) => plan.players.includes(x)),
      roundIdx: 0, done: {}, attempts: {}, status: 'lobby', messagesShown: {},
    },
  };
  putFile('schedule.json', JSON.stringify({ mode: plan.mode, goals: plan.goals, schedule: plan.schedule, checks: plan.checks, bots: W.session.tournament.bots }, null, 2));
  W.timers = W.timers.filter((tm) => tm.kind !== 'bot');
  for (const id of W.session.tournament.bots) { W.online[id] = true; W.session.tournament.ready[id] = true; }
  saveSession();
  for (const id of clients.keys()) { send(id, { type: 'mode', mode: plan.mode }); tourWait(id); }
  tourMaybeAutoStart();
}

// Auto start: a session set up with autoStart begins once every player is connected and has
// pressed Ready on the waiting screen. (The dashboard's Start button skips the Ready check.)
function tourMaybeAutoStart() {
  const ts = tourState();
  if (!ts || !ts.autoStart || ts.status !== 'lobby') return;
  ts.ready ||= {};
  if (W.session.participants.some((id) => !clients.has(id) || !ts.ready[id])) return;
  log(`Auto start: all ${W.session.participants.length} players connected and ready`);
  tourControl('start');
}

// ---------------------------------------------------------------- bots (Dylan's sender strategy)
// A bot player (setup option `bots`, e.g. ["P7"]) is always connected and ready and plays its SENDER
// turns by itself (receiver turns are not automated: use the pilot mode so the bot always sends).
// botReceiverPlan is kept for the bot-vs-bot strategy test only.
//  Sender:   move the circle to the goal tile (vertical first), stay there ~1.5 s, back to the centre,
//            then wiggle quickly between the centre and the tile on the side the triangle must point
//            (the centre's neighbour that way; a corner for diagonals) 3 times (2 for diagonals), end.
//  Receiver: reads the same convention from the Sender's moves: the farthest tile before the Sender comes
//            back to the centre is the goal tile; the centre-neighbour where the Sender then turned back
//            or stopped is the direction. Then walks there (vertical first), turns, and ends the turn.
// Timing is human-like (0.8-1.2 s to begin, 0.35-0.6 s per key press, a pause between parts).
function isBot(id) { return !!W.session?.tournament?.bots?.includes(id); }
// a real player whose SENDER turns the bot plays (setup option `autoSend`); they receive normally
function isAutoSend(id) { return !!W.session?.tournament?.autoSend?.includes(id); }
const STEP = { up: [0, 1], down: [0, -1], left: [-1, 0], right: [1, 0] };
const OPP = { up: 'down', down: 'up', left: 'right', right: 'left' };
function pathBetween(x, y, gx, gy) {
  const s = [];
  while (y < gy) { s.push('up'); y++; } while (y > gy) { s.push('down'); y--; }
  while (x < gx) { s.push('right'); x++; } while (x > gx) { s.push('left'); x--; }
  return s;
}
function botSenderPlan(goal) {
  const to = pathBetween(0, 0, goal.xPos, goal.yPos);
  const back = to.slice().reverse().map((d) => OPP[d]);
  // the direction tile: the centre's neighbour on the side the triangle points (a corner for diagonals)
  const a = goal.angle * Math.PI / 180, dx = Math.round(Math.sin(a)), dy = Math.round(Math.cos(a));
  const poke = pathBetween(0, 0, dx, dy);
  const wiggle = [...poke, ...poke.slice().reverse().map((d) => OPP[d])];
  // the direction is a quick back-and-forth (3 times; 2 for diagonals, whose corner is 2 steps away)
  const reps = poke.length > 1 ? 2 : 3;
  return [to, { hold: 1500, moves: back }, { fast: true, moves: Array.from({ length: reps }, () => wiggle).flat() }];
}
// Reads the convention back: excursions out of the centre; the farthest tile of the first is the goal
// tile, the farthest tile of the last later one is the direction (angle of that tile from the centre).
function botDecode(path) {
  const pos = path.filter((p) => p.role === 'sender' && STEP[p.a]).map((p) => [p.x, p.y]);
  const dist = ([x, y]) => Math.abs(x) + Math.abs(y);
  const trips = [];
  let cur = null;
  for (const p of pos) {
    if (dist(p) === 0) { if (cur) trips.push(cur); cur = null; continue; }
    if (!cur || dist(p) > dist(cur)) cur = p;
  }
  if (cur) trips.push(cur);
  const tile = trips[0] || [0, 0];
  const d = trips.length > 1 ? trips[trips.length - 1] : null;
  const angle = d ? ((Math.round(Math.atan2(d[0], d[1]) * 180 / Math.PI / 45) * 45) + 360) % 360 : 0;
  return { tile, angle };
}
function botReceiverPlan(path) {
  const { tile, angle } = botDecode(path);
  const k = angle / 45;
  const turn = k <= 4 ? Array(k).fill('rotateright') : Array(8 - k).fill('rotateleft');
  return [pathBetween(0, 0, tile[0], tile[1]), turn];
}
function botTurn(room, phase, num) {
  const t = room.t, id = room.players[num];
  const parts = phase === 'sender' ? botSenderPlan(t.tok[t.receiverNum].goal) : botReceiverPlan(t.path);
  const r = (a, b) => Math.round(a + Math.random() * (b - a)); // whole ms: the Postgres store keeps times as bigint
  let at = now() + r(800, 1200);
  const base = { kind: 'bot', id, room: room.key, trial: t.n, phase };
  parts.forEach((part, k) => {
    const moves = Array.isArray(part) ? part : part.moves, fast = !Array.isArray(part) && part.fast;
    if (k && moves.length) at += part.hold ? r(part.hold, part.hold + 300) : r(500, 800); // pause (or a long hold) between parts
    for (const action of moves) { W.timers.push({ ...base, at, action }); at += fast ? r(170, 240) : r(350, 600); }
  });
  W.timers.push({ ...base, at: at + r(500, 900), action: 'stop' });
}
function botAct(tm) {
  const room = W.rooms[tm.room];
  if (room && room.t && room.t.n === tm.trial && room.t.phase === tm.phase) room.act(tm.id, tm.action, true);
}

// One emailed link for everyone: players type their name; each new device gets a random free slot
// of the active session and the name is stored with it (ts.names, session.json, assignments.csv).
// The same device always gets its slot back; the same name on a new device takes back its slot if
// that slot is offline (switched browser). The confederate slot (player A in the pilot) is never
// handed out: the pilot confederate opens the link with ?id=<player A>.
// Before any session exists, devices that entered a name are listed as waiting (W.waiting) so the
// experimenter can see who is in the lobby and size the session to them.
const WAITING_MS = 90000; // background tabs may only check in once a minute; a closed page removes itself at once (leave)
function reservedSlots(ts) {
  if (ts.mode === 'pilot') return [W.session.participants[0]];
  return [];
}
function cleanName(n) { return String(n || '').replace(/[\u0000-\u001f<>]/g, '').replace(/\s+/g, ' ').trim().slice(0, 40); }
function isOnline(id, seen) { return !!(W.online[id] && seen[id] && now() - seen[id] < STALE_MS); }
function giveSlot(ts, device, id, name, why) {
  for (const [d, x] of Object.entries(ts.assigned)) if (x === id) delete ts.assigned[d];
  ts.assigned[device] = id;
  (ts.names ||= {})[id] = name;
  appendCsv('assignments.csv', ['session_id', 'player_id', 'device', 'timestamp', 'name'], [W.session.id, id, device, now(), name]);
  saveSession();
  log(`${why} ${id} = "${name}"`);
  if (W.online[id]) send(id, { type: 'name', name });
  return { code: 200, body: { id, name } };
}
function assignSlot(device, rawName, seen, leave) {
  const ts = tourState(), name = cleanName(rawName);
  if (!device) return { code: 400, body: { error: 'device id required' } };
  W.waiting ||= {};
  if (leave) { delete W.waiting[device]; return { code: 200, body: { left: true } }; }
  for (const [d, x] of Object.entries(W.waiting)) if (now() - x.at > WAITING_MS) delete W.waiting[d];
  if (!ts) {
    if (W.lobbyBlocked?.[device]) return { code: 403, body: { error: 'The experimenter has removed you from the lobby.' } };
    const dup = name && Object.entries(W.waiting).find(([d, x]) => d !== device && x.name.toLowerCase() === name.toLowerCase());
    if (dup) return { code: 409, body: { error: `Someone called "${name}" is already in the lobby. Please add your last name or initial.`, needName: true } };
    if (name) W.waiting[device] = { name, at: now(), first: W.waiting[device]?.first || now() };
    return { code: 404, body: { error: 'The game has not opened yet. Keep this page open; you will join as soon as it does.', retry: true, waiting: true } };
  }
  delete W.waiting[device];
  ts.assigned ||= {}; ts.names ||= {};
  if (ts.blocked?.[device]) return { code: 403, body: { error: 'The experimenter has removed you from this session.' } };
  const mine = ts.assigned[device];
  if (mine) {
    if (name && ts.names[mine] !== name) return giveSlot(ts, device, mine, name, 'Renamed');
    return { code: 200, body: { id: mine, name: ts.names[mine] || name } };
  }
  if (!name) return { code: 400, body: { error: 'Please enter your name.', needName: true } };
  const same = Object.keys(ts.names).find((id) => ts.names[id].toLowerCase() === name.toLowerCase());
  if (same) {
    if (isOnline(same, seen)) return { code: 409, body: { error: `Someone called "${name}" has already joined. Please add your last name or initial.`, needName: true } };
    return giveSlot(ts, device, same, name, 'Rejoined (new device)');
  }
  if (ts.locked) return { code: 403, body: { error: 'This session is closed to new players. Please contact the experimenter.' } };
  const taken = new Set([...Object.values(ts.assigned), ...reservedSlots(ts), ...(ts.bots || [])]);
  const free = W.session.participants.filter((id) => !taken.has(id) && !isOnline(id, seen));
  if (!free.length) return { code: 409, body: { error: 'This session is full. Please contact the experimenter.' } };
  return giveSlot(ts, device, free[Math.floor(Math.random() * free.length)], name, `Assigned (${free.length - 1} slots left)`);
}

// disconnect one player's window (the page stops and shows "Join again")
function dropPlayer(id) {
  if (!W.online[id] || isBot(id)) return;
  if (W.conns[id]) W.kicked = [...(W.kicked || []), W.conns[id]].slice(-50);
  onLeave(id);
}

function tourReady(id) {
  const ts = tourState();
  if (!ts || ts.status !== 'lobby' || !W.session.participants.includes(id)) return;
  (ts.ready ||= {})[id] = true;
  saveSession();
  W.session.participants.forEach((p) => clients.has(p) && tourWait(p));
  tourMaybeAutoStart();
}

// Token colour: original group colour in the tournament (kept after the swap), otherwise a
// fixed colour per player. Nothing else about the partner is shown.
function tourResults() {
  const ts = tourState();
  return RESULTS.computeResults(ts.log || [], W.session.participants.filter((id) => !isBot(id)), ts.names || {});
}

function tourColorOf(id) {
  const ts = tourState();
  if (ts.mode === 'tournament') { // fixed pairs: the first member blue, the second orange (stable while roles alternate)
    const rd = currentRound(), x = rd && Object.values(rd.groups).find((g) => g.sender === id || g.receiver === id);
    return PAIR_COLORS[x && x.members[1] === id ? 1 : 0];
  }
  if (ts.mode === 'botsend') { // per game: the Sender blue, the Receiver orange
    const rd = currentRound(), x = rd && Object.values(rd.groups).find((g) => g.sender === id || g.receiver === id);
    return PAIR_COLORS[x && x.receiver === id ? 1 : 0];
  }
  return PAIR_COLORS[W.session.participants.indexOf(id) === 1 ? 1 : 0];
}

function tourWait(id, title) {
  const ts = tourState();
  if (!ts) return;
  if (!W.session.participants.includes(id)) return send(id, { type: 'wait', title: 'Your ID is not in this session.', detail: 'Check the ID your experimenter gave you.' });
  if (!title && ts.status === 'lobby') {
    // waiting screen before the start: every player presses Ready (counts only, no names)
    const ready = ts.ready || {}, ps = W.session.participants;
    const notReady = ps.filter((p) => !ready[p] || !clients.has(p)).length;
    if (!ready[id]) return send(id, { type: 'wait', title: 'Press Ready when you are ready to start', detail: 'Read "How to play" below first.', canReady: true });
    return send(id, { type: 'wait', title: notReady ? `You are ready. Waiting for ${notReady === 1 ? 'the other player' : notReady + ' other players'}…` : 'Starting…', detail: '' });
  }
  const sitting = currentRound()?.sitout === id; // odd number of players: this person waits the whole phase
  if (!title && ts.status !== 'done' && W.tour.message) {
    // the partner-change announcement: a full, distinct screen
    if (sitting && W.tour.message === TOUR.SWAP_MESSAGE) return send(id, { type: 'wait', title: 'You sit out the next 10 rounds', detail: TOUR.SITOUT_DETAIL, announce: true });
    return send(id, { type: 'wait', title: W.tour.message, detail: W.tour.message === TOUR.SWAP_MESSAGE ? TOUR.SWAP_DETAIL : '', announce: true });
  }
  if (!title && sitting && ['running', 'pausing', 'paused'].includes(ts.status)) {
    const last = currentRound().phase === TOUR.PHASES;
    return send(id, { type: 'wait', title: 'You are sitting out these 10 rounds',
      detail: 'There is an odd number of players, so one person waits each part. ' + (last ? 'Please keep this page open until the game ends.' : 'Please keep this page open; you will get a partner in the next part.') });
  }
  if (!title && ts.status === 'done' && ts.mode === 'tournament' && ts.log?.length) {
    const res = tourResults();
    if (res.players[id]) return send(id, { type: 'results', me: res.players[id], leaderboard: res.leaderboard, awards: res.awards });
  }
  if (!title) {
    if (ts.status === 'done') title = 'All done. Thank you for playing!';
    else title = 'Waiting for next round…';
  }
  send(id, { type: 'wait', title, detail: '' });
}

class TourMatch extends Room {
  constructor(g, rd, attempt) {
    const ts = tourState();
    const x = rd.groups[g], goal = ts.goals[rd.round - 1];
    const rec = {
      round: rd.round, p1: x.sender, p2: x.receiver, status: 'playing', group: g, attempt,
      room: 'room' + String(++W.session.roomCounter).padStart(6, '0'), trialIndex: 0, successes: 0, trials: 0, startedAt: now(),
    };
    const spec = {
      type: 3,
      sender: { shape: SHAPE.CIRCLE, xPos: 0, yPos: 0, angle: 0, free: true }, // the sender has no target of their own
      receiver: { shape: SHAPE.TRIANGLE, xPos: goal.x, yPos: goal.y, angle: TOUR.DIR_ANGLE[goal.direction] },
    };
    super(`t:r${rd.round}:${g}:${attempt}`, { rec }, rd.round, {
      trials: [spec], hidePartner: true, tour: true, label: { trial: rd.round, total: ts.schedule.length },
    });
    this.kind = 'tour';
    this.g = g; this.rd = rd; this.goal = goal;
    this.botSender = !!x.botSender; // phase 1 group C: P7's sends are played by the bot in two games at once
  }

  begin() {
    const total = tourState().schedule.length;
    for (const n of [1, 2]) if (!(this.botSender && this.players[n] === this.rd.groups[this.g].sender))
      send(this.players[n], { type: 'paired', you: n, partner: null, hidePartner: true, round: this.rd.round, totalRounds: total, room: this.rec.room,
        ...(tourState().mode === 'tournament' ? { phase: this.rd.phase, phases: TOUR.PHASES, phaseRound: (this.rd.round - 1) % TOUR.ROUNDS_PER_PHASE + 1, phaseRounds: TOUR.ROUNDS_PER_PHASE } : {}) });
    this.startTrial();
  }

  // writes trials.csv / trials.jsonl / keypresses.csv / game_trial JSON as usual, then one row of rounds.csv
  recordTrial() {
    super.recordTrial();
    const ts = tourState(), t = this.t, x = this.rd.groups[this.g];
    const sTok = t.tok[t.senderNum], rTok = t.tok[t.receiverNum];
    const placed = rTok.state !== 'outside';
    appendCsv('rounds.csv', ROUNDS_HEADER, [
      W.session.id, this.rd.round, this.rd.phase, this.g, x.sender, x.receiver, x.sitout || this.rd.sitout || '',
      ts.originalGroup[x.sender] || '', ts.originalGroup[x.receiver] || '',
      this.goal.tile, this.goal.direction,
      t.path.filter((p) => p.role === 'sender').map((p) => `${p.a}(${p.x},${p.y})@${p.t}`).join(' '),
      placed ? TOUR.tileName(rTok.xPos, rTok.yPos) : 'none', placed ? TOUR.ANGLE_DIR[rTok.angle] : 'none',
      t.success ? 1 : 0, sTok.timeout, rTok.timeout, this.rec.attempt, this.rec.room, now(),
      x.separation === undefined || this.rd.phase === 1 ? '' : x.separation === null ? 'none' : x.separation,
    ]);
    // compact per-game record for the end-of-game results screen (leaderboard, play style, awards)
    const moves = t.path.filter((p) => p.role === 'sender').map((p) => `${p.a}(${p.x},${p.y})@${p.t}`).join(' ');
    const c = RESULTS.codeTurn(moves, this.goal.tile, this.goal.direction);
    (ts.log ||= []).push({ round: this.rd.round, g: this.g, s: x.sender, r: x.receiver, ok: !!t.success, code: c.code, maxWiggle: c.maxWiggle, rms: rTok.timeout || 0 });
  }

  finish() {
    this.clearTimer();
    this.rec.status = 'done';
    this.release();
    tourMatchDone(this);
  }

  release() {
    delete W.rooms[this.key];
    for (const n of [1, 2]) if (W.memberRoom[this.players[n]] === this.key) delete W.memberRoom[this.players[n]];
    if (W.tour.matches[this.g] === this.key) delete W.tour.matches[this.g];
  }

  // a player dropped: the unfinished trial is discarded and replayed (attempt + 1) when both are back.
  // During feedback the trial is already logged, so it simply counts as finished.
  pause(id) {
    if (this.t?.phase === 'feedback') return this.finish();
    this.clearTimer();
    this.release();
    const other = this.players[1] === id ? this.players[2] : this.players[1];
    tourWait(other, 'Your partner disconnected. This round will restart when they are back.');
  }

  resumeIfReady() {}
}

function tourLaunch(g) {
  const ts = tourState(), rd = currentRound();
  if (!rd || !['running', 'pausing'].includes(ts.status) || W.tour.timer) return;
  if (ts.done[rd.round]?.[g] || W.tour.matches[g]) return;
  const x = rd.groups[g];
  if ((!x.botSender && !clients.has(x.sender)) || !clients.has(x.receiver)) return;
  const att = (ts.attempts[rd.round] ||= {});
  att[g] = (att[g] || 0) + 1;
  const m = new TourMatch(g, rd, att[g]);
  W.tour.matches[g] = m.key;
  W.rooms[m.key] = m;
  if (!x.botSender) W.memberRoom[x.sender] = m.key;
  W.memberRoom[x.receiver] = m.key;
  saveSession();
  m.begin();
}

function tourStartRound() {
  const wasMessage = !!W.tour.message;
  W.tour.timer = null; W.tour.message = null;
  const ts = tourState(), rd = currentRound();
  if (!rd) return;
  if (ts.status === 'pausing') ts.status = 'paused';
  if (ts.status === 'paused' && wasMessage) delete ts.messagesShown[rd.round]; // show it again on resume
  if (ts.status !== 'running') { saveSession(); W.session.participants.forEach((id) => tourWait(id)); return; }
  // the neutral message before the swap, shown to everyone at once
  if (rd.messageBefore && !ts.messagesShown[rd.round]) {
    ts.messagesShown[rd.round] = true; saveSession();
    W.tour.message = rd.messageBefore;
    W.session.participants.forEach((id) => tourWait(id));
    W.tour.timer = { at: now() + MESSAGE_MS, kind: 'startRound' };
    return;
  }
  for (const g of Object.keys(rd.groups)) tourLaunch(g);
  for (const id of W.session.participants) if (!W.memberRoom[id]) tourWait(id);
}

function tourMatchDone(m) {
  const ts = tourState();
  (ts.done[m.rd.round] ||= {})[m.g] = true;
  saveSession();
  for (const n of [1, 2]) tourWait(m.players[n]);
  const rd = currentRound();
  if (rd && rd.round === m.rd.round && Object.keys(rd.groups).every((g) => ts.done[rd.round]?.[g])) {
    ts.roundIdx++;
    if (ts.roundIdx >= ts.schedule.length) ts.status = 'done';
    saveSession();
    if (ts.status === 'done') W.session.participants.forEach((id) => tourWait(id));
    else W.tour.timer = { at: now() + BETWEEN_ROUNDS_MS, kind: 'startRound' };
  }
  // pausing with nothing left running (e.g. the last group is waiting for a reconnect): pause now
  if (ts.status === 'pausing' && !tourMatchCount() && !W.tour.timer) return tourStartRound();
}

function tourOnJoin(id) {
  send(id, { type: 'mode', mode: tourState().mode });
  const rd = currentRound();
  if (rd) for (const g of Object.keys(rd.groups)) if (rd.groups[g].sender === id || rd.groups[g].receiver === id) tourLaunch(g);
  if (!W.memberRoom[id]) tourWait(id);
  tourMaybeAutoStart();
}

function tourControl(action) {
  const ts = tourState();
  if (!ts) throw new Error('No tournament session. Set one up first.');
  if (action === 'start') {
    if (ts.status !== 'lobby') throw new Error(`Already ${ts.status}.`);
    const missing = W.session.participants.filter((id) => !clients.has(id));
    if (missing.length) throw new Error('Not connected yet: ' + missing.join(', '));
    ts.status = 'running'; ts.startedAt = now();
    tourStartRound();
  } else if (action === 'pause') {
    if (ts.status !== 'running') throw new Error(`Cannot pause while ${ts.status}.`);
    ts.status = 'pausing'; // takes effect when the current round ends, so no trial is cut off
    if (!tourMatchCount() && !W.tour.timer) tourStartRound();
  } else if (action === 'resume') {
    if (ts.status === 'pausing') ts.status = 'running';
    else if (ts.status === 'paused') { ts.status = 'running'; tourStartRound(); }
    else throw new Error(`Cannot resume while ${ts.status}.`);
  } else throw new Error('Unknown action ' + action);
  saveSession();
}

function tourAdminSnapshot() {
  const ts = tourState(), rd = currentRound();
  const players = W.session.participants.map((id) => ({ id, name: ts.names?.[id] || '', online: clients.has(id), ready: !!ts.ready?.[id], bot: isBot(id), group: ts.originalGroup[id] || '', playing: !!W.memberRoom[id] }));
  const groups = rd ? Object.entries(rd.groups).map(([g, x]) => {
    const m = tourMatch(g);
    let state = ts.done[rd.round]?.[g] ? 'done' : m ? (m.t?.phase || 'starting') : 'waiting';
    if (state === 'waiting' && ['running', 'pausing'].includes(ts.status) && !W.tour.timer) {
      const off = [x.sender, x.receiver].filter((id) => !clients.has(id));
      if (off.length) state = `waiting for ${off.join(', ')} to reconnect`;
    }
    return { g, sender: x.sender, receiver: x.receiver, sitout: x.sitout, state, attempt: ts.attempts[rd.round]?.[g] || 0 };
  }) : [];
  return {
    type: 'admin', tournament: true,
    session: { id: W.session.id, name: W.session.name, design: ts.mode, turnMs: W.session.turnMs },
    status: ts.status, message: W.tour.message, round: rd?.round || null, phase: rd?.phase || null,
    totalRounds: ts.schedule.length, groups, players, schedule: ts.schedule, goals: ts.goals,
    showPartnerGroupColor: ts.showPartnerGroupColor, autoStart: !!ts.autoStart, seed: ts.seed ?? null, locked: !!ts.locked,
    waiting: waitingList(),
    results: ts.status === 'done' && ts.log?.length ? tourResults() : null,
  };
}

// ---------------------------------------------------------------- admin

function waitingEntries() {
  return Object.entries(W.waiting || {}).filter(([, x]) => now() - x.at <= WAITING_MS).sort((a, b) => (a[1].first || 0) - (b[1].first || 0));
}
function waitingList() { return waitingEntries().map(([, x]) => x.name); }
function waitingDetail() { return waitingEntries().map(([key, x]) => ({ key, name: x.name })); }

function adminSnapshot() {
  const session = W.session;
  if (!session) return { type: 'admin', session: null, waiting: waitingList(), waitingDetail: waitingDetail() };
  if (tourState()) return tourAdminSnapshot();
  const online = clients.keys();
  const participants = session.participants.map((id) => {
    const next = nextPairingFor(id);
    return { id, online: clients.has(id), inRoom: !!W.memberRoom[id], nextRound: next ? next.roundIdx + 1 : null, nextPartner: next?.partner || null };
  });
  const live = {};
  for (const [k, r] of Object.entries(W.rooms)) live[k] = { trial: r.t?.n ?? r.rec.trialIndex + 1, phase: r.ready ? 'waiting for ready' : r.t?.phase, paused: r.paused };
  return {
    type: 'admin',
    session: { id: session.id, name: session.name, design: session.design, rounds: session.rounds, trialsPerPairing: session.trialsPerPairing, turnMs: session.turnMs },
    pairings: session.pairings, participants, online, live,
  };
}

// admin routes; throws on bad input (the caller turns that into a 400)
function adminRoute(route, body) {
  if (route === 'session') {
    if (Object.keys(W.rooms).length) return { code: 409, body: { error: 'Pairings are in progress; wait until they finish.' } };
    W.tour.timer = null; W.tour.message = null;
    createSession(body);
    for (const id of clients.keys()) send(id, { type: 'mode', mode: 'ladder' });
    tryStartPairings();
    return { code: 200, body: adminSnapshot() };
  }
  if (route === 'preview') {
    const ids = body.participants.map((s) => s.trim()).filter(Boolean);
    const opts = { seed: Number(body.seed) || 1, numRounds: Number(body.numRounds) || 4 };
    const rounds = body.design === 'custom' ? parseCustomRounds(body.custom || '') : DESIGNS[body.design](ids, opts);
    return { code: 200, body: { rounds } };
  }
  if (route === 'status') return { code: 200, body: adminSnapshot() };
  if (route === 'tournament/dryrun') {
    const plan = TOUR.buildPlan(body);
    return { code: 200, body: { ...plan, table: TOUR.scheduleTable(plan), ok: plan.checks.every((c) => c.ok) } };
  }
  if (route === 'tournament/setup') {
    if (Object.keys(W.rooms).length) return { code: 409, body: { error: 'Games are in progress; wait until they finish.' } };
    createTournamentSession(body);
    return { code: 200, body: adminSnapshot() };
  }
  if (route === 'tournament/kickall') {
    // free every slot: forget which device and name has which slot and disconnect all real players;
    // their pages stop, and opening the link again assigns slots afresh
    const ts = tourState();
    if (!ts) return { code: 400, body: { error: 'No session.' } };
    ts.assigned = {}; ts.names = {};
    ts.ready = Object.fromEntries((ts.bots || []).map((id) => [id, true]));
    for (const id of Object.keys(W.online)) dropPlayer(id);
    saveSession();
    log('Kicked everyone; all slots free');
    return { code: 200, body: adminSnapshot() };
  }
  if (route === 'tournament/kick') {
    // remove one player: free their slot, disconnect them and refuse their device for this session
    const ts = tourState(), id = String(body.id || '');
    if (!ts || !W.session.participants.includes(id)) return { code: 400, body: { error: 'No such player.' } };
    ts.blocked ||= {};
    for (const [d, x] of Object.entries(ts.assigned || {})) if (x === id) { delete ts.assigned[d]; ts.blocked[d] = true; }
    const name = ts.names?.[id];
    if (ts.names) delete ts.names[id];
    if (ts.ready && !isBot(id)) delete ts.ready[id];
    dropPlayer(id);
    W.session.participants.forEach((p) => clients.has(p) && ts.status === 'lobby' && tourWait(p));
    saveSession();
    log(`Kicked ${id}${name ? ` ("${name}")` : ''}; slot free`);
    return { code: 200, body: adminSnapshot() };
  }
  if (route === 'tournament/rename') {
    const ts = tourState(), id = String(body.id || ''), name = cleanName(body.name);
    if (!ts || !W.session.participants.includes(id)) return { code: 400, body: { error: 'No such player.' } };
    (ts.names ||= {})[id] = name;
    if (!name) delete ts.names[id];
    send(id, { type: 'name', name: name || id });
    saveSession();
    return { code: 200, body: adminSnapshot() };
  }
  if (route === 'tournament/lock') {
    const ts = tourState();
    if (!ts) return { code: 400, body: { error: 'No session.' } };
    ts.locked = !!body.locked;
    saveSession();
    return { code: 200, body: adminSnapshot() };
  }
  if (route === 'tournament/results') {
    const ts = tourState();
    if (!ts || ts.status !== 'done') return { code: 400, body: { error: 'The game has not finished yet.' } };
    if (Array.isArray(body.log) && body.log.length) ts.log = body.log;
    saveSession();
    W.session.participants.forEach((id) => clients.has(id) && tourWait(id));
    const res = tourResults();
    return { code: 200, body: { ...adminSnapshot(), results: res } };
  }
  if (route === 'waiting/clear') { W.waiting = {}; return { code: 200, body: adminSnapshot() }; }
  if (route === 'lobby/kick') {
    // remove someone from the open lobby; their page is told and cannot come back until the next lobby
    const key = String(body.key || '');
    const name = W.waiting?.[key]?.name;
    if (!name) return { code: 400, body: { error: 'That person is no longer in the lobby.' } };
    delete W.waiting[key];
    (W.lobbyBlocked ||= {})[key] = true;
    log(`Removed "${name}" from the lobby`);
    return { code: 200, body: adminSnapshot() };
  }
  if (route === 'lobby/open') {
    // end the current session (its data stay saved) so people who open the link wait in the lobby by name
    if (Object.keys(W.rooms).length) return { code: 409, body: { error: 'A game is being played right now. Pause it and wait for the round to end first.' } };
    for (const id of Object.keys(W.online)) dropPlayer(id);
    if (W.session) log(`Lobby opened; session ${W.session.id} closed`);
    W.session = null; W.memberRoom = {}; W.tour = { matches: {}, timer: null, message: null }; W.lobbyBlocked = {};
    W.timers = W.timers.filter((tm) => tm.kind !== 'bot');
    return { code: 200, body: adminSnapshot() };
  }
  if (route === 'lobby/start') {
    // build the tournament for exactly the people waiting in the lobby, give each a slot (random order),
    // and start as soon as all of them are connected (they read the rules while waiting, so no Ready step)
    if (W.session) return { code: 409, body: { error: 'Open a new lobby first.' } };
    const entries = waitingEntries();
    if (entries.length < 2) return { code: 400, body: { error: 'Need at least 2 people in the lobby.' } };
    createTournamentSession({ ...body, mode: 'tournament', players: entries.length, autoStart: true });
    const ts = tourState();
    ts.assigned = {}; ts.names = {}; ts.fromLobby = true; ts.blocked = { ...(W.lobbyBlocked || {}) };
    const ids = [...W.session.participants];
    for (let i = ids.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [ids[i], ids[j]] = [ids[j], ids[i]]; }
    entries.forEach(([device, x], i) => {
      ts.assigned[device] = ids[i]; ts.names[ids[i]] = x.name; ts.ready[ids[i]] = true;
      appendCsv('assignments.csv', ['session_id', 'player_id', 'device', 'timestamp', 'name'], [W.session.id, ids[i], device, now(), x.name]);
    });
    W.waiting = {};
    saveSession();
    log(`Lobby start: ${entries.length} players`);
    return { code: 200, body: adminSnapshot() };
  }
  const tc = route.match(/^tournament\/(start|pause|resume)$/);
  if (tc) { tourControl(tc[1]); return { code: 200, body: adminSnapshot() }; }
  return null;
}

// ---------------------------------------------------------------- players: join / leave / messages

// No session at all (the lobby is open): someone who joined by an ID (an old page's typed-ID box) is sent
// back to the name lobby under that name instead of the ladder's "free pairing" screen.
function bounceToLobby(id) {
  send(id, { type: 'error', relogin: true, name: id, message: 'Please enter your name to join the lobby.' });
  onLeave(id);
}

function onJoin(id) {
  if (!W.session) return bounceToLobby(id);
  W.online[id] = true;
  const rk = W.memberRoom[id];
  if (tourState()) tourOnJoin(id);
  else if (rk && W.rooms[rk]) W.rooms[rk].resumeIfReady();
  else tryStartPairings();
}

function onLeave(id) {
  delete W.online[id];
  delete W.conns[id];
  delete W.adhoc[id];
  const rk = W.memberRoom[id];
  if (rk && W.rooms[rk]) W.rooms[rk].pause(id);
}

function onMessage(id, msg) {
  if (msg.type === 'adhoc') {
    const partner = String(msg.partner || '').trim();
    if (!partner || partner === id || W.memberRoom[id] || tourState()) return;
    W.adhoc[id] = partner;
    tryStartPairings();
  } else if (msg.type === 'cancelAdhoc') {
    delete W.adhoc[id];
    lobbyStatus(id);
  } else if (msg.type === 'ready' && tourState()?.status === 'lobby') {
    tourReady(id);
  } else if (msg.type === 'ready') {
    const rk = W.memberRoom[id];
    if (rk && W.rooms[rk]) W.rooms[rk].setReady(id);
  } else if (msg.type === 'action') {
    const rk = W.memberRoom[id];
    if (rk && W.rooms[rk]) W.rooms[rk].act(id, msg.action);
  } else if (msg.type === 'leave') {
    if (W.online[id]) onLeave(id);
  }
}

// ---------------------------------------------------------------- timers + presence

function allTimers() {
  const list = [];
  for (const r of Object.values(W.rooms)) if (r.timer) list.push({ at: r.timer.at, fire: () => { const tm = r.timer; r.timer = null; r.fire(tm); } });
  if (W.tour.timer) list.push({ at: W.tour.timer.at, fire: () => { W.tour.timer = null; tourStartRound(); } });
  W.timers.forEach((tm) => list.push({ at: tm.at, fire: () => { W.timers = W.timers.filter((x) => x !== tm); if (tm.kind === 'tryStart') tryStartPairings(); else if (tm.kind === 'bot') botAct(tm); } }));
  return list;
}

function runDueTimers() {
  for (let guard = 0; guard < 500; guard++) {
    const due = allTimers().filter((x) => x.at <= now()).sort((a, b) => a.at - b.at)[0];
    if (!due) return;
    due.fire();
  }
}

function checkPresence(seen, except) {
  for (const id of Object.keys(W.online)) {
    if (id === except || isBot(id)) continue;
    if (!(seen[id] && now() - seen[id] < STALE_MS)) { log(`${id} timed out (no poll for ${STALE_MS} ms)`); onLeave(id); }
  }
}

// the earliest time this world needs a step even if no player sends anything
function nextDue() {
  const ts = allTimers().map((x) => x.at);
  if (Object.keys(W.online).length) ts.push(now() + HEARTBEAT_MS);
  return ts.length ? Math.min(...ts) : 0;
}

// re-attach class methods to the plain room objects of a freshly loaded world
function attach(world) {
  const w = { ...emptyWorld(), ...world };
  for (const r of Object.values(w.rooms)) Object.setPrototypeOf(r, r.kind === 'tour' ? TourMatch.prototype : Room.prototype);
  return w;
}

// Public metadata the fast poll path reads without loading the world.
function meta() {
  return { conns: W.conns, hot: Object.fromEntries(Object.keys(W.memberRoom).map((id) => [id, 1])),
    kicked: Object.fromEntries((W.kicked || []).map((t) => [t, 1])) };
}

const KICKED_MSG = { type: 'error', kicked: true, message: 'This window was disconnected: either this player was opened in another window or device, or the experimenter reset the game. Press "Join again" to play here.' };

// ---------------------------------------------------------------- the one entry point
//
// step(world, seen, req, nowMs) runs one request against a loaded world, synchronously.
// req: { kind: 'client', id, token, msgs } | { kind: 'admin', route, body } | { kind: 'tick' }
// Returns { world, meta, nextDue, out, appends, puts, logs, reply, changed }.
function step(world, seen, req, nowMs) {
  const before = JSON.stringify(world || null);
  W = attach(world || emptyWorld());
  C = { now: nowMs, out: [], appends: [], puts: [], logs: [], resets: [] };
  let reply = null;
  try {
    let self = null;
    if (req.kind === 'client') {
      const id = String(req.id || '').trim();
      const join = (req.msgs || []).some((m) => m.type === 'join');
      if (id && (W.kicked || []).includes(req.token) && !join) {
        // a window that another window replaced: ignore what it sends
        reply = { direct: [KICKED_MSG] };
      } else if (id) {
        self = id;
        seen = { ...seen, [id]: nowMs };
      }
    }
    runDueTimers();
    checkPresence(seen, self);
    if (!W.session) for (const id of Object.keys(W.online)) if (id !== self && !isBot(id)) bounceToLobby(id);
    if (req.kind === 'client' && self) {
      const id = self;
      const takeover = W.conns[id] !== req.token;
      const msgs = req.msgs || [];
      const joinMsg = msgs.some((m) => m.type === 'join');
      const onlyLeave = msgs.length > 0 && msgs.every((m) => m.type === 'leave');
      if (joinMsg || (!onlyLeave && (takeover || !W.online[id]))) {
        // (re)joining replaces any earlier connection for this ID, like a socket reconnect:
        // an interrupted trial is replayed once both players are back. The newest window wins;
        // the one it replaced is told so and stops (it does not try to take the ID back).
        if (W.online[id] && takeover && W.conns[id]) W.kicked = [...(W.kicked || []), W.conns[id]].slice(-50);
        if (W.online[id]) onLeave(id);
        W.conns[id] = req.token;
        W.online[id] = true;
        C.resets.push({ client: id, upto: W.seq[id] || 0 }); // drop messages meant for the old window
        send(id, { type: 'joined', id, name: tourState()?.names?.[id] || id, mode: tourState()?.mode || 'ladder' });
        onJoin(id);
      }
      if (onlyLeave && takeover) return finish(before, req, { });
      for (const m of msgs) if (m.type !== 'join') onMessage(id, m);
      reply = reply || {};
    } else if (req.kind === 'assign') {
      reply = assignSlot(String(req.device || ''), req.name, seen, !!req.leave);
    } else if (req.kind === 'admin') {
      try {
        reply = adminRoute(req.route, req.body || {}) || { code: 404, body: { error: 'not found' } };
      } catch (e) {
        reply = { code: 400, body: { error: e.message } };
      }
    }
    runDueTimers();
  } finally {
    // nothing to clean: W and C are replaced on the next call
  }
  return finish(before, req, reply);
}

function finish(before, req, reply) {
  const plain = JSON.parse(JSON.stringify(W)); // drops prototypes and getters
  const after = JSON.stringify(plain);
  return {
    world: plain, meta: meta(), nextDue: nextDue(), out: C.out, appends: C.appends, puts: dedupePuts(C.puts), resets: C.resets,
    logs: C.logs, reply, changed: after !== before || C.out.length > 0 || C.appends.length > 0 || C.puts.length > 0,
    selfSeen: req.kind === 'client' && reply && !reply.direct && W.conns[String(req.id || '').trim()] === req.token ? String(req.id || '').trim() : null,
  };
}

function dedupePuts(puts) {
  const m = new Map();
  for (const p of puts) m.set(p.session + '\u0000' + p.name, p);
  return [...m.values()];
}

module.exports = { step, emptyWorld, STALE_MS, csvCell, SPEED, KICKED_MSG };
