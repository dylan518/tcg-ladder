// End-of-game results: leaderboard, play-style titles and awards for the tournament's final screen.
// Pure functions. The Sender-turn coding mirrors analysis/analyze_tournament.py (same codes and rules).

const TILE_XY = { 'top-left': [-1, 1], top: [0, 1], 'top-right': [1, 1], left: [-1, 0], center: [0, 0],
  right: [1, 0], 'bottom-left': [-1, -1], bottom: [0, -1], 'bottom-right': [1, -1] };
const DIR_VEC = { up: [0, 1], 'up-right': [1, 1], right: [1, 0], 'down-right': [1, -1], down: [0, -1],
  'down-left': [-1, -1], left: [-1, 0], 'up-left': [-1, 1] };
const DWELL_MS = 1500;
const key = (p) => p[0] + ',' + p[1];
const angle = (v) => ((Math.atan2(v[0], v[1]) * 180) / Math.PI + 360) % 360;
const adiff = (a, b) => { const d = Math.abs(a - b) % 360; return Math.min(d, 360 - d); };

// "start(0,0)@0 up(0,1)@512 ... stop(0,1)@3000" -> [[x, y, arriveMs, dwellMs], ...] (repeats merged)
function visits(moves) {
  const ev = [...String(moves || '').matchAll(/(\w+)\((-?\d),(-?\d)\)@(\d+)/g)].map((m) => [m[1], +m[2], +m[3], +m[4]]);
  if (!ev.length) return [];
  const end = ev[ev.length - 1][3], out = [];
  for (let i = 0; i < ev.length; i++) {
    const [a, x, y, t] = ev[i];
    if (a === 'stop' || a === 'timeout') break;
    const nxt = i + 1 < ev.length ? ev[i + 1][3] : end;
    const last = out[out.length - 1];
    if (last && last[0] === x && last[1] === y) last[3] += nxt - t;
    else out.push([x, y, t, nxt - t]);
  }
  return out;
}

// out-and-backs (A -> B -> A) per board edge
function edgeWiggles(seq) {
  const w = new Map();
  for (let i = 0; i + 2 < seq.length; i++) {
    const a = key(seq[i]), b = key(seq[i + 1]), c = key(seq[i + 2]);
    if (a === c && a !== b) { const e = [a, b].sort().join('|'); w.set(e, (w.get(e) || 0) + 1); }
  }
  return w;
}

function codeTurn(moves, goalTile, goalDir) {
  const seq = visits(moves);
  const G = TILE_XY[goalTile], d = DIR_VEC[goalDir];
  const w = edgeWiggles(seq);
  const maxWiggle = w.size ? Math.max(...w.values()) : 0;
  const out = (code) => ({ code, maxWiggle, moves: Math.max(seq.length - 1, 0) });
  if (!G || !d || seq.length <= 1) return out('none');
  const gk = key(G), Cd = [d[0], d[1]], cdk = key(Cd);
  const visitedGoal = seq.some((s) => key(s) === gk), endsOnGoal = key(seq[seq.length - 1]) === gk;
  const mids = seq.slice(1, -1);
  const dwell = mids.length ? mids.reduce((m, s) => (s[3] > m[3] ? s : m)) : null;
  const dwellKey = dwell && dwell[3] >= DWELL_MS ? key(dwell) : null;
  const goalArms = [];
  for (const [e, n] of w) { const [a, b] = e.split('|'); if (a === gk) goalArms.push([b, n]); else if (b === gk) goalArms.push([a, n]); }
  const goalW = goalArms.reduce((s, [, n]) => s + n, 0);
  let cdW = 0;
  if (cdk !== gk) { for (const [e, n] of w) if (e.split('|').includes(cdk)) cdW += n; if (dwellKey === cdk) cdW += 2; }
  if (cdW >= 2 && cdW >= goalW && visitedGoal) return out('centre-dir');
  if (goalW >= 2) {
    let vx = 0, vy = 0;
    for (const [a, n] of goalArms) { const [x, y] = a.split(',').map(Number); vx += (x - G[0]) * n; vy += (y - G[1]) * n; }
    if (!vx && !vy) return out('wiggle-other');
    const diff = adiff(angle([vx, vy]), angle(d));
    return out(diff <= 22.5 ? 'goal-wiggle-F' : diff >= 157.5 ? 'goal-wiggle-B' : 'goal-wiggle-S');
  }
  if (maxWiggle >= 2) return out('wiggle-other');
  return out(endsOnGoal ? 'goal-only' : 'other');
}

const STYLES = {
  'goal-wiggle-F': { title: 'The Pointer', emoji: '👉', blurb: 'You stood on the goal and wiggled toward where the triangle should point. Clear, direct, arrow-like.' },
  'goal-wiggle-B': { title: 'The Archer', emoji: '🏹', blurb: 'You pulled back from the goal like drawing a bow: your wiggles went the opposite way to the arrow.' },
  'goal-wiggle-S': { title: 'The Sidestepper', emoji: '🦀', blurb: 'Your wiggles went off at an angle. Mysterious, and a real test for your partners!' },
  'centre-dir': { title: 'The Compass', emoji: '🧭', blurb: 'You showed the direction from the middle of the board first, then went to the goal. A system all your own.' },
  'goal-only': { title: 'The Minimalist', emoji: '🎯', blurb: 'Straight to the goal, no fuss. You trusted your partner to work out the rest.' },
  'wiggle-other': { title: 'The Free Spirit', emoji: '🌀', blurb: 'You invented your own dances around the board.' },
  other: { title: 'The Explorer', emoji: '🗺️', blurb: 'You took the scenic route around the board.' },
  none: { title: 'The Statue', emoji: '🗿', blurb: 'Stillness is a strategy too.' },
};

// log: [{ round, g, s, r, ok, code, maxWiggle, rms }] (one per game). Returns { players: {id: result}, leaderboard, awards }.
function computeResults(log, participants, names = {}) {
  const nm = (id) => names[id] || id;
  const games = [...new Map(log.map((e) => [e.round + ':' + e.g, e])).values()].sort((a, b) => a.round - b.round);
  const P = {};
  for (const id of participants) P[id] = { id, name: nm(id), wins: 0, games: 0, sendN: 0, sendWins: 0, recvN: 0, recvWins: 0, recvMs: [], codes: {}, codeWins: {}, maxWiggle: 0, streak: 0, bestStreak: 0 };
  for (const e of games) {
    for (const [id, role] of [[e.s, 'send'], [e.r, 'recv']]) {
      const p = P[id];
      if (!p) continue;
      p.games++; p.wins += e.ok ? 1 : 0;
      p.streak = e.ok ? p.streak + 1 : 0; p.bestStreak = Math.max(p.bestStreak, p.streak);
      if (role === 'send') {
        p.sendN++; p.sendWins += e.ok ? 1 : 0;
        p.codes[e.code] = (p.codes[e.code] || 0) + 1; p.codeWins[e.code] = (p.codeWins[e.code] || 0) + (e.ok ? 1 : 0);
        p.maxWiggle = Math.max(p.maxWiggle, e.maxWiggle || 0);
      } else { p.recvN++; p.recvWins += e.ok ? 1 : 0; if (e.rms > 0) p.recvMs.push(e.rms); }
    }
  }
  const median = (a) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  const list = Object.values(P);
  for (const p of list) {
    p.recvMedianMs = median(p.recvMs);
    const top = Object.entries(p.codes).sort((a, b) => b[1] - a[1] || (p.codeWins[b[0]] || 0) - (p.codeWins[a[0]] || 0))[0];
    p.style = { code: top ? top[0] : 'none', ...STYLES[top ? top[0] : 'none'] };
    if (top && Object.keys(p.codes).length >= 4) p.style.blurb += ' You also mixed it up a lot: ' + Object.keys(p.codes).length + ' different styles!';
  }
  // competition ranking by wins (1, 1, 3 …)
  const sorted = [...list].sort((a, b) => b.wins - a.wins);
  sorted.forEach((p, i) => { p.rank = i && sorted[i - 1].wins === p.wins ? sorted[i - 1].rank : i + 1; });
  const tied = (p) => sorted.filter((q) => q.rank === p.rank).length > 1;
  const leaderboard = sorted.map((p) => ({ name: p.name, wins: p.wins, games: p.games, rank: p.rank, tied: tied(p), id: p.id }));

  const awards = [];
  const best = (label, emoji, desc, val, better, eligible = () => true, fmt = (v) => v) => {
    const c = list.filter((p) => eligible(p) && val(p) !== null && val(p) !== undefined);
    if (!c.length) return;
    const top = c.reduce((m, p) => (better(val(p), val(m)) ? p : m));
    const winners = c.filter((p) => val(p) === val(top));
    if (winners.length === list.length && list.length > 1) return; // everyone tied: no award
    awards.push({ label, emoji, desc, value: fmt(val(top)), winners: winners.map((p) => p.id), names: winners.map((p) => p.name) });
  };
  const hi = (a, b) => a > b, lo = (a, b) => a < b;
  best('Champion', '🏆', 'most games won', (p) => p.wins, hi, (p) => p.wins > 0, (v) => `${v} wins`);
  best('Best Sender', '📡', 'best win rate as Sender', (p) => (p.sendN ? p.sendWins / p.sendN : null), hi, (p) => p.sendN >= 3 && p.sendWins > 0, (v) => `${Math.round(v * 100)}%`);
  best('Mind Reader', '🔮', 'best win rate as Receiver', (p) => (p.recvN ? p.recvWins / p.recvN : null), hi, (p) => p.recvN >= 3 && p.recvWins > 0, (v) => `${Math.round(v * 100)}%`);
  best('Lightning Fingers', '⚡', 'fastest typical Receiver turn', (p) => p.recvMedianMs, lo, (p) => p.recvN >= 3, (v) => `${(v / 1000).toFixed(1)} s`);
  best('Wiggle Master', '🌀', 'most back-and-forths in a single turn', (p) => p.maxWiggle, hi, (p) => p.maxWiggle >= 3, (v) => `${Math.ceil(v / 2)} in one turn`);
  best('Hot Streak', '🔥', 'most wins in a row', (p) => p.bestStreak, hi, (p) => p.bestStreak >= 2, (v) => `${v} in a row`);

  const players = {};
  for (const p of list) {
    players[p.id] = {
      name: p.name, rank: p.rank, tied: tied(p), total: list.length, wins: p.wins, games: p.games,
      send: { n: p.sendN, wins: p.sendWins }, recv: { n: p.recvN, wins: p.recvWins, medianS: p.recvMedianMs === null ? null : +(p.recvMedianMs / 1000).toFixed(1) },
      bestStreak: p.bestStreak, maxWiggle: Math.ceil(p.maxWiggle / 2), style: p.style,
      awards: awards.filter((a) => a.winners.includes(p.id)).map((a) => a.label),
    };
  }
  return { players, leaderboard, awards: awards.map(({ winners, ...a }) => a) };
}

// minimal CSV parser (quoted fields with commas), for rebuilding the log from rounds.csv
function parseCsv(text) {
  const rows = []; let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c; }
    else if (c === '"') q = true;
    else if (c === ',') { row.push(cell); cell = ''; }
    else if (c === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
    else if (c !== '\r') cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const [h, ...body] = rows;
  return body.filter((r) => r.length === h.length).map((r) => Object.fromEntries(h.map((k, i) => [k, r[i]])));
}

function logFromRoundsCsv(text) {
  return parseCsv(text).map((r) => {
    const c = codeTurn(r.sender_moves, r.goal_tile, r.goal_direction);
    return { round: +r.round, g: r.group, s: r.sender_id, r: r.receiver_id, ok: r.success === '1', code: c.code, maxWiggle: c.maxWiggle, rms: +r.receiver_phase_ms || 0 };
  });
}

module.exports = { codeTurn, computeResults, logFromRoundsCsv, STYLES };
