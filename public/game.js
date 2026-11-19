const $ = (s) => document.querySelector(s);
const show = (el, on) => el.classList.toggle('hidden', !on);
const SHAPE_NAME = { 1: 'rectangle', 2: 'circle', 3: 'triangle' };

let link, myId, waitingReady = false, state = null, timerStart = 0, timerTotal = 1, timerLeft = 0;

// polling link to the server (net.js); it rejoins by itself after a network drop
function connect() {
  link = createLink(handle);
  if (myId) link.join(myId);
}

function sendMsg(m) { if (link.id) link.send(m); }

function handle(m) {
  if (m.type === 'error' && m.relogin) {
    // joined by typing a name as an ID while no session exists: move into the lobby under that name
    myId = null; link.stop();
    show($('#lobby'), false); show($('#game'), false); show($('#join'), true);
    joinName = cleanName(m.name || ''); store('tcg-name', joinName);
    $('#joinErr').textContent = '';
    return joinName ? autoJoin() : showNameForm();
  }
  if (m.type === 'error') {
    $('#joinErr').textContent = m.message;
    show($('#rejoinBtn'), !!m.kicked);
    if (m.kicked) $('#joinPrompt').textContent = '';
    myId = null;
    link.stop();
    show($('#join'), true); show($('#lobby'), false); show($('#game'), false);
  } else if (m.type === 'mode') {
    setMode(m.mode);
  } else if (m.type === 'results') {
    state = null;
    show($('#game'), false); show($('#lobby'), true);
    show($('#adhocForm'), false); show($('#cancelAdhoc'), false); show($('#lobbyReady'), false);
    $('#lobby').classList.remove('announce');
    $('#lobbyTitle').textContent = ''; $('#lobbyDetail').textContent = '';
    show($('#lobby .howto'), false);
    $('#results').innerHTML = resultsHtml(m);
    show($('#results'), true);
  } else if (m.type === 'wait') {
    show($('#results'), false); show($('#lobby .howto'), true);
    // tournament waiting screen: sitting out, between rounds, or before the start
    state = null;
    show($('#game'), false); show($('#lobby'), true);
    show($('#adhocForm'), false); show($('#cancelAdhoc'), false);
    $('#lobbyTitle').textContent = m.title;
    $('#lobbyDetail').textContent = m.detail || '';
    show($('#lobbyReady'), !!m.canReady);
    $('#lobby').classList.toggle('announce', !!m.announce);
  } else if (m.type === 'name') {
    $('#meId').textContent = m.name;
  } else if (m.type === 'joined') {
    setMode(m.mode);
    localStorage.setItem('tcg-id', m.id);
    $('#meId').textContent = m.name || m.id;
    show($('#join'), false); show($('#lobby'), true);
  } else if (m.type === 'lobby') {
    state = null;
    show($('#game'), false); show($('#lobby'), true);
    const adhoc = m.state === 'unscheduled';
    show($('#adhocForm'), adhoc);
    show($('#cancelAdhoc'), m.state === 'waiting' && m.round === 'free play');
    if (m.state === 'waiting') {
      $('#lobbyTitle').textContent = m.round === 'free play'
        ? `Waiting for ${m.partner}`
        : `Round ${m.round} of ${m.totalRounds}: your partner is ${m.partner}`;
      $('#lobbyDetail').textContent = m.detail || 'Ask your partner to enter your ID as their partner.';
    } else {
      $('#lobbyTitle').textContent = m.state === 'done' ? 'All done' : 'Free pairing';
      $('#lobbyDetail').textContent = m.message;
    }
  } else if (m.type === 'paired') {
    show($('#lobby'), false); show($('#game'), true);
    $('#gPartner').textContent = m.partner || '';
    phaseInfo = m.phases ? m : null;
    $('#gRound').textContent = m.round === 'free' ? 'Free play'
      : m.phases ? `Partner ${m.phase} of ${m.phases} · round ${m.phaseRound} of ${m.phaseRounds}` : `Round ${m.round}/${m.totalRounds}`;
  } else if (m.type === 'pairingDone') {
    show($('#game'), false); show($('#lobby'), true);
    $('#lobbyTitle').textContent = `Finished with ${m.partner}: ${m.successes}/${m.trials} wins`;
    $('#lobbyDetail').textContent = 'Looking for your next partner…';
    show($('#adhocForm'), false);
  } else if (m.type === 'paused') {
    setStatus(m.message);
    setControls(false);
  } else if (m.type === 'readyCheck') {
    state = null;
    waitingReady = true;
    const iAmReady = m.ready.includes(myId);
    const lines = [];
    if (m.reason) lines.push(m.reason);
    lines.push(iAmReady ? 'Waiting for your partner to press Start…' : `Trial ${m.trial} of ${m.totalTrials} is next. Press Start (or Space) when you are ready.`);
    setStatus(lines.join('\n'));
    show($('#readyBtn'), !iAmReady);
    setControls(false);
    timerLeft = 0;
    draw();
  } else if (m.type === 'state') {
    waitingReady = false;
    show($('#readyBtn'), false);
    state = m;
    render();
  }
}

// ---- end-of-game results: leaderboard, play style, awards
const escH = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const ORD = (n) => n + ((n % 100 >= 11 && n % 100 <= 13) ? 'th' : (['th', 'st', 'nd', 'rd'][n % 10] || 'th'));
function resultsHtml({ me, leaderboard, awards }) {
  const medal = (r) => ['🥇', '🥈', '🥉'][r - 1] || '';
  const place = `${me.tied ? 'Tied ' : ''}${ORD(me.rank)} of ${me.total}`;
  const pct = (w, n) => (n ? Math.round((100 * w) / n) + '%' : '–');
  return `
    <div class="res-head">
      <div class="res-medal">${medal(me.rank) || '🎉'}</div>
      <div><div class="res-place">${place}</div><div class="res-sub">${escH(me.name)}, you won <b>${me.wins}</b> of ${me.games} games</div></div>
    </div>
    <div class="res-style">
      <div class="res-emoji">${me.style.emoji}</div>
      <div><div class="res-label">Your play style</div><div class="res-title">${escH(me.style.title)}</div><p>${escH(me.style.blurb)}</p></div>
    </div>
    ${me.awards.length ? `<div class="res-mine">${me.awards.map((a) => { const x = awards.find((y) => y.label === a); return `<span>${x.emoji} ${escH(a)}</span>`; }).join('')}</div>` : ''}
    <div class="res-stats">
      <div><b>${pct(me.send.wins, me.send.n)}</b><span>as Sender (${me.send.wins}/${me.send.n})</span></div>
      <div><b>${pct(me.recv.wins, me.recv.n)}</b><span>as Receiver (${me.recv.wins}/${me.recv.n})</span></div>
      <div><b>${me.recv.medianS ?? '–'}${me.recv.medianS != null ? ' s' : ''}</b><span>typical Receiver turn</span></div>
      <div><b>${me.bestStreak}</b><span>best win streak</span></div>
    </div>
    <h3>Leaderboard</h3>
    <ol class="res-board">${leaderboard.map((p) => `<li class="${p.name === me.name ? 'me' : ''}"><span class="rk">${medal(p.rank) || p.rank}</span><span class="nm">${escH(p.name)}</span><span class="wn">${p.wins} wins</span></li>`).join('')}</ol>
    ${awards.length ? `<h3>Awards</h3><ul class="res-awards">${awards.map((a) => `<li><span class="ae">${a.emoji}</span><span><b>${escH(a.label)}</b>: ${a.names.map(escH).join(' & ')} <small>(${escH(a.desc)}, ${escH(a.value)})</small></span></li>`).join('')}</ul>` : ''}
    <p class="res-thanks">Thanks for playing! You can close this page.</p>`;
}

function setStatus(text) { const el = $('#status'); el.textContent = text; el.className = ''; }
// tournament modes: a short headline plus what to do, with key caps; `kind` colours the box
function setGuide(title, body, kind) {
  const el = $('#status');
  el.className = 'guide ' + kind;
  el.innerHTML = `<b>${title}</b><span>${body}</span>`;
}
const K = (t) => `<kbd>${t}</kbd>`;
const ARROWS = K('←') + K('↑') + K('↓') + K('→');

let tourMode = false, phaseInfo = null;
function setMode(mode) {
  tourMode = !!mode && mode !== 'ladder';
  document.body.classList.toggle('tour', tourMode);
}

function setControls(on) {
  document.querySelectorAll('.controls button').forEach((b) => {
    const rot = b.dataset.a.startsWith('rotate');
    const myShape = state && state.tokens[state.you].shape;
    b.disabled = !on || (rot && myShape === 2);
  });
}

function myTurn() {
  return state && ((state.phase === 'sender' && state.yourRole === 'sender') || (state.phase === 'receiver' && state.yourRole === 'receiver'));
}

function render() {
  const s = state;
  $('#gTrial').textContent = `${s.trial}/${s.totalTrials}`;
  $('#gWins').textContent = s.successes;
  const me = s.tokens[s.you];
  const color = me.colorName || (s.you === 1 ? 'blue' : 'orange');
  const ROLE = s.yourRole.toUpperCase();
  const shape = me.shape === 2 ? 'circle' : 'triangle';
  const solution = s.phase === 'feedback' && s.solution;
  if (tourMode && solution) {
    setGuide(s.success ? 'Success!' : 'Not this time',
      "The outline shows where the triangle had to go and which way it had to point. The solid shapes are where both keys ended up.", s.success ? 'intro' : 'fail');
  } else if (tourMode && s.phase !== 'feedback') {
    const sender = s.yourRole === 'sender';
    if (s.phase === 'roleassignment' || s.phase === 'tokenassignment') {
      const fresh = phaseInfo && phaseInfo.phaseRound === 1 ? (phaseInfo.phase > 1 ? 'New partner. ' : 'First round. ') : '';
      setGuide(`${fresh}You are the ${ROLE} this round`, sender
        ? `You control the ${color} ${shape}. You will see the goal; your partner will not.`
        : `You control the ${color} ${shape}. Your partner can see the goal and will try to show it to you.`, 'intro');
    } else if (myTurn() && sender && s.botSendsForYou) {
      setGuide('Your turn: you are the SENDER', 'The bot is moving your circle for you. Watch.', 'watch');
    } else if (myTurn() && sender) {
      setGuide(`Your turn: you are the SENDER`,
        `Communicate the triangle's tile and direction by moving your ${shape} with ${ARROWS}<br>Press ${K('Space')} when you are done`, 'go');
    } else if (myTurn()) {
      setGuide(`Your turn: you are the RECEIVER`,
        `Move your ${shape} to the goal tile with ${ARROWS} and turn it with ${K('A')} ${K('D')}<br>Press ${K('Space')} when you are done`, 'go');
    } else if (s.phase === 'sender') {
      setGuide('Watch your partner', `Your partner is moving their circle to show you where your triangle goes and which way it points. Your turn is next.`, 'watch');
    } else {
      setGuide('Your partner is placing their triangle', 'Watch where it ends up.', 'watch');
    }
  } else if (s.phase === 'roleassignment' || s.phase === 'tokenassignment') {
    setStatus(`You are the ${ROLE}\nYou control the ${color} key`);
  } else if (myTurn()) {
    setStatus(me.state === 'outside'
      ? (s.yourRole === 'sender' ? 'Show your partner the code, then move your key into position\n' : 'Move your key into position\n') + 'Press the space bar (or an arrow key) to make your key appear'
      : (me.shape !== 2 && s.yourRole === 'sender' && me.goal && me.angle !== me.goal.angle && me.xPos === me.goal.xPos && me.yPos === me.goal.yPos
        ? "Don't forget to rotate\n" : 'Arrow keys move, A / D rotate\n') + 'Press the space bar when you are done');
  } else if (s.phase === 'sender') {
    setStatus('Watch your partner, the SENDER\nYour turn comes next');
  } else if (s.phase === 'receiver') {
    setStatus('Your partner, the RECEIVER, is placing their key');
  } else if (s.phase === 'feedback') {
    setStatus(s.success ? 'You cracked the vault together!' : 'The vault stayed locked');
  }
  setControls(myTurn());
  if (s.remaining !== null) { timerStart = performance.now(); timerTotal = s.turnMs; timerLeft = s.remaining; }
  else timerLeft = 0;
  draw();
}

// ---------------------------------------------------------------- drawing
// Look modelled on the original game: separate light-blue cells, the code shown as
// nested outlines, flat solid keys, and a key that waits below the board.

const canvas = $('#board'), ctx = canvas.getContext('2d');
const W = 520, CELL = 150, GAP = 12, X0 = (W - (3 * CELL + 2 * GAP)) / 2, Y0 = 20;
const TRAY_Y = Y0 + 3 * CELL + 2 * GAP + 50;
const center = (x, y) => [X0 + (x + 1) * (CELL + GAP) + CELL / 2, Y0 + (1 - y) * (CELL + GAP) + CELL / 2];
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

function shapePath(shape, inset = 0, scale = 1) {
  ctx.beginPath();
  if (shape === 1) {
    const w = 120 * scale / 2 - inset, h = 34 * scale / 2 - inset;
    ctx.rect(-w, -h, 2 * w, 2 * h);
  } else if (shape === 2) {
    ctx.arc(0, 0, 46 * scale - inset, 0, Math.PI * 2);
  } else {
    const r = 52 * scale - inset * 2; // circumradius of an upward-pointing triangle
    ctx.moveTo(0, -r);
    ctx.lineTo(r * Math.cos(Math.PI / 6), r / 2);
    ctx.lineTo(-r * Math.cos(Math.PI / 6), r / 2);
    ctx.closePath();
  }
}

function drawKey(t, cx, cy, angle, mode, scale = 1) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate((angle * Math.PI) / 180);
  if (mode === 'goal') {
    ctx.lineWidth = 1.3;
    ctx.strokeStyle = rgba(t.color, 0.95);
    for (let k = 0; k < 5; k++) { shapePath(t.shape, k * 3, scale); ctx.stroke(); }
  } else {
    shapePath(t.shape, 0, scale);
    ctx.fillStyle = rgba(t.color, 1);
    ctx.fill();
  }
  if (t.shape === 3 && tourMode) { // a dot on the tip marks the front corner (8 directions)
    ctx.beginPath();
    ctx.arc(0, -52 * scale, 9 * scale, 0, Math.PI * 2);
    if (mode === 'goal') { ctx.lineWidth = 2; ctx.strokeStyle = rgba(t.color, 0.95); ctx.fillStyle = '#fff'; ctx.fill(); ctx.stroke(); }
    else { ctx.fillStyle = '#fff'; ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = rgba(t.color, 1); ctx.stroke(); }
  }
  ctx.restore();
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// feedback picture: a safe in a green (open) or red (locked) frame
function drawVault(success) {
  const accent = success ? '#3aa655' : '#e8402a';
  const cx = W / 2, cy = Y0 + 230, S = 360;
  ctx.lineWidth = 14; ctx.strokeStyle = accent;
  roundRect(cx - S / 2, cy - S / 2, S, S, 16); ctx.stroke();
  ctx.lineWidth = 2;
  roundRect(cx - S / 2 + 18, cy - S / 2 + 18, S - 36, S - 36, 2); ctx.stroke();
  // body
  ctx.fillStyle = '#b9b9bb'; roundRect(cx - 120, cy - 120, 240, 220, 18); ctx.fill();
  ctx.strokeStyle = '#6b6b6d'; ctx.lineWidth = 6; roundRect(cx - 104, cy - 104, 208, 188, 10); ctx.stroke();
  ctx.fillStyle = '#6b6b6d';
  ctx.fillRect(cx - 95, cy + 100, 50, 12); ctx.fillRect(cx + 45, cy + 100, 50, 12);
  // handle
  ctx.beginPath(); ctx.arc(cx - 60, cy - 15, 14, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.moveTo(cx - 66, cy - 8); ctx.lineTo(cx - 54, cy - 8); ctx.lineTo(cx - 58, cy + 55); ctx.lineTo(cx - 62, cy + 55); ctx.closePath(); ctx.fill();
  // dials tinted with the two key colours
  [[cy - 45, [49, 113, 166]], [cy + 30, [223, 129, 51]]].forEach(([dy, c]) => {
    ctx.fillStyle = '#9d9d9f'; ctx.beginPath(); ctx.arc(cx + 45, dy, 34, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#f4f4f4'; ctx.beginPath(); ctx.arc(cx + 45, dy, 26, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#555'; ctx.lineWidth = 2;
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4;
      ctx.beginPath(); ctx.moveTo(cx + 45 + Math.cos(a) * 17, dy + Math.sin(a) * 17); ctx.lineTo(cx + 45 + Math.cos(a) * 23, dy + Math.sin(a) * 23); ctx.stroke();
    }
    ctx.fillStyle = rgba(c, 1); ctx.beginPath(); ctx.arc(cx + 45, dy, 7, 0, Math.PI * 2); ctx.fill();
  });
  // check or cross
  ctx.strokeStyle = accent; ctx.lineWidth = 16; ctx.lineCap = 'round';
  ctx.beginPath();
  if (success) { ctx.moveTo(cx - 70, cy); ctx.lineTo(cx - 15, cy + 55); ctx.lineTo(cx + 85, cy - 70); }
  else { ctx.moveTo(cx - 70, cy - 70); ctx.lineTo(cx + 70, cy + 70); ctx.moveTo(cx + 70, cy - 70); ctx.lineTo(cx - 70, cy + 70); }
  ctx.stroke();
  ctx.lineCap = 'butt';
}

function draw() {
  ctx.clearRect(0, 0, W, canvas.height);
  const solution = state && state.phase === 'feedback' && state.solution;
  if (state && state.phase === 'feedback' && !solution) return drawVault(state.success);
  if (solution) { // green / red frame around the board
    ctx.lineWidth = 8; ctx.strokeStyle = state.success ? '#3aa655' : '#e8402a';
    roundRect(X0 - 14, Y0 - 14, 3 * CELL + 2 * GAP + 28, 3 * CELL + 2 * GAP + 28, 10); ctx.stroke();
  }
  ctx.strokeStyle = '#a8c3df'; ctx.lineWidth = 2;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    ctx.strokeRect(X0 + i * (CELL + GAP), Y0 + j * (CELL + GAP), CELL, CELL);
  }
  if (!state) return;
  const toks = [state.tokens[1], state.tokens[2]];
  for (const t of toks) if (t.goal) drawKey(t, ...center(t.goal.xPos, t.goal.yPos), t.goal.angle, 'goal');
  for (const t of toks) if (t.state === 'in-use' || (solution && t.state === 'locked')) drawKey(t, ...center(t.xPos, t.yPos), t.angle, 'key');
  // my own key waits below the board until I press space
  const me = state.tokens[state.you];
  if (me.state === 'outside' && !solution) drawKey(me, W / 2, TRAY_Y, 0, 'key', 0.55);
}

function tick() {
  const fill = $('#timerFill');
  if (timerLeft > 0) {
    const left = Math.max(0, timerLeft - (performance.now() - timerStart));
    fill.style.width = (100 * left / timerTotal) + '%';
    fill.style.background = left < 5000 ? '#c0392b' : '#3171a6';
  } else fill.style.width = '0';
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- input

const KEYS = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right', a: 'rotateleft', A: 'rotateleft', d: 'rotateright', D: 'rotateright', ' ': 'stop' };

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const a = KEYS[e.key];
  if (a === 'stop' && !$('#lobbyReady').classList.contains('hidden') && !$('#lobby').classList.contains('hidden')) {
    e.preventDefault();
    show($('#lobbyReady'), false);
    return sendMsg({ type: 'ready' });
  }
  if (a === 'stop' && waitingReady && !$('#readyBtn').classList.contains('hidden')) {
    e.preventDefault();
    return sendMsg({ type: 'ready' });
  }
  if (!a || !myTurn()) return;
  e.preventDefault();
  sendMsg({ type: 'action', action: a });
});

document.querySelectorAll('.controls button').forEach((b) =>
  b.addEventListener('click', () => myTurn() && sendMsg({ type: 'action', action: b.dataset.a })));

// One shared link for everyone. Tournament / pilot / one-on-one: the server assigns this device
// a slot (and gives the same one back later). Ladder sessions: tap a name or type an ID.
// one ID per browser tab (sessionStorage): a reload keeps it, a second tab counts as another person
function deviceId() {
  let d = null;
  try { d = sessionStorage.getItem('tcg-device'); } catch (e) { /* storage blocked */ }
  if (!d) {
    d = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now().toString(36);
    try { sessionStorage.setItem('tcg-device', d); } catch (e) { /* a refresh then counts as a new person */ }
  }
  return d;
}
// Tournament modes: the player types their name; the server gives this device a slot and keeps the name.
// The name is remembered for this tab (sessionStorage), so a reload rejoins without asking again,
// while a new tab or a shared lab computer asks for the name first (prefilled from localStorage).
const cleanName = (n) => String(n).replace(/\s+/g, ' ').trim().slice(0, 40);
const store = (k, v) => { try { if (v === undefined) return sessionStorage.getItem(k); sessionStorage.setItem(k, v); } catch (e) { return null; } };
let joinName = store('tcg-name') || '', joinTimer = null;
function showNameForm(msg) {
  show($('#slots'), false); show($('#joinForm'), false);
  show($('#nameForm'), true); show($('#joinPrompt'), !!msg);
  $('#joinPrompt').textContent = msg || '';
  if (!$('#pname').value) $('#pname').value = joinName || localStorage.getItem('tcg-name') || '';
}
async function autoJoin() {
  clearTimeout(joinTimer);
  if (myId) return;
  if (!joinName) return showNameForm();
  try {
    const res = await fetch('/api/assign', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ device: deviceId(), name: joinName }) });
    const r = await res.json();
    if (res.ok && r.id) { myId = r.id; $('#joinErr').textContent = ''; show($('#nameForm'), false); show($('#joinHowto'), false); return link.join(myId); }
    if (r.needName) { joinName = ''; store('tcg-name', ''); $('#joinErr').textContent = r.error; return showNameForm(); }
    show($('#nameForm'), false); show($('#joinPrompt'), true);
    $('#joinPrompt').textContent = r.waiting ? `Thanks, ${joinName}. You're in the lobby. The game starts when the experimenter starts it; keep this page open and read how to play below.` : (r.error || 'Could not join.');
    if (r.waiting) setMode('tournament');
    show($('#joinHowto'), !!r.waiting);
    if (r.retry) joinTimer = setTimeout(autoJoin, 3000);
  } catch (e) { joinTimer = setTimeout(autoJoin, 3000); }
}
// closing the page while waiting in the lobby takes the name off the experimenter's list at once
addEventListener('pagehide', () => {
  if (joinName && !myId && navigator.sendBeacon) navigator.sendBeacon('/api/assign', new Blob([JSON.stringify({ device: deviceId(), leave: true })], { type: 'text/plain' }));
});
$('#nameForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const n = $('#pname').value.replace(/\s+/g, ' ').trim();
  if (!n) return;
  joinName = n; store('tcg-name', n);
  try { localStorage.setItem('tcg-name', n); } catch (err) { /* fine */ }
  $('#joinErr').textContent = '';
  show($('#nameForm'), false); show($('#joinPrompt'), true);
  $('#joinPrompt').textContent = 'Joining…';
  autoJoin();
});

async function loadSlots() {
  if ($('#join').classList.contains('hidden') || myId) return setTimeout(loadSlots, 3000);
  try {
    const r = await (await fetch('/api/slots', { cache: 'no-store' })).json();
    if (!r.mode || r.mode !== 'ladder') {
      // no session yet, or a tournament / pilot / one-on-one: join by name (waits for the session if needed)
      if (r.mode) lastMode = r.mode;
      if (urlId && r.mode && r.mode !== 'tournament') { myId = urlId; link.join(myId); return setTimeout(loadSlots, 3000); }
      if (!joinName) showNameForm();
      else if (!joinTimer) autoJoin();
      return setTimeout(loadSlots, 3000);
    }
    lastMode = r.mode;
    if (urlId) { myId = urlId; link.join(myId); return setTimeout(loadSlots, 3000); }
    show($('#nameForm'), false);
    const names = r.names || [];
    show($('#slots'), names.length > 0);
    show($('#joinForm'), names.length === 0);
    show($('#joinPrompt'), names.length === 0);
    $('#slotList').innerHTML = '';
    for (const n of names) {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = n.id; // a name open elsewhere is taken over by the new window
      b.addEventListener('click', () => { myId = n.id; $('#joinErr').textContent = ''; link.join(myId); });
      $('#slotList').appendChild(b);
    }
  } catch (e) { /* keep the typed form */ }
  setTimeout(loadSlots, 3000);
}

// after being disconnected (opened elsewhere, or the experimenter pressed "Kick everyone")
$('#rejoinBtn').addEventListener('click', () => {
  show($('#rejoinBtn'), false);
  $('#joinErr').textContent = '';
  if (urlId && (forceId || lastMode !== 'tournament')) { myId = urlId; link.join(myId); } else { myId = null; autoJoin(); }
});

$('#joinForm').addEventListener('submit', (e) => {
  e.preventDefault();
  myId = $('#pid').value.trim();
  $('#joinErr').textContent = '';
  if (myId) link.join(myId);
});

$('#adhocForm').addEventListener('submit', (e) => {
  e.preventDefault();
  sendMsg({ type: 'adhoc', partner: $('#partnerId').value.trim() });
});
$('#readyBtn').addEventListener('click', () => sendMsg({ type: 'ready' }));
$('#lobbyReady').addEventListener('click', () => { show($('#lobbyReady'), false); sendMsg({ type: 'ready' }); });
$('#cancelAdhoc').addEventListener('click', () => sendMsg({ type: 'cancelAdhoc' }));

// ?id=P01 in the link skips the join screen; a remembered ID is only prefilled,
// so a shared lab computer never silently joins as the previous participant
const urlId = new URLSearchParams(location.search).get('id');
$('#pid').value = urlId || localStorage.getItem('tcg-id') || '';
// ?id=P3 joins as that player straight away only with &force=1 (experimenter testing). Otherwise the
// page first asks which kind of session is running: in a tournament ?id is ignored and the server
// assigns this device a free slot, so two people sharing one link can never knock each other out.
const forceId = new URLSearchParams(location.search).get('force') === '1';
if (urlId && forceId) myId = urlId;
let lastMode = null;
// same rules on the game screen, so players can check them between trials
$('#gameHowto').appendChild($('#lobby .howto').cloneNode(true));
$('#joinHowto').appendChild($('#lobby .howto').cloneNode(true));
connect();
loadSlots();
draw();
tick();
