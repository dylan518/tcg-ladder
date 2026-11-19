// Pictures for "How to play". They show only rules (who sees what, turn order, what counts as
// success, the keys). On purpose they never show the Sender's key on the board or any movement,
// so no signalling strategy is suggested. Drawn in the board's own style.
(function () {
  const BLUE = '#3171a6', ORANGE = '#df8133', OUTLINE = '#b4561a', CELLC = '#a8c3df', GREEN = '#3aa655', RED = '#e8402a';
  const C = 30, G = 4, B = 3 * C + 2 * G; // mini board: 3 cells of 30 px, 4 px gaps

  const cx = (x) => (x + 1) * (C + G) + C / 2; // x, y in board coords: -1..1, y = 1 is the top row
  const cy = (y) => (1 - y) * (C + G) + C / 2;
  const ANG = { up: 0, 'up-right': 45, right: 90, 'down-right': 135, down: 180, 'down-left': 225, left: 270, 'up-left': 315 };

  function cells() {
    let s = '';
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += `<rect x="${i * (C + G)}" y="${j * (C + G)}" width="${C}" height="${C}" fill="#fff" stroke="${CELLC}" stroke-width="1.2"/>`;
    return s;
  }
  function tri(x, y, r, a, attrs) {
    const p = [[0, -r], [r * Math.cos(Math.PI / 6), r / 2], [-r * Math.cos(Math.PI / 6), r / 2]].map(([u, v]) => `${u.toFixed(1)},${v.toFixed(1)}`).join(' ');
    // tournament modes: a dot on the tip marks the front corner (shown via .only-tour)
    const dot = `<circle class="only-tour" cx="0" cy="${(-r).toFixed(1)}" r="${(r * 0.2).toFixed(1)}" fill="#fff" stroke="${/fill="none"/.test(attrs) ? (attrs.match(/stroke="([^"]+)"/) || [])[1] : (attrs.match(/fill="([^"]+)"/) || [])[1]}" stroke-width="1"/>`;
    return `<g transform="translate(${x} ${y}) rotate(${a})"><polygon points="${p}" ${attrs}/>${r >= 9 ? dot : ''}</g>`;
  }
  function circ(x, y, r, attrs) { return `<circle cx="${x}" cy="${y}" r="${r}" ${attrs}/>`; }
  // the goal: nested outlines, as on the real board
  const goalTri = (x, y, dir, col) => [11, 9, 7].map((r) => tri(cx(x), cy(y), r, ANG[dir], `fill="none" stroke="${col}" stroke-width="1.1"`)).join('');
  const goalCirc = (x, y, col) => [10, 8, 6].map((r) => circ(cx(x), cy(y), r, `fill="none" stroke="${col}" stroke-width="1"`)).join('');
  const keyTri = (x, y, dir) => tri(cx(x), cy(y), 10, ANG[dir], `fill="${ORANGE}"`);

  function board(inner, below, label) {
    const h = B + (below ? 26 : 0);
    return `<figure class="fig-board"><svg viewBox="-2 -2 ${B + 4} ${h + 4}" width="${Math.round((B + 4) * 1.35)}" height="${Math.round((h + 4) * 1.35)}" aria-hidden="true">${cells()}${inner}${below ? below(B / 2, B + 14) : ''}</svg>${label ? `<figcaption>${label}</figcaption>` : ''}</figure>`;
  }
  const mark = (ok) => `<span class="fig-mark ${ok ? 'ok' : 'no'}">${ok ? '✓' : '✗'}</span>`;
  const key = (t, wide) => `<kbd class="fig-key${wide ? ' wide' : ''}">${t}</kbd>`;

  // 1. what each player sees (ladder trials also give the Sender a target of their own)
  const sees = `
    <div class="fig">
      <div class="fig-title">What each player sees</div>
      <div class="fig-row">
        ${board(goalTri(1, 1, 'left', OUTLINE) + `<g class="only-ladder">${goalCirc(-1, -1, BLUE)}</g>`,
          (x, y) => circ(x, y, 7, `fill="${BLUE}"`), '<b>Sender</b> sees the goal outline')}
        ${board('', (x, y) => tri(x, y + 1, 8, 0, `fill="${ORANGE}"`), '<b>Receiver</b> sees an empty board')}
      </div>
      <p class="fig-note">The outline shows where the Receiver's triangle must end up and which way it must point.<span class="only-ladder"> The Sender may also have an outline for their own key.</span> <span class="only-ladder"> Your key waits below the board until you press ${key('Space', 1)}.</span><span class="only-tour"> When your turn starts, your key is already in the middle of the board.</span></p>
    </div>`;

  // 2. turn order (no board, no positions)
  const turns = `
    <div class="fig">
      <div class="fig-title">Taking turns</div>
      <div class="fig-steps">
        <div class="fig-step"><span class="fig-num">1</span><svg viewBox="0 0 30 30" width="30" height="30" aria-hidden="true">${circ(15, 15, 11, `fill="${BLUE}"`)}</svg><div><b>Sender</b> moves their key<span class="only-tour"> to communicate the goal</span></div></div>
        <span class="fig-arrow" aria-hidden="true">→</span>
        <div class="fig-step"><span class="fig-num">2</span><svg viewBox="0 0 30 30" width="30" height="30" aria-hidden="true">${tri(15, 17, 13, 0, `fill="${ORANGE}"`)}</svg><div><b>Receiver</b> places and turns theirs</div></div>
        <span class="fig-arrow" aria-hidden="true">→</span>
        <div class="fig-step"><span class="fig-num">3</span><span class="fig-vault" aria-hidden="true">${mark(true)}${mark(false)}</span><div>Vault opens or stays locked</div></div>
      </div>
    </div>`;

  // 3. success: the goal is the same in all three; only the Receiver's final triangle differs
  const g = goalTri(0, 1, 'right', '#e7b48c'); // faded so the Receiver's triangle reads clearly on top
  const success = `
    <div class="fig">
      <div class="fig-title">When does the vault open?</div>
      <div class="fig-row">
        ${board(g + keyTri(0, 1, 'right'), null, `${mark(true)} Right tile, right direction`)}
        ${board(g + keyTri(0, 1, 'left'), null, `${mark(false)} Right tile, wrong direction`)}
        ${board(g + keyTri(-1, -1, 'right'), null, `${mark(false)} Wrong tile`)}
      </div>
      <p class="fig-note only-ladder">If the Sender has an outline too, their key must also end up on it.</p>
    </div>`;

  // 4. keys
  const keys = `
    <div class="fig">
      <div class="fig-title">Keys</div>
      <div class="fig-keys">
        <div>${key('Space', 1)}<span><span class="only-ladder">show your key, then </span>end your turn</span></div>
        <div>${key('←')}${key('↑')}${key('↓')}${key('→')}<span>move one tile</span></div>
        <div>${key('A')}${key('D')}<span>turn left / right<span class="only-tour"> by an eighth of a turn: the triangle can point 8 ways, including diagonals; the dot marks its tip</span> (circles can't turn)</span></div>
      </div>
      <p class="fig-note">Each turn has a time limit, shown by the bar above the board. Colours and shapes in the game may differ from these pictures.</p>
    </div>`;

  document.querySelectorAll('.howto-figs').forEach((el) => {
    el.innerHTML = sees + turns + success + keys;
  });
})();
