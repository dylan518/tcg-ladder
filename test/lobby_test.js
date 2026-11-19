// Open lobby → people join by name → admin starts → tournament built for exactly them → starts once all connected.
//   ADMIN_KEY=test PORT=8094 node dev-server.js &   then   URL=http://localhost:8094 ADMIN_KEY=test node test/lobby_test.js
const URL_ = process.env.URL || 'http://localhost:8094', KEY = process.env.ADMIN_KEY || 'test';
const post = async (p, b) => { const r = await fetch(`${URL_}/api/${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }); return { code: r.status, j: await r.json() }; };
const admin = (route, b = {}) => post(`${route}?key=${KEY}`, b);
const status = async () => (await fetch(`${URL_}/api/status?key=${KEY}`)).json();
const errs = [], check = (ok, msg) => { if (!ok) errs.push(msg); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  await admin('tournament/setup', { mode: 'tournament', players: 4, autoStart: true }); // an old session to close
  const open = await admin('lobby/open');
  check(open.code === 200 && !open.j.session, 'lobby/open should clear the session');
  const early = await admin('lobby/start', { turnSeconds: 3 });
  check(early.code === 400, 'start with nobody should be refused: ' + early.code);

  const people = ['Ann', 'Ben', 'Cat', 'Dev', 'Eli'];
  for (const [i, n] of people.entries()) {
    const r = await post('assign', { device: 'L' + i, name: n });
    check(r.code === 404 && r.j.waiting, `${n} should wait in the lobby: ${JSON.stringify(r)}`);
  }
  let st = await status();
  check(JSON.stringify(st.waiting) === JSON.stringify(people), 'waiting list (join order) ' + JSON.stringify(st.waiting));
  // a second tab with a name already waiting is asked for another name
  const dup = await post('assign', { device: 'other-tab', name: 'ann' });
  check(dup.code === 409 && dup.j.needName, 'duplicate lobby name should be refused: ' + JSON.stringify(dup));
  // kick from the lobby: gone from the list, and their retries are refused
  await post('assign', { device: 'Z', name: 'Zed' });
  st = await status();
  const zKey = st.waitingDetail.find((x) => x.name === 'Zed')?.key;
  await admin('lobby/kick', { key: zKey });
  const zBack = await post('assign', { device: 'Z', name: 'Zed' });
  check(zBack.code === 403, 'kicked lobby member should be refused: ' + zBack.code);
  st = await status();
  check(!st.waiting.includes('Zed'), 'Zed still listed after kick');

  const start = await admin('lobby/start', { turnSeconds: 3, firstTurnSeconds: 0, seed: 11 });
  check(start.code === 200 && start.j.players.length === 5, 'start should build a 5-player session: ' + JSON.stringify(start.j).slice(0, 200));
  check(start.j.status === 'lobby', 'should wait for everyone to connect');
  check(start.j.schedule.every((rd) => rd.sitout), 'odd group: someone sits out every round');
  const again = await admin('lobby/start', {});
  check(again.code === 409, 'second start should be refused');

  // their pages retry /api/assign and get their slot, then connect
  const slot = {};
  for (const [i, n] of people.entries()) {
    const r = await post('assign', { device: 'L' + i, name: n });
    check(r.code === 200 && r.j.name === n, `${n} should get a slot: ${JSON.stringify(r)}`);
    slot[n] = r.j.id;
  }
  check(new Set(Object.values(slot)).size === 5, 'distinct slots ' + JSON.stringify(slot));
  const late = await post('assign', { device: 'L9', name: 'Late' });
  check(late.code === 409, 'a name arriving after start should be told it is full: ' + late.code);
  for (const n of people.slice(0, 4)) await post('c', { id: slot[n], token: 't' + n, since: 0, msgs: [{ type: 'join', id: slot[n] }] });
  st = await status();
  check(st.status === 'lobby', 'must not start with one person missing');
  await post('c', { id: slot.Eli, token: 'tEli', since: 0, msgs: [{ type: 'join', id: slot.Eli }] });
  await sleep(300);
  st = await status();
  check(st.status === 'running', 'should start once all 5 are connected, got ' + st.status);
  check(st.players.every((p) => p.name === people.find((n) => slot[n] === p.id)), 'names on dashboard');

  console.log('slots:', JSON.stringify(slot), '| status:', st.status);
  console.log(errs.length ? 'FAILED\n  ' + errs.join('\n  ') : 'ALL CHECKS PASSED');
  process.exit(errs.length ? 1 : 0);
})();
