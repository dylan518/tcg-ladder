// Name login + admin lobby control. Run against a FRESH local server (no session yet):
//   ADMIN_KEY=test PORT=8094 node dev-server.js &   then   URL=http://localhost:8094 ADMIN_KEY=test node test/assign_test.js
const URL_ = process.env.URL || 'http://localhost:8094', KEY = process.env.ADMIN_KEY || 'test';
const post = async (p, b) => { const r = await fetch(`${URL_}/api/${p}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }); return { code: r.status, j: await r.json() }; };
const admin = (route, b = {}) => post(`${route}?key=${KEY}`, b);
const status = async () => (await fetch(`${URL_}/api/status?key=${KEY}`)).json();
const joinAs = (id, token) => post('c', { id, token, since: 0, msgs: [{ type: 'join', id }] });
const errs = [], check = (ok, msg) => { if (!ok) errs.push(msg); };
(async () => {
  // before any session: name is listed as waiting
  const pre = await post('assign', { device: 'w1', name: 'Alice' });
  check(pre.code === 404 && pre.j.retry && pre.j.waiting, 'pre-session assign should say waiting: ' + JSON.stringify(pre));
  const st0 = await status();
  check(!st0.session || (st0.waiting || []).includes('Alice'), 'waiting list missing Alice: ' + JSON.stringify(st0.waiting));

  await admin('tournament/setup', { mode: 'tournament', players: 4, name: 'assign', autoStart: false });
  const noName = await post('assign', { device: 'd0' });
  check(noName.code === 400 && noName.j.needName, 'no name should be refused');
  const names = ['Alice', 'Bob', 'Cara', 'Dan'], got = {};
  for (let i = 0; i < 4; i++) {
    const r = await post('assign', { device: i === 0 ? 'w1' : 'd' + i, name: names[i] });
    check(r.code === 200 && r.j.name === names[i], `${names[i]}: ${r.code} ${JSON.stringify(r.j)}`);
    got[names[i]] = r.j.id;
  }
  check(new Set(Object.values(got)).size === 4, 'duplicate slots ' + JSON.stringify(got));
  let st = await status();
  check(st.players.every((p) => p.name === names.find((n) => got[n] === p.id)), 'dashboard names wrong ' + JSON.stringify(st.players));
  check((st.waiting || []).length === 0, 'waiting list should be empty after Alice got a slot');
  const again = await post('assign', { device: 'd1', name: 'Bob' });
  check(again.j.id === got.Bob, 'same device got a different slot');
  const full = await post('assign', { device: 'd9', name: 'Eve' });
  check(full.code === 409, '5th player not refused: ' + full.code);

  // same name, other device: refused while online, takes the slot back while offline
  await joinAs(got.Cara, 'tokCara');
  const dupOnline = await post('assign', { device: 'cara-phone', name: 'cara' });
  check(dupOnline.code === 409 && dupOnline.j.needName, 'duplicate online name not refused: ' + JSON.stringify(dupOnline));
  const dupOffline = await post('assign', { device: 'dan-laptop2', name: 'Dan' });
  check(dupOffline.code === 200 && dupOffline.j.id === got.Dan, 'offline same name should take its slot back: ' + JSON.stringify(dupOffline));

  // rename, kick (device blocked, slot free), lock
  await admin('tournament/rename', { id: got.Bob, name: 'Bobby' });
  st = await status();
  check(st.players.find((p) => p.id === got.Bob).name === 'Bobby', 'rename failed');
  await admin('tournament/kick', { id: got.Cara });
  st = await status();
  const caraRow = st.players.find((p) => p.id === got.Cara);
  check(!caraRow.name && !caraRow.online, 'kick did not clear Cara: ' + JSON.stringify(caraRow));
  const back = await post('assign', { device: 'd2', name: 'Cara' });
  check(back.code === 403, 'kicked device should be refused: ' + back.code);
  await admin('tournament/lock', { locked: true });
  const locked = await post('assign', { device: 'd8', name: 'Fay' });
  check(locked.code === 403, 'locked lobby should refuse: ' + locked.code);
  await admin('tournament/lock', { locked: false });
  const fay = await post('assign', { device: 'd8', name: 'Fay' });
  check(fay.code === 200 && fay.j.id === got.Cara, 'Fay should get the freed slot: ' + JSON.stringify(fay));

  // kick everyone: names cleared
  await admin('tournament/kickall');
  st = await status();
  check(st.players.every((p) => !p.name), 'kickall left names');

  await admin('tournament/setup', { mode: 'oneonone', playerA: 'Dylan', playerB: 'Pav', rounds: 2, autoStart: true });
  const a = await post('assign', { device: 'x1', name: 'One' }), b = await post('assign', { device: 'x2', name: 'Two' });
  check(new Set([a.j.id, b.j.id]).size === 2 && [a.j.id, b.j.id].every((x) => ['Dylan', 'Pav'].includes(x)), 'one-on-one assignment wrong');
  console.log('slots:', JSON.stringify(got), '| one-on-one:', a.j.id, b.j.id);
  console.log(errs.length ? 'FAILED\n  ' + errs.join('\n  ') : 'ALL CHECKS PASSED');
  process.exit(errs.length ? 1 : 0);
})();
