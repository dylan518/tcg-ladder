// Connection to the game server by polling (works on any host, including Vercel).
// link.join(id) starts it; link.send(msg) queues a message (sent in order); onMessage(m)
// receives every server message once, in order. Polls fast while in a game, slowly otherwise.
function createLink(onMessage) {
  let token;
  try { token = sessionStorage.getItem('tcg-token'); } catch (e) { /* storage blocked */ }
  if (!token) {
    token = (crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now().toString(36));
    try { sessionStorage.setItem('tcg-token', token); } catch (e) { /* fine */ }
  }
  let id = null, since = 0, queue = [], sending = false, hot = false, pollTimer = null, polling = false, gen = 0, joining = false;

  function deliver(r) {
    if (r.direct) r.direct.forEach(onMessage);
    for (const x of r.msgs || []) if (x.s > since) { since = x.s; onMessage(x.m); }
    if (typeof r.hot === 'boolean') hot = r.hot;
  }

  async function flush() {
    if (sending || !queue.length || !id) return;
    sending = true;
    const msgs = queue.splice(0, queue.length), myGen = gen;
    try {
      const res = await fetch('/api/c', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id, token, since, msgs }) });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      if (myGen === gen) { deliver(await res.json()); if (msgs.some((m) => m.type === 'join')) joining = false; }
    } catch (e) {
      if (myGen === gen) queue.unshift(...msgs); // retry in order
      await new Promise((r) => setTimeout(r, 500));
    }
    sending = false;
    if (queue.length) flush();
  }

  async function poll() {
    clearTimeout(pollTimer);
    if (!id) return;
    if (polling || joining) return schedule(); // the join reply carries the first messages
    polling = true;
    const myGen = gen;
    try {
      const res = await fetch(`/api/c?id=${encodeURIComponent(id)}&token=${encodeURIComponent(token)}&since=${since}`, { cache: 'no-store' });
      if (res.ok && myGen === gen) {
        const r = await res.json();
        deliver(r);
        if (r.rejoin && !joining && id) { joining = true; send({ type: 'join', id }); } // timed out (e.g. laptop slept): join again
      }
    } catch (e) { /* network blip: keep polling */ }
    polling = false;
    schedule();
  }
  function schedule() { clearTimeout(pollTimer); if (id) pollTimer = setTimeout(poll, document.hidden && !hot ? 1500 : hot ? 120 : 600); }

  function send(m) { queue.push(m); flush(); }

  function join(newId) {
    gen++; id = newId; since = 0; queue = []; joining = true;
    send({ type: 'join', id });
    schedule();
  }

  function stop() { gen++; id = null; queue = []; joining = false; clearTimeout(pollTimer); }

  // tell the server right away when the page closes, so a partner is not left waiting
  addEventListener('pagehide', () => {
    if (id && navigator.sendBeacon) navigator.sendBeacon('/api/c', new Blob([JSON.stringify({ id, token, since, msgs: [{ type: 'leave' }] })], { type: 'text/plain' }));
  });

  return { join, send, stop, get id() { return id; } };
}
