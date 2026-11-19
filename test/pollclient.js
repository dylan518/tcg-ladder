// Socket-shaped client for the polling API, so the bot tests keep their ws-style code:
// new PollClient(url) emits 'open', 'message' (raw JSON string) and 'close'; send(json); close().
// close() just stops polling (like a dropped connection: the server notices after TCG_STALE_MS).
const { EventEmitter } = require('events');
const crypto = require('crypto');
const POLL_MS = Number(process.env.POLL_MS || 25);

class PollClient extends EventEmitter {
  constructor(url) {
    super();
    this.base = url.replace(/^ws/, 'http').replace(/\/$/, '');
    this.token = crypto.randomUUID();
    this.id = null; this.since = 0; this.queue = []; this.sending = false; this.readyState = 0; this.joining = false;
    setImmediate(() => { this.readyState = 1; this.emit('open'); });
  }
  deliver(r) {
    for (const m of r.direct || []) this.emit('message', JSON.stringify(m));
    for (const x of r.msgs || []) if (x.s > this.since) { this.since = x.s; this.emit('message', JSON.stringify(x.m)); }
  }
  send(raw) {
    if (this.readyState !== 1) return;
    const m = JSON.parse(raw);
    if (m.type === 'join') { this.id = m.id; this.since = 0; this.joining = true; this.loop(); }
    this.queue.push(m); this.flush();
  }
  async flush() {
    if (this.sending || !this.queue.length || this.readyState !== 1) return;
    this.sending = true;
    const msgs = this.queue.splice(0);
    try {
      const r = await fetch(this.base + '/api/c', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: this.id, token: this.token, since: this.since, msgs }) });
      if (this.readyState === 1) { this.deliver(await r.json()); if (msgs.some((m) => m.type === 'join')) this.joining = false; }
    } catch (e) { if (this.readyState === 1) { this.queue.unshift(...msgs); await new Promise((r) => setTimeout(r, 100)); } }
    this.sending = false;
    if (this.queue.length) this.flush();
  }
  async loop() {
    if (this.looping) return;
    this.looping = true;
    while (this.readyState === 1) {
      await new Promise((r) => setTimeout(r, POLL_MS));
      if (this.joining || this.readyState !== 1) continue;
      try {
        const r = await (await fetch(`${this.base}/api/c?id=${encodeURIComponent(this.id)}&token=${this.token}&since=${this.since}`)).json();
        if (this.readyState !== 1) break;
        this.deliver(r);
        if (r.rejoin && !this.joining) { this.joining = true; this.queue.push({ type: 'join', id: this.id }); this.flush(); }
      } catch (e) { /* retry */ }
    }
    this.looping = false;
  }
  close() { if (this.readyState === 1) { this.readyState = 3; this.emit('close'); } }
}

module.exports = PollClient;
