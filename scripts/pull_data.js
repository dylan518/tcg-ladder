// Download a session from the deployment into data/<session>/ in the original layout, so
// analysis/label_strategies.py and the course's read_json_tcg.py work unchanged.
// Usage: node scripts/pull_data.js [session-id | --list]   (default: the active session)
//   env: URL (default https://tcg-ladder.vercel.app), ADMIN_KEY (default: contents of .admin_key)
const fs = require('fs');
const path = require('path');
const URL_ = process.env.URL || 'https://tcg-ladder.vercel.app';
const KEY = process.env.ADMIN_KEY || fs.readFileSync(path.join(__dirname, '..', '.admin_key'), 'utf8').trim();
const arg = process.argv[2];

(async () => {
  const get = async (p) => { const r = await fetch(`${URL_}/api/${p}${p.includes('?') ? '&' : '?'}key=${encodeURIComponent(KEY)}`); const j = await r.json(); if (!r.ok) throw new Error(j.error); return j; };
  if (arg === '--list') return (await get('sessions')).sessions.forEach((s) => console.log(s));
  const { session, files } = await get('export' + (arg ? `?session=${encodeURIComponent(arg)}` : ''));
  const dir = path.join(__dirname, '..', 'data', session);
  for (const [name, content] of Object.entries(files)) {
    const f = path.join(dir, name);
    if (!f.startsWith(dir)) continue;
    fs.mkdirSync(path.dirname(f), { recursive: true });
    fs.writeFileSync(f, content);
  }
  console.log(`${Object.keys(files).length} files -> ${dir}`);
})().catch((e) => { console.error(e.message); process.exit(1); });
