// Local server: serves public/ and the same HTTP API the Vercel deployment uses (lib/api.js).
// Run: ADMIN_KEY=demo PORT=8080 node dev-server.js   (or npm start)
// Without DATABASE_URL the game state is kept in memory and data are written to data/<session>/.
// With DATABASE_URL (Neon) it shares state and data with the Vercel deployment.

const http = require('http');
const fs = require('fs');
const path = require('path');
process.env.TCG_LOCAL_DATA_DIR ||= path.join(__dirname, 'data');
const { handleApi, ADMIN_KEY, store } = require('./lib/api');

const PORT = Number(process.env.PORT || 8080);
const PUBLIC_DIR = path.join(__dirname, 'public');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  if (url.pathname.startsWith('/api/')) return handleApi(req, res);
  const p = url.pathname === '/' ? '/index.html' : url.pathname === '/admin' ? '/admin.html' : url.pathname === '/schedule' ? '/schedule.html' : url.pathname;
  const file = path.join(PUBLIC_DIR, path.normalize(p).replace(/^(\.\.[/\\])+/, ''));
  if (!file.startsWith(PUBLIC_DIR) || !fs.existsSync(file)) { res.writeHead(404); return res.end('not found'); }
  res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});

server.listen(PORT, () => {
  const lan = Object.values(require('os').networkInterfaces()).flat()
    .filter((a) => a && a.family === 'IPv4' && !a.internal).map((a) => a.address);
  console.log(`TCG ladder running on http://localhost:${PORT} (store: ${store.kind})`);
  for (const ip of lan) console.log(`Players on this Wi-Fi join at: http://${ip}:${PORT}`);
  console.log(`Experimenter dashboard: http://localhost:${PORT}/admin?key=${ADMIN_KEY}`);
});
