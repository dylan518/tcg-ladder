// Vercel function: every /api/* request is rewritten here (see vercel.json).
const { handleApi } = require('../lib/api');

module.exports = (req, res) => handleApi(req, res);
