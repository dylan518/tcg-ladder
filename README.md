# tcg-ladder

Code for "How Groups Build Shared Understanding: Signalling Travels, Reading Stays Local in the Tacit Communication Game" (PSYC 63, Dartmouth College).

The Tacit Communication Game rebuilt as an online tournament in which the experimenter controls who is paired with whom, plus the analysis scripts and the report source.

- `engine.js`, `tournament.js`, `store.js`, `api/`, `public/` — the game server, tournament pairing, and browser client (`npm install && npm start`).
- `test/` — bot players for testing sessions.
- `analysis/` — preprocessing and analysis scripts (`analysis/models/` holds the neutral-reader, counterfactual, style-embedding, convergence, and generative-model code).
- `paper/` — report source (`report.tex`), bibliography, figure scripts, and compiled PDFs.

## Data

The game logs are not included, to protect participants' privacy. The analysis scripts expect them under `data/` and write intermediate results to `analysis/`; the logs are available from the author on request.
