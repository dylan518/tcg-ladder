#!/usr/bin/env python3
"""Rewrite the prose of report.tex in a humanistic, Bostrom-inspired register with GPT-6.2, paragraph by paragraph.
Safety: every number, \\cite/\\ref key, and math snippet must survive unchanged; otherwise the original paragraph is kept.
Usage: python3 humanize.py report.tex report_humanistic.tex"""
import concurrent.futures as cf
import json
import os
import re
import sys
import urllib.request

MODEL = 'gpt-6.2-sol'
KEY = os.environ['OPENAI_API_KEY']

STYLE = """You are revising one paragraph of an undergraduate psychology research report (APA style, for a course on
human interaction) into a more humanistic register, inspired by the essay style of the philosopher Nick Bostrom.
Do not quote or imitate any specific passage of his; emulate only these general features of his prose:
- Lucid, measured, analytically careful sentences; plain words chosen with precision rather than jargon.
- Concepts introduced with crisp definitions and clean distinctions; possibilities laid out as an orderly set.
- Occasionally, a brief, apt analogy or thought experiment that makes an abstract point concrete (at most one per
  paragraph, and only where it genuinely helps; never in Method or Results).
- A mix of long, balanced sentences and short, decisive ones; dry, understated wit at most once in a while.
- Calibrated epistemic humility: claims are as strong as the evidence and no stronger.
- A willingness to step back and note why a point matters for people more broadly, then return to the argument.
Hard rules:
- Keep EVERY number, percentage, statistic, confidence interval, p value, and sample size exactly as written.
- Keep every LaTeX command exactly: \\parencite{...}, \\textcite{...}, \\ref{...}, \\emph{...}, \\enquote{...}, math in $...$,
  and \\item[...] labels. Do not add, remove, or change citations. Do not add new facts, findings, or claims.
- Keep roughly the same length (within 25%). It is a scientific report: no hype, no rhetorical flourishes that
  overstate, no first-person singular, no em dashes.
- Output ONLY the revised LaTeX paragraph, nothing else."""

INTENSITY = {
    'Introduction': 'Section: Introduction. A full humanistic revision is welcome.',
    'Discussion': 'Section: Discussion. A full humanistic revision is welcome.',
    'Abstract': 'Section: Abstract. Light touch: improve rhythm and clarity only; keep it compact.',
    'Method': 'Section: Method. Light touch only: smoother, clearer sentences; keep it plain and procedural.',
    'Results': 'Section: Results. Light touch only: smoother sentences; keep every statistic in place; no analogies.',
}

TOK = re.compile(r'\\(?:paren|text)?cite\{[^}]*\}|\\ref\{[^}]*\}|\$[^$]*\$|\\item\[[^\]]*\]|\d[\d,.]*')


def tokens(s):
    return sorted(TOK.findall(s))


def call(prompt, para):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'system', 'content': STYLE},
                                                     {'role': 'user', 'content': prompt + '\n\nParagraph:\n' + para}]}).encode()
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions', body,
                                 {'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)['choices'][0]['message']['content'].strip()


def rewrite(job):
    sec, para = job
    want = tokens(para)
    for attempt in range(3):
        try:
            out = call(INTENSITY[sec], para)
        except Exception as e:  # network/transient
            print('retry', e, file=sys.stderr)
            continue
        out = out.strip('`').removeprefix('latex').strip()
        if tokens(out) == want:
            return out, 'ok'
    return para, 'kept-original'


def main(src, dst):
    text = open(src).read()
    # abstract lives in \abstract{...} in the preamble
    m = re.search(r'\\abstract\{(.*?)\}\n\\keywords', text, re.S)
    head, body = text.split('\\begin{document}', 1)
    body, tail = body.split('\\printbibliography', 1)
    blocks = re.split(r'(\n\s*\n)', body)
    sec, jobs, idx = None, [], []
    in_float = False
    for i, b in enumerate(blocks):
        s = b.strip()
        for name in INTENSITY:
            if re.search(r'\\section\{' + name, s):
                sec = name
        if '\\begin{figure' in s:
            in_float = True
        if in_float:
            if '\\end{figure' in s:
                in_float = False
            continue
        if not s or s.startswith(('\\section', '\\subsection', '\\subsubsection', '%', '\\maketitle', '\\begin{enumerate}', '\\end{')) or sec is None:
            continue
        if len(s) < 200 and not s.startswith('\\item'):
            continue
        jobs.append((sec, b)); idx.append(i)
    if m:
        jobs.append(('Abstract', m.group(1))); idx.append('abs')
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(rewrite, jobs))
    log = []
    for i, (new, status), (sec, old) in zip(idx, res, jobs):
        log.append({'section': sec, 'status': status, 'old': old, 'new': new})
        if i == 'abs':
            head = head.replace(old, new)
        else:
            blocks[i] = new
    open(dst, 'w').write(head + '\\begin{document}' + ''.join(blocks) + '\\printbibliography' + tail)
    json.dump(log, open(dst.replace('.tex', '_log.json'), 'w'), indent=1)
    print({s: sum(1 for l in log if l['status'] == s) for s in ('ok', 'kept-original')})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
