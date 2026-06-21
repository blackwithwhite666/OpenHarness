# ohmo-skills — document / office / canvas / diagram skills

Skills the **ohmo** gateway loads from `~/.ohmo/skills` (via `extra_skill_dirs`) on the
agent host (`93.77.160.211`). They are **not** part of the `openharness-ai` wheel — they are
deployed as a directory tree alongside it. This folder is the version-controlled source.

Ported from [`anthropics/skills`](https://github.com/anthropics/skills)
(docx/pptx/xlsx/canvas-design/pdf) and Arcadia `alice/docs/.agents/skills/visual-explainer`
(visual-explainer), all **server-adapted**: each `SKILL.md` carries a
`Runtime on this server (ohmo)` banner (Python → shared venv, Node → per-skill `node_modules`,
or — for visual-explainer — headless delivery via Telegram attach).

## Skills

| Skill | What it does | Engine |
|---|---|---|
| `pdf` | **merge** of our extractor + Anthropic manipulation. Extract/OCR (`scripts/pdf-extract-ocr`, Codex-VLM) **and** merge/split/rotate/watermark/encrypt/create + fill forms (`scripts/*.py`, `FORMS.md`, `REFERENCE.md`). | venv Python (`pypdf`/`pdfplumber`/`reportlab`) + `codex` |
| `docx` | create / edit / read Word docs. | **docx-js (Node)** create · venv Python office scripts edit · `python-docx` read (no pandoc on host) |
| `pptx` | create / edit / read decks. | **pptxgenjs (Node)** create · venv Python scripts · `markitdown` read · soffice→pdf images |
| `xlsx` | create / edit / analyze spreadsheets, recalc formulas. | venv `openpyxl`/`pandas` · `scripts/recalc.py` via LibreOffice |
| `canvas-design` | poster / static-art `.png`/`.pdf` from a design philosophy. | venv `matplotlib`/`Pillow`/`reportlab` + bundled `canvas-fonts/` |
| `visual-explainer` | self-contained HTML diagrams / dashboards / data-tables (Mermaid, Chart.js, CSS). | **no venv/node** — pure HTML + browser-side CDN; delivered as a Telegram attachment (headless host) |

## Runtime model

- **Python** → one shared venv at `~/.ohmo/venvs/docs` built from `requirements.txt`
  (exact lock in `requirements.lock.txt`). The `SKILL.md` banners call
  `~/.ohmo/venvs/docs/bin/python` and run scripts **from the skill dir** so the local
  `office`/`helpers` packages import.
- **Node** (docx, pptx only) → per-skill `node_modules` (git-ignored; recreate with
  `npm install` in the skill dir — `package.json`/`package-lock.json` pin them). `require()`
  resolves from the **script's own dir**, so generators run with the skill dir as the script
  location or with `NODE_PATH=~/.ohmo/skills/<skill>/node_modules`.
- **visual-explainer** needs **neither venv nor Node** — it emits a self-contained `.html`
  (Mermaid/Chart.js/fonts load from CDN in the *viewer's* browser). On this headless host it is
  delivered by attaching the file to Telegram (`[[attach: …]]`, output dir `~/.agent/diagrams`);
  optional in-chat preview renders the `file://` URL to a PNG via the `browser` skill, and AI
  images use the `falai` skill (not `surf-cli`).

## Host system dependencies (already present on 93.77)

- **LibreOffice** (`soffice`) — docx/pptx/xlsx conversions, xlsx formula recalc, pptx thumbnails.
- **poppler** (`pdftoppm`, `pdfimages`) — PDF→image, image extraction.
- **node + npm** (v18 / 9) — docx-js / pptxgenjs.
- *Not installed / not required*: `pandoc` (use venv `python-docx`), `qpdf`/`pdftk`/`tesseract`/
  ImageMagick (use venv Python + the `ocr` skill).

## Deploy / refresh

```bash
# 1) venv (once, or to refresh deps)
python3.12 -m venv ~/.ohmo/venvs/docs
~/.ohmo/venvs/docs/bin/pip install -U pip wheel
~/.ohmo/venvs/docs/bin/pip install -r ohmo-skills/requirements.txt

# 2) skills → the gateway's skill dir
cp -r ohmo-skills/{pdf,docx,pptx,xlsx,canvas-design,visual-explainer} ~/.ohmo/skills/
mkdir -p ~/.agent/diagrams          # visual-explainer output dir

# 3) node deps for the two JS-creation skills
( cd ~/.ohmo/skills/docx && npm install )
( cd ~/.ohmo/skills/pptx && npm install )

# 4) reload the registry (idle-guarded restart)
export XDG_RUNTIME_DIR=/run/user/1000
systemctl --user restart ohmo-gateway.service
```

New skill dirs require the gateway restart to register; binary/SKILL.md-only swaps of an
existing skill also need a restart for the doc to reload.

## Licensing

`docx`, `pptx`, `xlsx`, `pdf` carry Anthropic's **"Proprietary — source-available, not open
source"** terms (see each `LICENSE.txt`). `canvas-design` and its bundled fonts are open
(OFL / see `LICENSE.txt`). `visual-explainer` is **MIT** (open; author nicobailon). They are
vendored here **as-is, with their LICENSE.txt unchanged**, for personal-VM deployment
provenance. Redistribution risk of the source-available skills is **accepted by the repo
owner** — do not strip or alter the bundled license files.
