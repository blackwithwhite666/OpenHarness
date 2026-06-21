---
name: pdf
description: >
  Work with PDF files end to end. EXTRACT/ANALYZE: read the text layer, extract embedded
  images or rasterize scanned pages, run OCR/VLM via the `ocr` skill / Codex CLI, and return a
  Markdown report. MANIPULATE/CREATE: merge, split, rotate, watermark, extract images,
  encrypt/decrypt, extract tables, and create new PDFs. FORMS: detect fillable fields and fill
  them, or annotate non-fillable forms. Trigger on "разбери PDF", "извлеки текст из PDF",
  "PDF в картинки", scanned/lab PDFs, "объедини PDF", "разбей PDF", "поверни страницы",
  "водяной знак", "заполни форму PDF", "создай PDF", or any request to merge/split/rotate/
  watermark/encrypt/fill/create a .pdf.
user-invocable: true
allowed-tools: Bash(*), Read(*), Write(*), Glob(*)
argument-hint: "<pdf-path> [--out report.md] [--prompt '...']"
---

# PDF — extraction, OCR, manipulation & forms

This skill has two engines:

- **Extraction + OCR** — text layer, scanned pages, embedded images → a Markdown report.
  Driven by the `pdf-extract-ocr` helper (`pypdf` + `pdftoppm` + Codex-VLM via the `ocr` skill).
- **Manipulation / creation / forms** — merge, split, rotate, watermark, extract images/tables,
  encrypt, create, and fill forms. Driven by `pypdf` / `pdfplumber` / `reportlab` and the form
  scripts in `scripts/`.

## Runtime on this server (IMPORTANT)

- For **manipulation/creation/forms**, run Python with the **docs venv** interpreter:
  **`~/.ohmo/venvs/docs/bin/python`** (it has `pypdf`, `pdfplumber`, `reportlab`, `pdf2image`,
  `pypdfium2`, `numpy`, `pandas`, `Pillow`). Bare `python3` does NOT have most of these.
- Run the **form scripts from the skill directory** so their imports resolve:
  `cd ~/.ohmo/skills/pdf && ~/.ohmo/venvs/docs/bin/python scripts/<tool>.py ...`
- The `pdf-extract-ocr` helper runs as-is (it shells `codex` for OCR) — no venv needed.
- CLI tools present on this host: `pdftotext`, `pdfinfo`, `pdftoppm`, `pdfimages` (poppler).
  **`qpdf` / `pdftk` / `tesseract` / ImageMagick are NOT installed** — use the venv Python
  (`pypdf`/`reportlab`/`Pillow`) and the `ocr` skill instead of those tools.

---

## 1) Extract / OCR  →  `pdf-extract-ocr`

Use this for PDFs that may contain either a normal text layer, scanned pages, embedded
photos/images, or a mix of both.

The helper combines:

1. PDF text extraction via `pypdf`.
2. Redaction of obvious personal identifiers in the text layer: addresses, СНИЛС-like values.
3. Embedded image extraction from scanned PDFs.
4. Page rasterization fallback through `pdftoppm` when needed.
5. Image OCR/VLM using the `ocr` skill pattern: `codex exec -i`.
6. A final Markdown report file containing redacted text layer + OCR output.

Helper CLI:

```bash
~/.ohmo/skills/pdf/scripts/pdf-extract-ocr <pdf> [flags]
```

### When to use (extraction)

- extract text from a PDF; convert PDF pages to images; OCR a scanned PDF;
- parse medical/lab PDF documents; produce structured text from a PDF;
- combine PDF text layer and visual OCR; return the result as a file.

If the user sends medical/lab PDFs, use this before free-form interpretation.

### Workflow

1. **Collect PDFs** — Telegram attachment paths or user-provided paths; batch multiple PDFs
   with separate output files.
2. **Run the helper.**
   ```bash
   ~/.ohmo/skills/pdf/scripts/pdf-extract-ocr /path/to/input.pdf --out /tmp/input_report.md
   ```
   With a custom structured prompt:
   ```bash
   ~/.ohmo/skills/pdf/scripts/pdf-extract-ocr /path/to/input.pdf --out /tmp/report.md \
     --prompt 'Извлеки медицинские показатели: название, значение, единицы, референс, отклонение.'
   ```
3. **Rasterization mode** — default `--rasterize auto` (embedded images first, else `pdftoppm`).
   Force every page: `--rasterize always --dpi 300`. Disable: `--rasterize never`.
4. **Inspect the report** — the helper prints the Markdown path; read it before answering.
   Every important fact must cite a source page/image; rerun with a stricter `--prompt` or
   fewer `--max-images` if OCR is truncated/too broad.
5. **Return output** — summarize in chat (short tasks) or attach the report
   (`[[attach: /absolute/path/to/report.md]]`). For medical PDFs separate facts from
   interpretation; avoid definitive diagnosis.

### Helper flags

```text
pdf                         PDF path
--out PATH                  Output Markdown path
--work-dir DIR              Directory for extracted images and intermediate OCR result
--prompt TEXT               Custom OCR/VLM prompt
--prompt-file PATH          Read OCR/VLM prompt from file
--dpi N                     DPI for rasterization fallback, default 300
--rasterize auto|always|never
--max-images N              Limit images sent to Codex; useful if output truncates
```

Default output: `/tmp/pdf_ocr/<pdf-stem>/<pdf-stem>.md`.
Intermediate: `/tmp/pdf_ocr/<pdf-stem>/{embedded_images,pages}/`, `ocr_result.md`.

### Relation to `ocr` skill

This skill handles PDF-specific plumbing. For the actual visual extraction from images it
follows the `ocr` skill workflow and calls:

```bash
codex exec --skip-git-repo-check --sandbox read-only -i image1.jpg -i image2.jpg - < prompt.txt
```

Load `ocr` as well when you need detailed OCR prompt templates or image-only behavior.

---

## 2) Manipulate / create

Use `~/.ohmo/venvs/docs/bin/python` with `pypdf` / `reportlab` / `pdfplumber`. Either write a
short `.py` file and run it, or use a heredoc:
`~/.ohmo/venvs/docs/bin/python - <<'PY' … PY`.

**Merge**
```python
from pypdf import PdfReader, PdfWriter
w = PdfWriter()
for f in ["a.pdf", "b.pdf", "c.pdf"]:
    for p in PdfReader(f).pages:
        w.add_page(p)
with open("merged.pdf", "wb") as o:
    w.write(o)
```

**Split** (one file per page)
```python
from pypdf import PdfReader, PdfWriter
r = PdfReader("input.pdf")
for i, page in enumerate(r.pages):
    w = PdfWriter(); w.add_page(page)
    with open(f"page_{i+1}.pdf", "wb") as o:
        w.write(o)
```

**Rotate** — `page.rotate(90)` before `writer.add_page(page)`.
**Watermark** — `page.merge_page(PdfReader("watermark.pdf").pages[0])` per page.
**Encrypt** — `writer.encrypt("userpw", "ownerpw")` before writing.
**Decrypt** — `r = PdfReader("enc.pdf"); r.decrypt("pw")`.
**Extract images** — `pdfimages -all input.pdf /tmp/outdir/prefix` (poppler, no Python).
**Extract tables** — `pdfplumber` → `pandas` (`page.extract_tables()` → `pd.DataFrame`).

**Create** (reportlab)
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
c = canvas.Canvas("hello.pdf", pagesize=letter)
_, h = letter
c.drawString(100, h - 100, "Hello World!")
c.save()
```
For multi-page/flowing docs use `reportlab.platypus` (`SimpleDocTemplate`/`Paragraph`).
**Never** use Unicode sub/superscript glyphs in reportlab — use `<sub>`/`<super>` markup in
`Paragraph` (built-in fonts render those Unicode glyphs as black boxes).

See `REFERENCE.md` for advanced `pypdfium2` usage and troubleshooting.

---

## 3) Fill PDF forms  →  `FORMS.md`

If the user wants to fill a form, **read `FORMS.md` and follow it in order — do not skip ahead
to writing code.** Start by detecting fillable fields (from the skill dir):

```bash
cd ~/.ohmo/skills/pdf
~/.ohmo/venvs/docs/bin/python scripts/check_fillable_fields.py <file.pdf>
```

Then follow the **Fillable fields** or **Non-fillable fields** branch in `FORMS.md`. All form
scripts live in `scripts/` and must be run with `~/.ohmo/venvs/docs/bin/python` from the skill
directory. (Note: `FORMS.md` Approach B mentions ImageMagick `magick`/`convert` for zoom-crops —
not installed here; crop with `Pillow` in the venv instead.)

---

## Rules

- Do not rely only on visual OCR if a clean text layer exists; include both and cross-check
  important numbers.
- Do not invent unreadable values. Mark them as unclear.
- Cite the source page/image for important extracted facts (menu items, ticket seats, lab
  values, dates, prices, rules).
- When returning evidence, attach or point to the exact rendered page/image/crop containing the
  fact; do not attach a random first page.
- Treat medical/identity/financial PDFs as sensitive. Do not repeat unnecessary identifiers
  (addresses, СНИЛС, policy/passport numbers).
- For high-stakes medical/legal/financial material, say OCR is assistive and should be checked
  against originals.
- Do not install missing PDF tools without user approval. The docs venv
  (`~/.ohmo/venvs/docs`) already has everything this skill needs.

## Smoke test

```bash
# extraction engine (existing):
~/.ohmo/skills/pdf/scripts/pdf-extract-ocr /path/to/sample.pdf --out /tmp/sample_report.md --max-images 1
sed -n '1,120p' /tmp/sample_report.md

# manipulation/forms engine (venv):
cd ~/.ohmo/skills/pdf
~/.ohmo/venvs/docs/bin/python - <<'PY'
from reportlab.pdfgen import canvas
c = canvas.Canvas("/tmp/_pdf_smoke.pdf"); c.drawString(72, 720, "smoke"); c.save()
print("created /tmp/_pdf_smoke.pdf")
PY
~/.ohmo/venvs/docs/bin/python scripts/check_fillable_fields.py /tmp/_pdf_smoke.pdf
```
