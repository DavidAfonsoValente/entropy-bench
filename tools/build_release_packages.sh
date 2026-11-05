#!/usr/bin/env bash
# Build every deliverable of a paper release from one committed revision:
#
#   COMPANY_paper_named.pdf          the preprint build (\anonfalse): author, affiliation, repo links
#   Overleaf_COMPANY_named.zip       its LaTeX source, compiles standalone (also the arXiv source)
#   TMLR_paper_ANONYMOUS.pdf         the submission build (\anontrue): no author, no links
#   Overleaf_TMLR_anonymous.zip      its LaTeX source
#   TMLR_supplementary_ANONYMOUS.zip the public tree, scrubbed of every identifying string, with the
#                                    anonymous PDF and no git history (TMLR: anonymised, <= 100 MB)
#   0_READ_ME_FIRST.txt              which file goes where
#
#   bash tools/build_release_packages.sh [REV] [OUT_DIR]      # defaults: HEAD, $SCRATCH/back_to_entropy_release
#
# The tree is exported with `git archive`, so uncommitted changes never reach a package, and the
# public exclusion list (tools/public_exclude.sh) is applied before anything is scrubbed. The scrub
# is a fixed table of replacements (ANON_RULES below) over every text file; the script then greps
# the scrubbed tree for the identifying strings and refuses to zip if one survives. Run the full
# gate on the scrubbed tree separately (make check inside it) before submitting.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REV="${1:-HEAD}"
OUT="${2:-${SCRATCH:-/tmp}/back_to_entropy_release}"
PYTHON="${PYTHON:-$ROOT/venv_lm_adapt/bin/python}"
# shellcheck source=tools/public_exclude.sh
source "$ROOT/tools/public_exclude.sh"

WORK="$(mktemp -d "${SCRATCH:-/tmp}/release_build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SHA="$(git -C "$ROOT" rev-parse --short "$REV")"
DATE="$(date -u +%F)"

echo "==> exporting $REV ($SHA) minus the internal paths"
mkdir -p "$WORK/named"
git -C "$ROOT" archive "$REV" | tar -x -C "$WORK/named"
( cd "$WORK/named" && git -C "$ROOT" ls-files | grep -E "$EXCLUDE_RE" | xargs -r rm -f )
find "$WORK/named" -type d -empty -delete
# The private map describes files the public tree withholds; regenerate it from what is here
# (repo_map reads `git ls-files`, so the export is staged in a throwaway repository first).
( cd "$WORK/named" && git init -q && git add -A && "$PYTHON" tools/repo_map.py >/dev/null && rm -rf .git )

build_pdf() {  # build_pdf DIR -> DIR/paper_sota.pdf
  ( cd "$1" && pdflatex -interaction=nonstopmode -halt-on-error paper_sota.tex >/dev/null \
    && bibtex paper_sota >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error paper_sota.tex >/dev/null \
    && pdflatex -interaction=nonstopmode -halt-on-error paper_sota.tex >/dev/null )
  grep -q "LaTeX Warning: There were undefined references" "$1/paper_sota.log" && { echo "undefined references in $1" >&2; exit 1; }
  return 0
}

overleaf_zip() {  # overleaf_zip DIR ZIP: the files the paper needs to compile, and nothing else
  local dir="$1" zip="$2"
  rm -f "$zip"   # zip adds to an existing archive instead of replacing it
  ( cd "$dir" && zip -q -r "$zip" paper_sota.tex paper_refs.bib tmlr.sty tmlr.bst fancyhdr.sty \
      $(ls ./*_table.tex) figures/*.pdf figures/*.tex 2>/dev/null )
}

echo "==> named build"
build_pdf "$WORK/named"
mkdir -p "$OUT"
cp "$WORK/named/paper_sota.pdf" "$OUT/COMPANY_paper_named.pdf"
overleaf_zip "$WORK/named" "$OUT/Overleaf_COMPANY_named.zip"

echo "==> anonymous tree"
cp -a "$WORK/named" "$WORK/anon"
rm -f "$WORK/anon/tools/build_release_packages.sh"   # its scrub table names what it removes
# The map was generated with that script present; regenerate it so the gate finds no dangling path.
( cd "$WORK/anon" && git init -q && git add -A && "$PYTHON" tools/repo_map.py >/dev/null && rm -rf .git )
"$PYTHON" - "$WORK/anon" <<'SCRUB'
import re, sys
from pathlib import Path

root = Path(sys.argv[1])
TEXT = {".py", ".md", ".tex", ".bib", ".sh", ".json", ".jsonl", ".yml", ".yaml", ".toml", ".cff",
        ".txt", ".cfg", ".ini", ".csv", ".sty", ".bst", ".mk", ".html", ".css", ".sbatch",
        ""}  # "" = LICENSE, dotfiles
# Order matters: the longer, more specific strings first.
ANON_RULES = [
    (r"https://github\.com/DavidAfonsoValente/entropy-bench(\.git)?", "https://ANONYMIZED-REPOSITORY"),
    (r"DavidAfonsoValente/entropy-bench", "ANONYMIZED-REPOSITORY"),
    (r"DavidAfonsoValente", "anonymous"),
    (r"David Afonso Prazeres Da Cruz Valente|David Afonso Valente|Valente, David Afonso", "Anonymous Authors"),
    (r"\bValente\b", "Anonymous"),
    (r"dvalente@bottlecapai\.com|dvalente_bottlecapai_com|dvalente", "anonymous"),
    (r"Bottlecap AI|BottleCapAI|bottlecapai|bottlecap", "anonymous"),
    (r"dprazere", "user"),
    (r"gcluster-rtx[a-z0-9-]*", "hpc-node"),
    (r"gpu-llm-training[a-z0-9-]*", "anon-bucket"),
    (r"AIFAC_[A-Z0-9_]+", "HPC_ACCOUNT"),
    (r"Leonardo HPC|Leonardo \(CINECA\)|[Cc][Ii][Nn][Ee][Cc][Aa]'s|[Cc][Ii][Nn][Ee][Cc][Aa]", "the HPC cluster"),
    (r"\bLeonardo\b", "the HPC cluster"),
    (r"leonardo_scratch|leonardo_work", "hpc_scratch"),
    (r"IS_LEONARDO", "IS_HPC"),
    (r"[Ll][Ee][Oo][Nn][Aa][Rr][Dd][Oo]", "hpc"),
    (r"valente2026backtoentropy", "anonymous2026backtoentropy"),
]
FORBIDDEN = re.compile("|".join(p for p, _ in ANON_RULES), re.I)

n = 0
for f in root.rglob("*"):
    if not f.is_file() or f.suffix not in TEXT or f.suffix == "" and f.stat().st_size > 2_000_000:
        continue
    try:
        s = f.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    t = s
    for pat, rep in ANON_RULES:
        t = re.sub(pat, rep, t)
    if t != s:
        f.write_text(t, encoding="utf-8")
        n += 1
print(f"  scrubbed {n} files")

tex = root / "paper_sota.tex"
s = tex.read_text()
s = s.replace(r"\newif\ifanon \anonfalse", r"\newif\ifanon \anontrue")
s = re.sub(r"\\author\{.*?\}\n", "\\\\author{\\\\name Anonymous authors}\n", s, count=1, flags=re.S)
tex.write_text(s)

readme = root / "README.md"
s = readme.read_text()
s = "\n".join(l for l in s.splitlines() if not l.startswith("[![")) + "\n"   # badges link the repo
s = s.replace("python -m pip install -e .\n", "python -m pip install -e .\ngit init -q && git add -A   # the repository-map check lists files with git\n", 1)
readme.write_text(s)

cff = root / "CITATION.cff"
if cff.exists():
    cff.write_text(re.sub(r'family-names: ".*?"\n\s*given-names: ".*?"', 'family-names: "Anonymous"', cff.read_text()))

left = []
for f in root.rglob("*"):
    if f.is_file() and f.suffix in TEXT:
        try:
            if FORBIDDEN.search(f.read_text(encoding="utf-8")):
                left.append(str(f.relative_to(root)))
        except UnicodeDecodeError:
            pass
if left:
    sys.exit("identifying strings survive in: " + ", ".join(left[:20]))
print("  no identifying string left")
SCRUB

echo "==> anonymous build"
build_pdf "$WORK/anon"
cp "$WORK/anon/paper_sota.pdf" "$OUT/TMLR_paper_ANONYMOUS.pdf"
overleaf_zip "$WORK/anon" "$OUT/Overleaf_TMLR_anonymous.zip"
# The supplementary is the scrubbed tree with the anonymous PDF and without build residue.
( cd "$WORK/anon" && rm -f paper_sota.aux paper_sota.bbl paper_sota.blg paper_sota.log paper_sota.out \
    && mkdir -p "$WORK/supp" && cp -a . "$WORK/supp/entropy-bench" )
rm -f "$OUT/TMLR_supplementary_ANONYMOUS.zip"
( cd "$WORK/supp" && zip -q -r "$OUT/TMLR_supplementary_ANONYMOUS.zip" entropy-bench )

cat > "$OUT/0_READ_ME_FIRST.txt" <<EOF
Back to Entropy: release package ($DATE, built from $SHA)

FOR THE COMPANY (named, public)
  Public code + paper:       https://github.com/DavidAfonsoValente/entropy-bench
  COMPANY_paper_named.pdf    the paper with author and affiliation (same as the repo's PDF)
  Overleaf_COMPANY_named.zip LaTeX source; upload to Overleaf as a new project, or post to arXiv

  Allowed: blog posts, announcements, social posts, arXiv, the public repo.
  Not allowed while the paper is at TMLR: sending this paper (or its text, figures or
  results) to any other peer-reviewed conference or journal.

FOR TMLR (anonymous, https://openreview.net/group?id=TMLR -> Submit)
  TMLR_paper_ANONYMOUS.pdf          upload as the submission PDF
  TMLR_supplementary_ANONYMOUS.zip  upload as supplementary material (limit 100 MB)
  Overleaf_TMLR_anonymous.zip       the anonymous source, for the camera-ready later

  Fill in the author fields normally (OpenReview hides them from reviewers).
  Do not mention the GitHub repo, the arXiv version or the company in any field.
  Do not upload the named files above to OpenReview.

Policy source: https://jmlr.org/tmlr/author-guide.html and
https://jmlr.org/tmlr/editorial-policies.html (preprints allowed at any time provided the
submission does not link to them; supplementary material must be anonymised, PDF or ZIP).
EOF

echo "==> done"
ls -la "$OUT"
