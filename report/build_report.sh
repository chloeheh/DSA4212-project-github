#!/usr/bin/env bash
# Compile the submission from its self-contained source folder.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
if command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs --keep-intermediates --outdir build main.tex
elif command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
elif [[ -x /private/tmp/maestro-tectonic/tectonic ]]; then
    # Temporary compiler downloaded while preparing this report on the current machine.
    /private/tmp/maestro-tectonic/tectonic --keep-logs --keep-intermediates --outdir build main.tex
else
    echo 'No compiler found. Install Tectonic or a TeX distribution with latexmk, or compile main.tex on Overleaf.' >&2
    exit 1
fi
cp build/main.pdf main.pdf
printf 'Updated report/main.pdf\n'
