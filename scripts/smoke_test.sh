#!/usr/bin/env bash
# End-to-end software check on SYNTHETIC pages (about 2-3 minutes on a laptop CPU).
# The numbers it prints are NOT results about real Braille and must not be reported.
set -euo pipefail
W=${1:-work/smoke}
gujbraille synth --out "$W/data" --pages 30 --docs 10 --dpi 200
gujbraille build --manifest "$W/data/pages.csv" --out "$W/cells" --dpi 200
gujbraille run --data "$W/cells" --manifest "$W/data/pages.csv" --out "$W/results" \
    --seeds 0 1 --epochs 8 --dpi 200
