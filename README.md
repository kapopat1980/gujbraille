# gujbraille: printed Braille to Gujarati text

Reference implementation of the method described in

> N. Jariwala and K. Popat, *A Proposed Model to Convert Printed Braille Text into Gujarati as an Aid for Blind People* (revised manuscript).

The pipeline takes a scanned page of Gujarati Braille (embossed, or printed in ink) and produces Gujarati Unicode text. It has four parts:

1. **Pre-processing.** Grey-scale conversion, median denoising, dot detection (Otsu for ink dots; shadow-based detection for embossed dots under the scanner's oblique light), size filtering derived from the physical dot diameter, and skew correction from the projection profile of the dot centroids.
2. **Grid segmentation.** A regular Braille grid (dot pitch, cell pitch, line pitch) is fitted to the dot centroids. Every dot is then assigned to a line, a cell and a dot position. Blank cells give the word boundaries. Each cell is cut out as a fixed, grid-anchored 28×28 window, so the position of the dots inside the cell is preserved.
3. **CNN cell classifier.** Nine Keras layers and 117,440 parameters. The network predicts one of the 64 six-dot patterns.
4. **Rule-based Gujarati mapping.** Context rules (consonant + vowel → matra, virama conjuncts, ક્ષ/જ્ઞ, two-cell ઋ, number sign) turn the cell sequence into Unicode Gujarati.

Every pixel threshold is computed from the physical Braille geometry (in mm) and the scan resolution, so the code is not tuned to one image size.

## Install

```bash
python -m pip install -e ".[test]"
pytest            # 36 unit tests, < 1 minute
```

Tested with Python 3.13, TensorFlow 2.21, OpenCV 5.0 and scikit-learn 1.9 on CPU.

## Reproducing the paper (ink-printed Braille on A4)

```bash
# 1. ground truth: Gujarati source text -> Braille pages that fit A4 at standard size (30 cells x 25 lines)
gujbraille prepare --text sources/SOCSCI-STD10-CH08.txt --out data/real/gt --prefix SOCSCI-STD10-CH08 --cells 30 --lines 25

# 2. one print-ready PDF with every page at true size (dot pitch 2.5 mm, cell 6.0 mm, line 10.0 mm, dot 1.5 mm)
gujbraille printpdf --gt data/real/gt --out print/braille_pages.pdf --per-doc 3
#    --per-doc 3 prints the 3 fullest pages of each chapter (51 pages for 17 chapters)
#    print at 100 % / "Actual size", black; capture with a flatbed scanner (300 dpi grey)
#    or a phone document-scanning app that crops the sheet (then add --auto-dpi below)

# 3. identify each scanned sheet (any order, upside-down allowed) and write images/ + pages.csv
gujbraille import-scans --scans scans/ --gt data/real/gt --out data/real --only print/braille_pages_order.txt [--auto-dpi]

# 4. segment every page, align it with its ground truth, save labelled cells
gujbraille build --manifest data/real/pages.csv --out work/real --dpi 300

# 5. baselines + CNN (5 seeds, document-level split) + end-to-end CER/WER
gujbraille run --data work/real --manifest data/real/pages.csv --out results/real

# 6. robustness: degrade the real test-page scans (resolution, blur, noise, faded toner) and re-read them
gujbraille stress --manifest data/real/pages.csv --results results/real --out results/real/stress
```

The paper's scans were captured at 150 dpi, so add `--dpi 150` to `import-scans`, `build`, `run` and `stress`.

For embossed paper, prepare 40 x 25 pages, send the `.brf` files to the embosser without translation, and use `--material emboss` in `import-scans`.

`results/real/summary.md` holds the tables reported in the paper. `results.json` holds every run, the per-class metrics and the software environment. `split.json` records which documents went to train, validation and test. For each seed there is a folder with the trained model, the training history, the predictions and the confusion matrix.

Convert a single page:

```bash
gujbraille convert --image page.png --material emboss --model results/real/cnn_grid/seed0/model.keras
```

## Software check without real data

`scripts/smoke_test.sh` renders **synthetic** pages and runs the whole protocol in about 2 minutes. It only checks that the software works. Its numbers say nothing about recognition of real Braille and must not be reported as results.

## Repository layout

| Path | Contents |
|---|---|
| `src/gujbraille/braille_table.py` | Gujarati Braille table, dot-mask helpers, BRF export |
| `src/gujbraille/transliterate.py` | rule-based Gujarati ↔ Braille conversion (rules R1–R7) |
| `src/gujbraille/segment.py` | pre-processing, dot detection, deskew, grid fitting, cell crops |
| `src/gujbraille/model.py` | CNN definition and data augmentation |
| `src/gujbraille/dataset.py` | page manifest → aligned, labelled cells + QA report |
| `src/gujbraille/experiment.py` | split, baselines, CNN training, metrics, end-to-end CER/WER |
| `src/gujbraille/prepare.py` | ground-truth page preparation (text → paginated Braille, .brf) |
| `src/gujbraille/printing.py` | true-size print PDF and automatic identification of scanned sheets |
| `src/gujbraille/stress.py` | robustness experiment on degraded real scans |
| `docs/table1_braille_gujarati.csv` | Table 1 of the paper, generated from the code |
| `docs/DATASET_PROTOCOL.md` | how the dataset is created, labelled, split and released |

## Data

The dataset (scans, ground truth, page manifest, QA report and split) is deposited at **Zenodo, DOI: 10.5281/zenodo.XXXXXXX** *(to be completed)*.

## Licence

Code: MIT. Data: see the Zenodo record.
