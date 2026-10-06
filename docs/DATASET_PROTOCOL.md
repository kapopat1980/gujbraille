# Dataset protocol: Gujarati Braille page images with exact ground truth

The aim is a dataset whose origin can be documented end to end. Every label is known *before* the page is scanned, and nothing is labelled by the recogniser itself.

## 1. Source texts
- Use Gujarati educational prose in the same subjects as before (history, civics, geography). Record title, standard, chapter, edition and page range for every source, and give each chapter a `doc_id` (e.g. `HIST-STD8-CH01`).
- **Permissions.** For textbook material, get written permission to reproduce and redistribute the Braille page images from the copyright holder (e.g. the textbook board). Alternatively, use openly licensed text (e.g. CC BY-SA Gujarati Wikipedia articles on the same subjects) for the publicly released part. Record the licence of every source.
- Target size: **at least 20 chapters / 200 pages** (roughly 100,000 cells). This allows a document-level split with several test documents. Absolute minimum: 15 chapters / 100 pages.

## 2. Ground-truth generation
1. Type or copy the source text as UTF-8 Gujarati and proofread it.
2. Run `gujbraille prepare --text <file> --prefix <doc_id> --out gt/` to get `<page>.guj.txt`, `<page>.brl.txt` and `<page>.brf`. The pages are 40 cells × 25 lines. Change `--cells/--lines` to match the embosser.
3. A Braille teacher checks a sample of pages against the Bharati/Gujarati Braille standard and signs a verification sheet (name, date, pages checked, corrections made). Keep the sheet.
4. Emboss exactly the `.brf` files. If the school's own transliteration software is used instead, save the exact file sent to the embosser and convert it with `from_brf`.

## 3. Production and scanning
- **Embossed set (primary).** Use the school's embosser (record make/model), standard Braille paper (record gsm), single-sided. Optionally emboss an interpoint (double-sided) subset as a harder test set.
- **Ink set (optional, secondary).** Print the same Braille on a laser printer for the "printed Braille for sighted readers" use case.
- **Scanning.** Use a flatbed scanner (record make/model) at **300 dpi**, 8-bit greyscale, lossless PNG/TIFF, with automatic enhancement switched OFF. Use the same sheet orientation every time. Do not press the lid hard, because it flattens the dots. Lower resolutions lose dots on embossed paper (see the unit tests).
- Measure the actual dot pitch, cell pitch, line pitch and dot diameter on 10 cells with a ruler or loupe, and record the mean values.
- Optionally capture a phone-camera subset as an out-of-distribution test.

## 4. Manifest and automatic labelling
- Fill in `pages.csv` (see `data/pages_template.csv`) with one row per scan.
- Run `gujbraille build`. Each page is segmented, and its detected cell sequence is aligned with the ground-truth sequence by minimum edit distance. Each aligned cell takes its label from the ground truth.
- `qa_pages.csv` reports, per page: coverage of ground-truth cells, agreement of the rule-based dot reading with the ground truth, skew, and rejected dots. Pages below 90% coverage are excluded automatically. Inspect them by hand and report the number excluded.
- **Manual verification.** Two people independently check a random 2% sample of cell crops against their labels. Report the sample size and the agreement.

## 5. Split
- Split by `doc_id` (chapter), 70/10/20, with fixed seed 2026 (`split.json`). No page or chapter appears in more than one partition.
- Report the cell count per class in each partition (from `cells.csv`).

## 6. Release (data availability)
- Deposit on Zenodo: the scans, `gt/`, `pages.csv`, `qa_pages.csv`, `split.json`, the measured geometry, this protocol, the data card and the licence. Reserve the DOI before submission.
- Release the code on GitHub and archive the tagged release on Zenodo, so it has its own DOI.
- If part of the text cannot be redistributed, release that part only on request under the same terms. State this explicitly in the Data Availability statement.

## 7. Data card (fill in and include in the deposit)
| Field | Value |
|---|---|
| Creators | |
| Collection period | |
| Source documents (count, subjects, standards) | |
| Text licence(s) / permissions | |
| Embosser / printer (make, model) | |
| Paper | |
| Scanner (make, model), resolution, format | |
| Measured geometry (dot, cell, line pitch; dot diameter) | |
| Pages (total / used after QA) | |
| Cells (total; per partition) | |
| Classes present | |
| Ground-truth verification (who, sample size, corrections) | |
| Manual crop check (sample size, agreement) | |
| Known limitations | |
| Licence of the dataset | |
| DOI | |
