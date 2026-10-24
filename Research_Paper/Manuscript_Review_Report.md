# Manuscript review record

Reviewed: 17 October 2026. Scope: revised IEEE-style handwriting OCR manuscript.

## Authorship and writing

The three names and department details match the request, in the requested order. No institution, postal address, email, or author contribution was invented. All sections, including the abstract and conclusion, were rewritten for clear, direct academic prose. This was editorial assistance, not evidence that the text was written without AI assistance. Authors should review and approve the manuscript and follow the intended venue's disclosure rules.

## Limited originality and attribution checks

This is a limited wording and citation review, not a certified plagiarism report. Turnitin, iThenticate, Copyleaks, and proprietary submission databases were not available. No full manuscript was uploaded to a plagiarism or AI-detection service. No plagiarism percentage or AI-detector result is claimed.

The abstract and eight sections were compared, after lowercase alphanumeric tokenization, with the public Microsoft model card, Teklia dataset card, and the TrOCR paper abstract. Bibliography entries and the manuscript title were excluded from this comparison. The method looked for identical consecutive sequences of seven words. Source snapshots are included in the review evidence archive. Short common phrases, paraphrases, translated copying, images, uncited ideas, and material outside these three texts are not comprehensively assessed by this method.

Sources checked:

- https://huggingface.co/microsoft/trocr-small-handwritten/raw/main/README.md
- https://huggingface.co/datasets/Teklia/IAM-line/raw/main/README.md
- https://arxiv.org/abs/2109.10282 (abstract only for the token comparison)

Seven-word matches by source:

- model_card: 0 distinct matching sequences.
- dataset_card: 0 distinct matching sequences.
- trocr_abstract: 0 distinct matching sequences.

Four additional quoted-phrase web searches were performed:

- "The divergence between training loss and validation error shows why adaptation"
- "Artifact recovery is a further limitation" handwriting
- "This work implements an inspectable English handwriting transcription workflow"
- "The selection rule consequently chooses epoch one"

The returned results did not display exact matches to these quoted phrases. Some results were unrelated or approximate. This is a small search sample, not proof of uniqueness; search indexes omit private, unindexed, and subscription material.

Citations [1]-[5] are present in the text and correspond to the five reference entries. The model card and dataset card support the model purpose and data description; the original TrOCR abstract supports its attribution. The IAM article's title, authors, year, volume, pages, and DOI were checked against its publisher page: https://doi.org/10.1007/s100320200071. The IAM database page timed out in this review; its terms had been consulted during preparation and should be rechecked before redistribution: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database.

## Numerical and scientific checks

18 reported numerical values were checked against corrected_metrics.json; all agree at the displayed rounding. The epoch-one selection also agrees with minimum validation CER. These are consistency checks against the saved record, not an independent rerun of training.

- The measured regression is retained: baseline CER/WER 5.9206%/13.6304%; adapted CER/WER 22.8084%/38.1608%.
- Percentage-point increases, epoch losses, validation error rates, and the recorded elapsed interval agree with the saved metrics.
- The manuscript distinguishes line metrics from unmeasured real-page accuracy and identifies prior IAM exposure, unverified writer independence, and the single-seed design.
- The missing corrected checkpoint is disclosed. The earlier recovered checkpoint is not represented as the model responsible for corrected scores.
- No confidence interval, significance result, performance improvement, or new architecture is fabricated.

## Document checks and remaining requirements

The PDF has three US Letter pages with two text columns, the requested author block, numbered sections, equations, one table, one figure, and five references. Every rendered page was visually inspected for clipping, overlap, legibility, and reading flow. Author names were checked in both the PDF and editable source. PDF and LaTeX share the same revised prose.

The editable source uses IEEEtran conference format. A LaTeX compilation test was not available; the supplied PDF was generated separately in an IEEE-style layout. This is not a venue-specific PDF eXpress certification. A target venue has not been specified, so its exact author-block rules, page limit, anonymization requirements, and submission checks remain unverified. Institutional affiliation details have not been supplied.

No document can be guaranteed to pass every plagiarism checker, AI detector, peer review, or submission test. For an institutional similarity report, run the final version through the service supplied by the institution and review flagged passages in context. Do not treat a low similarity number as proof that all citations and claims are correct.
