# English handwritten-page OCR

See the repository-root README for installation, model download, and the corrected T4 experiment results in reports/corrected_metrics.json. CER and WER are edit rates (lower is better); case and punctuation count, whitespace is normalized. Exact-line accuracy counts perfectly transcribed lines. Validation CER selects the checkpoint. The interface uses the stronger original pretrained model. The exported dataset and model weights are available in the original local project folder and are excluded from this code repository.

Use after downloading the pretrained checkpoint: python ocr_page.py page.jpg --model base_model --out page_output

Review page_output/transcription.txt and the line crops. Page segmentation supports clean, upright, single-column pages only; real-page accuracy has not been measured. Base model was previously trained on IAM. See DATA_SOURCES.md for research-use terms.
