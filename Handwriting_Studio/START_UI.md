# Ink & Type

Current UI: http://127.0.0.1:7861

To restart later, double-click Start_UI.command, or run python app.py in an environment with ui_requirements.txt installed.

The UI uses base_model, the original pretrained TrOCR-small checkpoint. Its measured CER was 5.92% and WER was 13.63% on 512 test lines. The corrected fine-tuning experiment regressed to 22.81% CER and 38.16% WER. Its metrics are in reports/corrected_metrics.json; its weights could not be recovered after Colab expired. The model folder contains the earlier misconfigured checkpoint with 47.96% CER and is not used by the UI.

All 6482 training, 256 validation, and 512 test images and labels selected in the archived run are under data/. The original checkpoint's complete historical pretraining corpus is not included.

Use upright, clean, single-column English pages. Edit the transcription and click Save edited text before downloading. Check the line preview for segmentation errors. Real-page accuracy is unmeasured. Original IAM terms specify non-commercial research use.

The IEEE-style paper and editable LaTeX are in the sibling Research_Paper folder, with the supplied author details. Review the paper against the intended venue requirements before submission.
