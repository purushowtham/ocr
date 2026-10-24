# Ink & Type: English handwriting OCR

A local web interface for converting English handwritten lines and clean, single-column pages into editable text. The project uses Microsoft's TrOCR-small handwritten recognizer, image preprocessing, and heuristic page segmentation. Users can inspect line crops, correct the transcript, and download text.

## Run the interface

Use Python 3.12. From the repository root:

```bash
cd Handwriting_Studio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ui_requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='microsoft/trocr-small-handwritten', local_dir='base_model')"
PORT=7861 python app.py
```

Open http://127.0.0.1:7861/ after startup completes. On Windows, activate with `.venv\Scripts\activate`, set `$env:PORT=7861` in PowerShell, and run `python app.py`.

Model weights and training images are downloaded separately; they are not committed to Git. The original local project folder contains the exported data used in the experiment. A fresh clone has no bundled sample images. Loading the model for the first time requires an internet connection and sufficient free disk space for the model and Python libraries.

## Training and notebooks

- `Handwriting_Studio/Train_Model.ipynb`: Colab training workflow; select a T4 GPU.
- `Handwriting_Studio/Launch_UI.ipynb`: notebook interface workflow.
- `Handwriting_Studio/train_corrected.py`: corrected experimental training script, designed for Colab `/content` paths.
- `Handwriting_Studio/train_ocr.py`: earlier experiment, retained for provenance.
- `Handwriting_Studio/ocr_page.py`: page segmentation and recognition.

The notebooks contain their own setup steps. Training downloads the Teklia IAM-line mirror and exports the selected images, labels, and metrics. Saved execution outputs are cleared in the published notebooks; measured reports are included separately.

## Measured results

The corrected experiment used 6,482 training lines, 256 validation lines, and 512 test lines. These are line-level results from `Handwriting_Studio/reports/corrected_metrics.json`:

| Model | Character error rate | Word error rate | Exact-line accuracy |
| --- | ---: | ---: | ---: |
| Original pretrained model, used by the interface | 5.9206% | 13.6304% | 40.8203% |
| Corrected fine-tuning, selected epoch 1 | 22.8084% | 38.1608% | 2.7344% |

Fine-tuning performed worse in this experiment. The corrected checkpoint weights were not recovered after the Colab runtime expired. The earlier archived checkpoint and its reports describe a different, misconfigured experiment. They must not be associated with the corrected scores. The web interface uses the original pretrained model.

Full-page accuracy has not been measured. Multiple columns, tables, overlapping writing, and strong perspective distortion are unsupported. The pretrained model already had IAM exposure, and this experiment does not establish performance on an independent unseen corpus.

## Research paper and data sources

`Research_Paper/` contains the IEEE-style PDF, editable LaTeX, and the limited manuscript review report. The review is not a certified plagiarism scan or venue acceptance certification.

- TrOCR model: https://huggingface.co/microsoft/trocr-small-handwritten
- Dataset mirror: https://huggingface.co/datasets/Teklia/IAM-line
- Original IAM database and terms: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

Observe the original IAM research-use terms and citation requirements. This repository does not grant rights to the third-party model or dataset; consult their respective terms.
