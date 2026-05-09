# Palmprint Recognition Using Siamese Neural Networks

Deep Learning (CS 4263-901)
Cameron Ortiz, Myar Nguyen, Edison La

---

## Project Overview

This project implements a **Siamese Neural Network (SNN)** for contactless palmprint verification using the **Birjand University Mobile Palmprint Database (BMPD)**.

Rather than treating palmprint recognition as a standard multi-class classification problem, this project uses a **verification-based approach**. The model receives two palmprint images and predicts whether the images belong to the same palm identity or to different identities.

The final pipeline includes:

- BMPD dataset preprocessing
- Palm region of interest (ROI) extraction
- Subject-level train, validation, and test splitting
- Balanced genuine and impostor pair generation
- Siamese CNN training with contrastive loss
- Verification evaluation using accuracy, ROC-AUC, EER, FAR/FRR, and ROC curve visualization

---

## Dataset

This project uses the **Birjand University Mobile Palmprint Database (BMPD)**.

Dataset link: [Birjand University Mobile Palmprint Database (BMPD) on Kaggle](https://www.kaggle.com/datasets/mahdieizadpanah/birjand-university-mobile-palmprint-databasebmpd)

The dataset is not included in this repository due to licensing and file size considerations. To run the project, download the BMPD dataset separately and place the raw images under:

```text
data/raw/
```

Only BMPD was used for the final experiments. PolyU and IITD were considered earlier as possible datasets, but they were not used in the final submitted implementation. Expanding to PolyU and IITD is listed as future work.

---

## Problem Formulation

This project focuses on **palmprint verification**.

Given two palmprint ROI images:

- **Genuine pair:** both images belong to the same palm identity
- **Impostor pair:** the images belong to different palm identities

The Siamese network learns an embedding space where genuine pairs are closer together and impostor pairs are farther apart.

In the final implementation, left and right hands are treated as separate palm identities. For example, a subject's left palm and right palm are not treated as the same biometric identity.

---

## Model Approach

The model uses a Siamese architecture with shared weights:

1. Each image is passed through the same CNN backbone.
2. The CNN produces a fixed-length embedding vector for each image.
3. The Euclidean distance between the two embeddings is calculated.
4. Contrastive loss is used during training to pull genuine pairs closer and push impostor pairs apart.

Triplet loss is not part of the final implementation. The repository may contain an early placeholder or experimental triplet sampling file, but the final training and evaluation workflow uses pair generation and contrastive loss.

---

## Project Structure

```text
palmprint-verification-siamese/
│
├── README.md
├── requirements.txt
│
├── checkpoints/
│   └── Saved model checkpoints. Large checkpoint files may not be committed.
│
├── data/
│   ├── raw/                  # Raw BMPD dataset, not included in submission
│   ├── processed/            # Processed ROI images
│   └── qc/                   # ROI quality control files, if generated
│
├── experiments/              # Experiment notes or configuration files
├── logs/                     # Training logs, if generated
├── reports/                  # Report figures, evaluation outputs, and visualizations
├── splits/                   # Saved split information, if generated
│
├── notebooks/
│   ├── 01_roi_extraction_experiments.ipynb
│   ├── 02_preprocess_bmpd_roi.ipynb
│   ├── 03_check_splits_and_pairs.ipynb
│   ├── 04_training_siamese.ipynb
│   └── 05_evaluation.ipynb
│
├── scripts/
│   ├── create_contact_sheets.py
│   ├── create_roi_quality_template.py
│   └── preprocess_bmpd_roi.py
│
└── src/
    └── palmprint/
        ├── datasets/
        │   ├── bmpd.py              # BMPD loading, split configuration, and quality filtering
        │   ├── pair_dataset.py      # PyTorch dataset for image pairs
        │   ├── roi_quality.py       # ROI quality filtering helpers
        │   ├── iitd.py              # Placeholder for future IITD support
        │   └── polyu.py             # Placeholder for future PolyU support
        │
        ├── preprocessing/
        │   ├── roi.py               # ROI extraction and preprocessing pipeline
        │   └── contact_sheet.py     # Contact sheet generation for ROI review
        │
        ├── sampling/
        │   ├── pairs.py             # Genuine/impostor pair generation
        │   └── triplets.py          # Experimental placeholder, not used in final workflow
        │
        ├── models/
        │   ├── backbone.py          # CNN embedding backbone
        │   ├── losses.py            # Contrastive loss
        │   └── siamese.py           # Siamese network wrapper
        │
        ├── training/
        │   ├── train.py             # Training helper functions
        │   ├── eval.py              # Evaluation helper functions
        │   └── metrics.py           # Verification metrics
        │
        └── utils/
            ├── seed.py              # Reproducibility helpers
            ├── config.py            # Reserved for future configuration helpers
            └── logging.py           # Reserved for future logging helpers
```

---

## Installation and Requirements

This project was developed using Python and PyTorch. The repository includes a `requirements.txt` file with the Python package dependencies used for the final project.

### 1. Create a virtual environment

On Linux, macOS, or WSL:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make local source imports available

When running notebooks, the notebooks add the local `src/` folder to `sys.path`. If running scripts manually, use one of the following approaches from the project root:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Or on Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
```

---

## How to Run the Final Demo Workflow

The final workflow is run through the notebooks in the `notebooks/` directory. Open Jupyter Notebook or JupyterLab from the project root:

```bash
jupyter notebook
```

Then run the notebooks in this order.

### 1. ROI Extraction Experiments

```text
notebooks/01_roi_extraction_experiments.ipynb
```

Use this notebook to inspect and test ROI extraction behavior. This notebook is mainly exploratory and was used to determine the preprocessing approach.

### 2. Preprocess BMPD ROI Images

```text
notebooks/02_preprocess_bmpd_roi.ipynb
```

This notebook processes the raw BMPD images and writes extracted palm ROI images to the processed data directory.

Expected output location:

```text
data/processed/bmpd_roi/
```

### 3. Check Splits and Pair Generation

```text
notebooks/03_check_splits_and_pairs.ipynb
```

This notebook verifies the train, validation, and test splits and checks the generated genuine and impostor pairs.

Final split strategy:

- 70% training identities
- 10% validation identities
- 20% test identities

Pair generation is balanced so that approximately half of the pairs are genuine pairs and half are impostor pairs.

### 4. Train the Siamese Model

```text
notebooks/04_training_siamese.ipynb
```

This notebook trains the Siamese CNN using the processed BMPD ROI images and contrastive loss.

Typical checkpoint output:

```text
checkpoints/siamese_bmpd_roi_best.pt
```

Checkpoint files may be large and may not be included in the submitted repository.

### 5. Evaluate the Model

```text
notebooks/05_evaluation.ipynb
```

This notebook loads a trained checkpoint and evaluates model performance on the test split.

Evaluation outputs include:

- Accuracy
- ROC-AUC
- Equal Error Rate (EER)
- FAR and FRR
- TAR at selected FAR thresholds
- ROC curve figure
- Confusion matrix or pair classification results, if generated

---

## Notes About Scripts

The `scripts/` folder contains command-line helpers used during preprocessing and quality control.

```text
scripts/preprocess_bmpd_roi.py
```

Runs BMPD ROI preprocessing from the command line.

```text
scripts/create_contact_sheets.py
```

Creates contact sheets for reviewing extracted ROI images visually.

```text
scripts/create_roi_quality_template.py
```

Creates a CSV template for marking poor-quality ROI images so they can be excluded from training and evaluation.

The final training and evaluation demo should be run through the notebooks, not through a command-line training script.

---

## Reproducibility Notes

The project uses a fixed random seed where possible to make splits and pair generation reproducible.

Important reproducibility details:

- BMPD is split by identity, not by individual image.
- Left and right palms are treated as separate identities.
- Train, validation, and test identities are kept separate.
- Genuine and impostor pairs are generated within each split.
- Poor-quality ROI images can be excluded using the ROI quality CSV workflow.

---

## Current Limitations

- The final experiments use only the BMPD dataset.
- ROI extraction is based on image processing heuristics and may fail on some difficult hand poses or lighting conditions.
- The model was trained and evaluated on a single dataset, so cross-dataset generalization was not tested.
- Triplet loss was not evaluated in the final implementation.
- PolyU and IITD loaders are placeholders for future expansion and are not part of the final submitted experiment.

---

## Future Work

Possible future improvements include:

- Expand evaluation to PolyU and IITD palmprint datasets.
- Improve ROI extraction using a more robust landmark-based or deep learning-based method.
- Compare different CNN backbones.
- Evaluate triplet loss or other metric learning losses.
- Test cross-dataset generalization.
- Increase image resolution or experiment with different ROI sizes.
- Perform more detailed threshold analysis for biometric verification settings.

---

## License and Data Notice

This repository contains project code, notebooks, and supporting files only. The BMPD dataset is not included and must be downloaded separately from Kaggle. Dataset usage is subject to the dataset owner's terms and licensing requirements.
