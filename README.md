# Palmprint Recognition Using Siamese Neural Networks

Deep Learning (CS 4263-901)
Cameron Ortiz, Myar Nguyen, Edison La

---

## Project Overview

This project implements a **Siamese Neural Network (SNN)** for palmprint verification.

The goal is to develop a reliable contactless biometric authentication system suitable for payment environments. Instead of treating palmprint recognition as a classification task, we model it as a **similarity-based verification problem**.

Given two palmprint images, the system predicts whether they belong to the same individual.

---

## Motivation

In prior work, we evaluated PCA, CNN, hand-crafted, and hybrid models for palmprint recognition and achieved a best CNN accuracy of **70.10%**.

Key limitations included
- No palm ROI extraction
- Low image resolution
- Left/right palm ambiguity
- Classification-based setup instead of verification

This project addresses those limitations using ROI extraction and Siamese metric learning.

---

## Datasets

This project supports multiple palmprint datasets:

- PolyU Contactless Palmprint Database
- IITD Palmprint Database
- BMPD (Kaggle) – used for initial experimentation

Datasets are **not included** in this repository due to licensing restrictions.

Place raw datasets in:

    data/raw/

---

## Problem Formulation

This project focuses on **verification**, not classification.

Given two palmprint images:

- Output: Same person / Different person
- Training Objective: Learn an embedding space where:
  - Same-person pairs are close
  - Different-person pairs are far apart

---

## Architecture

- Shared backbone encoder (CNN-based)
- Siamese twin structure
- Contrastive loss (optional: triplet loss)
- Evaluation using:
  - ROC curve
  - Equal Error Rate (EER)
  - Verification accuracy

---

## Project Structure

    palmprint-verification-siamese/
    │
    ├── data/
    │   ├── raw/                 # not committed
    │   ├── interim/
    │   └── processed/
    │
    ├── src/palmprint/
    │   ├── datasets/            # BMPD, IITD, PolyU loaders
    │   ├── preprocessing/       # ROI extraction, normalization
    │   ├── sampling/            # pair/triplet generation
    │   ├── models/              # backbone + siamese model
    │   ├── training/            # train + eval logic
    │   └── utils/               # config, logging, seeding
    │
    ├── scripts/                 # dataset preparation
    ├── experiments/configs/     # experiment configuration files
    ├── splits/                  # committed train/test splits
    ├── checkpoints/             # not committed
    ├── logs/                    # not committed
    ├── notebooks/
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

## Setup

Create environment:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

---

## Training

Example:

    python -m src.palmprint.training.train \
        --dataset bmpd \
        --config experiments/configs/bmpd_siamese.yaml

---

## Evaluation

    python -m src.palmprint.training.eval \
        --checkpoint checkpoints/model.pth

Metrics:
- ROC curve
- AUC
- Equal Error Rate (EER)

---

## Research Goals

- Compare performance across PolyU, IITD, BMPD
- Evaluate cross-dataset generalization
- Optimize embedding dimensionality
- Experiment with different backbone architectures
- Analyze verification robustness

---

## License

This repository contains code only.
Datasets are subject to their respective licenses.
