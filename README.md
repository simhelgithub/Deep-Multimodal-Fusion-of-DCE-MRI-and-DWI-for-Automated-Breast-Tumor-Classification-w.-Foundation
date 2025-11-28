# Deep Multimodal Fusion of DCE-MRI and DWI for Automated Breast Tumor Classification

**Author:** Simon Hellberg

## Overview

This project implements deep multimodal fusion of Dynamic Contrast-Enhanced MRI (DCE-MRI) and Diffusion-Weighted Imaging (DWI) for automated breast tumor classification, leveraging foundation-model pretraining.

## Quick Start

1. **Run the complete experiment:**
   ```bash
   python run.py
   ```
   Set the base path in `run.py` before running.

2. **Generate parameter configuration:**
   ```bash
   python parameters_generate.py
   ```
   This creates a blank parameter dictionary. Set additional paths if needed within the file.

## Project Structure

### Main Control Files

- **`run.py`** - Main entry point that executes the entire experiment pipeline
- **`parameters_generate.py`** - Configures all experimental parameters and options

### Models

- **`model_module.py`** - Contains model architectures for DWI, DCE-MRI, and fusion learning

### Training, Validation & Testing

- **`train.py`** - Handles training of individual models
- **`train_fusion.py`** - Manages fusion model training and optional fine-tuning
- **`model_test.py`** - Evaluates final model performance on test data

### Workflow Simplifiers

- **`prepare_single_model.py`** - Prepares and saves dataloaders for individual models (used in fusion)
- **`prepare_fusion_model.py`** - Loads single models and creates fusion model for training
- **`run_training.py`** - Initializes optimizers, loss criteria, and orchestrates training pipeline

### Helper Modules

- **`preprocess_helpers.py`** - Data normalization and formatting utilities
- **`selector_helpers.py`** - Selection utilities for loss functions and other components
- **`loss.py`** - Custom loss function implementations
- **`foundation_model.py`** - Foundation model initialization utilities

## Dataset

The processed datasets are openly available at:  
[Breast Cancer Subtypes Dataset on Kaggle](https://www.kaggle.com/datasets/starzh10/breastcaner-subtypes)

## Attribution

This work is based on:

**Original Repository:** [DWI_DCE_CDFR-DNN](https://github.com/ZH1O/DWI_DCE_CDFR-DNN_)

**Original Paper:**  
*A Deep Learning Model for Predicting Breast Cancer Molecular Subtypes on Overall b-Value Diffusion-weighted MRI*  
by XinXiang Zhou, Lan Zhang, Xiang-Quan Cui, Hui Li, Zhi-Chang Ba, Hong-Xia Zhang, Yue-Min Zhu, Zi-Xiang Kuai

## License

Please refer to the original repository and paper for licensing information.