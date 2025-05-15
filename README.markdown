# ResidualSpatialNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

## Overview

**ResidualSpatialNet** is a deep learning model designed for predicting cell type annotations from spatial transcriptomics data using histological image patches. The model employs a ResNet50 backbone, pretrained on ImageNet, to extract features from image patches, followed by a fully connected regressor to predict expression levels for 35 cell types (C1–C35). Developed for the Elucidata AI Challenge, it processes HDF5 files containing histological images and spot tables with spatial coordinates and cell type expressions. The model is trained with mean squared error (MSE) loss, optimized using the Adam optimizer, and evaluated via Spearman correlation, supporting multi-slide training and test set inference. This repository provides the complete Python codebase, implemented with PyTorch, for researchers and data scientists in computational biology and spatial transcriptomics.

## Features

- **Data Loading**: Extracts histological images and spot tables from HDF5 files.
- **Patch-Based Dataset**: Generates 64x64 patches for training/validation and 128x128 patches for testing, centered on spatial coordinates.
- **Model Architecture**: ResNet50 backbone with a linear regressor outputting 35 cell type expressions.
- **Training Pipeline**: Multi-epoch training with MSE loss, batch processing, and validation using Spearman correlation.
- **Inference**: Produces predictions for test slides in a structured DataFrame format.
- **Device Support**: Runs on CPU or CUDA-enabled GPUs.

## Requirements

- Python 3.8 or higher
- PyTorch 1.9 or higher
- torchvision
- h5py
- numpy
- pandas
- matplotlib
- seaborn
- scipy

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/ResidualSpatialNet.git
   cd ResidualSpatialNet
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install torch torchvision h5py numpy pandas matplotlib seaborn scipy
   ```

   Alternatively, use the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   Check PyTorch and CUDA compatibility (if using a GPU):
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## Dataset

The model is designed for the Elucidata AI Challenge dataset (`elucidata_ai_challenge_data.h5`), structured as follows:
- **Train Images**: Histological images under `images/Train`.
- **Train Spot Tables**: Spatial coordinates (x, y) and cell type expressions (C1–C35) under `spots/Train`.
- **Test Images**: Test slide image under `images/Test`.
- **Test Spot Table**: Spatial coordinates under `spots/Test`.

### Data Preparation

1. **Obtain the Dataset**:
   Download `elucidata_ai_challenge_data.h5` and place it in the `data/` directory or update the file path in the code.

2. **Directory Structure**:
   ```
   ResidualSpatialNet/
   ├── data/
   │   └── elucidata_ai_challenge_data.h5
   ├── models/
   │   └── best_model.pth  # Saved model weights (generated after training)
   ├── main.py
   ├── requirements.txt
   └── README.md
   ```

3. **Preprocessing**:
   The code automatically loads HDF5 data, extracts patches, and handles padding for boundary cases. No manual preprocessing is needed.

## Usage

### Training

Train the model with default hyperparameters:
```bash
python main.py --mode train --data_path data/elucidata_ai_challenge_data.h5 --val_slides S_6 --epochs 15
```

**Command-Line Arguments**:
- `--mode`: Set to `train` (default: `train`).
- `--data_path`: Path to the HDF5 dataset file.
- `--val_slides`: Validation slide IDs (e.g., `S_6`).
- `--patch_size`: Training patch size (default: 64).
- `--batch_size`: Batch size (default: 32).
- `--epochs`: Training epochs (default: 15).
- `--lr`: Learning rate (default: 1e-4).
- `--device`: Device (`cuda` or `cpu`, default: `cuda`).

The model saves the best weights (based on validation Spearman correlation) to `models/best_model.pth`.

### Inference

Generate predictions for the test slide:
```bash
python main.py --mode predict --data_path data/elucidata_ai_challenge_data.h5 --model_path models/best_model.pth
```

**Command-Line Arguments**:
- `--mode`: Set to `predict`.
- `--data_path`: Path to the HDF5 dataset file.
- `--model_path`: Path to trained model weights.
- `--patch_size`: Testing patch size (default: 128).
- `--batch_size`: Batch size (default: 64).
- `--device`: Device (`cuda` or `cpu`, default: `cuda`).

**Output**:
A CSV file (`test_predictions.csv`) with predicted cell type expressions (C1–C35) and an `ID` column for spot identifiers:
```csv
ID,C1,C2,...,C35
0,0.123,0.456,...,0.789
1,0.234,0.567,...,0.890
...
```

## Code Structure

- **main.py**: Entry point for training and inference.
- **dataset.py**: Defines `SpotDataset` for training/validation and `SpatialSpotDataset` for testing.
- **model.py**: Implements `CellTypeModel` with ResNet50 and a linear regressor.
- **train.py**: Training and validation logic, including MSE loss and Spearman correlation.
- **predict.py**: Inference pipeline for test predictions.
- **utils.py**: Helper functions for HDF5 data loading and processing.

## Model Architecture

**ResidualSpatialNet** uses a ResNet50 backbone pretrained on ImageNet to extract features from image patches. The backbone’s fully connected layer is replaced with a regressor comprising:
- A linear layer (2048 → 512) with ReLU activation.
- Dropout (0.3) for regularization.
- A linear layer (512 → 35) to predict 35 cell type expressions.

The model is trained with MSE loss and evaluated using Spearman correlation, optimized for spatial transcriptomics tasks.

## Performance

The model achieves a Spearman correlation of approximately 0.4–0.6 on validation data, depending on the dataset and hyperparameters. Training on an NVIDIA V100 GPU takes 1–2 hours for 15 epochs with a batch size of 32.

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please adhere to PEP 8 style guidelines and include tests where applicable.

## Issues

Report bugs or ask questions on the [GitHub Issues page](https://github.com/<your-username>/ResidualSpatialNet/issues). Include:
- Python and library versions.
- Dataset details.
- Error messages or logs.
- Steps to reproduce.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Elucidata AI Challenge for the dataset.
- PyTorch and torchvision for deep learning frameworks.
- The spatial transcriptomics community for inspiration.

## Contact

For inquiries, contact [Your Name] at [your.email@example.com] or open a GitHub issue.