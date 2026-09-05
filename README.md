# PyWaveCNN

## Contents
- [Overview](#overview)
- [Technical Features](#technical-features)
- [Machine Learning Models](#machine-learning-models)
- [Project Layout](#project-layout)
- [Usage](#usage)
- [Output](#output)
- [References](#references)

## Overview

Gravitational Wave Classification CNN is a convolutional neural network (CNN) project, leveraging TensorFlow to analyse and categorise various types of gravitational wave data represented in contour plots (the [Gravity Spy](https://doi.org/10.5281/zenodo.1476551) dataset).

## Technical Features

- **TensorFlow and Keras Integration:** Built using TensorFlow's high-level Keras API.
- **Convolutional Neural Network:** Uses `Conv2D` and `MaxPooling2D` layers to extract features from contour plots of gravitational waves.
- **Model Optimisation:** Uses the `RMSProp` optimiser.
- **Parallel Preprocessing:** Uses `multiprocessing` so each class's images are cropped concurrently rather than one at a time.

## Machine Learning Models

A CNN with several convolutional and max-pooling layers followed by dense layers and a softmax output for multi-class classification. The number of output classes is inferred from the training data directory rather than hardcoded.

## Project Layout

```
.
├── pywavecnn/
│   ├── config.py     # paths, hyperparameters
│   ├── data.py       # download / extract / crop / split the dataset
│   ├── models.py     # CNN architecture
│   ├── plotting.py   # sample grid, accuracy/loss, confusion matrix
│   └── train.py       # CLI entry point that wires everything together
├── tests/            # smoke tests for the model builder
├── data/             # dataset lives here at runtime (gitignored)
├── figures/          # generated plots land here (gitignored)
├── requirements.txt      # cross-platform core dependencies
└── requirements-gpu.txt  # optional pinned CUDA libs for Linux+NVIDIA
```

## Usage

1. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Optional, for pinned CUDA libs on Linux+NVIDIA:
   # pip install -r requirements-gpu.txt
   ```
3. **Run training** (from the repository root):
   ```bash
   python -m pywavecnn.train
   ```
   The first run downloads and prepares the dataset automatically; later runs
   reuse the prepared split unless you pass `--force-redownload`.

   Useful flags:
   ```bash
   python -m pywavecnn.train --epochs 30 --batch-size 64 --img-size 300 300 --seed 42
   ```

4. **Run tests**:
   ```bash
   pytest
   ```

## Output

The model processes gravitational wave data, offering insightful visualisations and performance metrics saved to `figures/`:

- **Data Visualisation:** Sample images per class.

  ![gw_types.png](figures/gw_types.png)

- **Model Performance Metrics:** Accuracy and loss over training epochs.

  ![model_accuracy_loss_plot.png](figures/model_accuracy_loss_plot.png)

- **Confusion Matrix:** Predictive performance on unseen data.

  ![confusion_matrix.png](figures/confusion_matrix_plot.png)

## References

Coughlin, S. (2018). Updated Gravity Spy Data Set (v1.1.0) [Data set]. Zenodo. [DOI:10.5281/zenodo.1476551](https://doi.org/10.5281/zenodo.1476551)

## [Back to Top](#pywavecnn)
