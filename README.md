# Early Degradation Detection in NASA C-MAPSS Turbofan Engines

This project explores **predictive maintenance** using the NASA **C-MAPSS** turbofan engine dataset.  
The main goal is to detect **early engine degradation** from multivariate sensor time series using an **anomaly detection** approach.

The main project artifact is the notebook:

- `nasa.ipynb`

## Project Overview

Instead of training a model to directly predict failure labels, this project focuses on:

- learning normal / early-life engine behavior
- computing anomaly scores as the engine degrades
- checking whether anomaly scores rise as the engine approaches failure

The current notebook workflow focuses mainly on **FD001**, which is the simplest subset of the C-MAPSS dataset:

- 1 operating condition
- 1 fault mode

This makes FD001 a good starting point before moving to harder subsets such as FD002, FD003, or FD004.

## What Is Implemented

The notebook currently includes:

1. **Loading and validating FD001**
2. **Exploratory Data Analysis (EDA)**
3. **Preprocessing and sensor selection**
4. **Row-level RUL construction for evaluation**
5. **Sliding window generation**
6. **Healthy training-window selection**
7. **PCA reconstruction-error baseline**
8. **LSTM autoencoder section** for deep learning comparison

## Current Status

The classical anomaly-detection pipeline is already working through the PCA baseline.

Validated progress so far:

- raw data loading works
- EDA is completed
- preprocessing is completed
- RUL construction is completed
- sliding windows are built
- healthy windows are selected
- PCA baseline is implemented and evaluated

Current PCA baseline results from the notebook:

- **Train ROC-AUC:** `0.8628`
- **Test ROC-AUC:** `0.8338`

Note: the LSTM autoencoder section requires **TensorFlow** in the active notebook environment.

## Dataset

The dataset files are stored in:

- `nasa/`

Included files:

- `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
- `train_FD003.txt`, `test_FD003.txt`, `RUL_FD003.txt`
- `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`
- `readme.txt`
- `Damage Propagation Modeling.pdf`

For the main notebook workflow, **FD001** is the primary subset used.

## Dataset Format

Each row in the C-MAPSS files contains:

- `unit`: engine ID
- `cycle`: time step / usage cycle
- `op1`, `op2`, `op3`: operational settings
- `s1` to `s21`: sensor measurements

In simple terms:

- each **unit** is one engine
- each **cycle** is one sequential operating step in that engine's life
- the training set runs each engine **to failure**
- the test set stops **before failure**
- the `RUL` files give the remaining useful life after the last observed test cycle

## Project Structure

```text
.
|-- nasa/
|   |-- train_FD001.txt
|   |-- test_FD001.txt
|   |-- RUL_FD001.txt
|   |-- ...
|   |-- readme.txt
|   `-- Damage Propagation Modeling.pdf
|-- nasa.ipynb
|-- src/
|-- scripts/
`-- tests/
```

Even though some helper files are present, the notebook remains the main place where the project is developed and explained.

## Methods Used

### 1. PCA Reconstruction Baseline

A simple anomaly-detection baseline is built using **PCA reconstruction error**:

- train PCA on healthy windows
- reconstruct each window
- use reconstruction error as the anomaly score

### 2. LSTM Autoencoder

The notebook also includes an **LSTM autoencoder** section intended to:

- learn healthy temporal patterns
- reconstruct windows of sensor data
- use reconstruction error as an anomaly score

## How To Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Open the notebook

Open:

- `nasa.ipynb`

You can run it in:

- Jupyter Notebook
- JupyterLab
- VS Code notebook interface

### 3. Install dependencies

At minimum, make sure the active Python environment has:

```bash
pip install pandas matplotlib scikit-learn jupyter
```

For the LSTM section, also install:

```bash
pip install tensorflow
```

## Suggested Notebook Flow

If you are reading the project for the first time, go through the notebook in this order:

1. data loading and validation
2. EDA
3. preprocessing
4. RUL construction
5. window generation
6. PCA baseline
7. LSTM section
8. evaluation and visualizations

## Why This Project Is Interesting

This project is interesting because it combines:

- predictive maintenance
- multivariate time-series analysis
- anomaly detection
- early degradation detection
- classical and deep-learning approaches

It is also a good applied ML case study because the data has:

- realistic sensor behavior
- progressive degradation
- multiple engines with different lifetimes
- a clear evaluation target through RUL

## Possible Next Improvements

Some natural next steps are:

- finish validating the LSTM autoencoder end to end
- compare PCA and LSTM with detection lead time
- try FD003 as a second benchmark
- add more interpretation of sensor-level anomaly behavior
