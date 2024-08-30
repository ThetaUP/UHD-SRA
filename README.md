# Approaches to Dimensionality Reduction for Ultra-High Dimensional Models

This repository contains the implementation of a research project focused on dimensionality reduction techniques for ultra-high dimensional models. The project explores feature selection using 1D-SRA and MD-SRA approaches, followed by K-means clustering to identify relevant and irrelevant feature sets and calculate L2 distances to cluster centers. Finally, a CNN classification model is applied to analyze the reduced feature sets.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Run 1D-SRA or MD-SRA Feature Selection](#1-run-1d-sra-or-md-sra-feature-selection)
  - [2. K-means Clustering](#3-k-means-clustering)
  - [3. Run the CNN Classification Model](#4-run-the-cnn-classification-model)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project investigates various approaches to dimensionality reduction in ultra-high dimensional datasets. Feature selection is performed using 1D-SRA (one-dimensional supervised rank aggregation) or MD-SRA (multidimensional-dimensional rank aggregation). After reduced models' estimation, K-means clustering is applied to identify relevant and irrelevant features, with L2 distance calculations providing additional insights into feature grouping. Finally, a Convolutional Neural Network (CNN) classification model is used to analyze the reduced feature sets.

For more detailed information, please refer to the [original manuscript](#) or [preprint](https://www.biorxiv.org/content/10.1101/2024.08.20.608783v1) that provides a comprehensive explanation of the methodologies.

## Features

- Feature selection using 1D-SRA and MD-SRA.
- K-means clustering for distinguishing relevant and irrelevant feature sets and calculating L2 distance to clusters.
- CNN classification model for high-dimensional data analysis.
  
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kotlarzkrzysztof/UHD-SRA.git
    cd UHD-SRA
    ```

2. Install the required Python packages:
    ```bash
    pip install pandas scikit-learn numpy joblib seaborn tensorflow matplotlib neptune
    ```
    
## Usage

### 1. Run 1D-SRA or MD-SRA Feature Selection

To perform feature selection, use the scripts `scripts/scripts_run_FS/1D-SRA.py` or `scripts/scripts_run_FS/MD-SRA.py`. Set the input variables by specifying the paths to the datasets.

- **1D-SRA:** Set the window size for feature selection and path to input files in `params`.
  
    ```python
    params = {'X_train': '',
              'Y_train': '',
              'temp_folder': '',
              'windows_size': 250}
    ```
    **Note:** Please note that aggregation step for 1D-SRA based on Linear Mixed Model (LMM) us using the external software.

- **MD-SRA:** Set both the window size with the number of repetitions to obtain multidimensional feature selection and path to input files in `params`.
  
    ```python
    params = {'X_train': '',
              'Y_train': '',
              'temp_folder': '',
              'N_MODELS': 5,
              'windows_size': 250}
    ```

### 3. K-means Clustering

After feature selection with 1D-SRA or MD-SRA, apply K-means clustering to divide the feature set into relevant and irrelevant subsets. This step should be done **before** running the CNN classification model.

- Run the K-means clustering script:
    ```python
    from sklearn.cluster import KMeans
    import numpy as np
    import pandas as pd

    # Load the feature set
    features = pd.read_csv("path/to/{1D-SRA,MD-SRA}_results.csv")

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit_predict(features)

    # Save the cluster labels
    pd.Series(kmeans.labels_, name='Labels').to_csv('path/to/selected_features.csv)

    # Calculate L2 distance to the nearest cluster
    distances = np.linalg.norm(kmeans.cluster_centers_, axis=1)
    np.savetxt("l2_distances.csv", distances, delimiter=",")
    ```

- **L2 Distance Calculation:** After running K-means, calculate the L2 (Euclidean) distance from each data point to the nearest cluster center.


### 4. Run the CNN Classification Model

After feature selection, use the CNN model for classification. Update the paths to the training and testing datasets in the script `scripts/scripts_run_models/model_{1D_SRA,MD_SRA}_CNN.py`.

- Script modification:
    ```python
    data_train_X = "path/to/train_data_X.csv"
    data_train_y = "path/to/test_data_y.csv"

    data_test_X = "path/to/test_data_X.csv"
    data_test_y = "path/to/test_data_y.csv"


    df = "path/to/result_table.csv"
    ```

- Optionally, add your Neptune.ai token to monitor the training process:
    ```python
    import neptune

    run = neptune.init_run(
        project="your_project_name",
        api_token="your_neptune_api_token",
    )
    ```

## Contributing

Contributions are welcome! Please feel free to open an issue to discuss potential changes or improvements with the author of the article.

**Note:** To obtain the original dataset used in this research, please contact the corresponding author of the article.
