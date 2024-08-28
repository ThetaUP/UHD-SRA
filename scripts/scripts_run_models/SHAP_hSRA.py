# %%
# %%
import os
import random
import gc

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import metrics

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import shap

import matplotlib.pyplot as plt

import seaborn as sns

import neptune as neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

# %%
def model_L(params):

    model_multiclass = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (params['input_dim_conti'], 1)),
    tf.keras.layers.Conv1D(filters=10, kernel_size=150, strides=50, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool1D(5),
    tf.keras.layers.Conv1D(filters=10, kernel_size=150, strides=50, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool1D(5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(params['yencoded_layer_3'], activation='relu'),
    tf.keras.layers.Dense(params['yencoded_layer_2'], activation='relu'),
    tf.keras.layers.Dense(params['yencoded_layer_1'], activation='relu'),
    tf.keras.layers.Dense(params['n_classes'], activation='softmax')
    ])

    print(model_multiclass.summary())

    f1 = tf.keras.metrics.F1Score(average='macro')
    opt_ad = tf.keras.optimizers.Adam(0.00005)
    #alpha=[1.43842365, 1.55319149, 0.39300135, 2.16296296, 1.52879581]
    loss_f = tf.keras.losses.CategoricalFocalCrossentropy(alpha=[1.43842365, 1.55319149, 0.39300135, 2.16296296, 1.52879581])

    model_multiclass.compile(optimizer=opt_ad, 
                             loss=loss_f, 
                             metrics=[f1, 
                                      'AUC'])

    return model_multiclass

# %%
def confusion_matrix(y_true, y_pred, labels):
    titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
    ]
    
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            normalize=normalize,
            )

        print(title)
        print(disp.confusion_matrix)

# %%
if __name__ == "__main__":

    # Init params
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    
    # Load data
    data_train_X = np.load('X_train_matrix_unpruned.npy')
    data_train_y = pd.read_csv('Y_train_matrix_unpruned.csv').to_numpy()

    data_test_X = np.load('X_test_matrix_unpruned.npy')
    data_test_y = pd.read_csv('Y_test_matrix_unpruned.csv').to_numpy()

    # Prepare y train
    lab_train = OneHotEncoder(sparse_output=False)
    data_train_y_OHE = lab_train.fit_transform(data_train_y)

    # Prepare y test
    lab_test = OneHotEncoder(sparse_output=False)
    data_test_y_OHE = lab_test.fit_transform(data_test_y)

    # Feature indicator
    ## sraSNP
    hsra_cluster = pd.read_csv('hSRA_weg_max_cluster_res.csv')
    hsra_cluster_id = hsra_cluster['hSRA_label']==1

    # Feature selection

    ## sraSNP and transform to SHAP subset
    X_cluster_train = data_train_X[:, hsra_cluster_id]
    X_cluster_train = X_cluster_train.reshape(X_cluster_train.shape[0], X_cluster_train.shape[1], 1)
    print(f'shape of TRAIN X with selected features {X_cluster_train.shape}')

    del data_train_X

    X_cluster_test = data_test_X[:, hsra_cluster_id]

    X_cluster_test = X_cluster_test.reshape(X_cluster_test.shape[0], X_cluster_test.shape[1], 1)

    print(f'shape of TEST X with selected features {X_cluster_test.shape}')

    del data_test_X

    gc.collect()

    # Define PARAMS space
    model_name = 'hsra'
    params = {
        "epoch": 30,
        "batch_size": 52,
        "yencoded_layer_3": 32,
        "yencoded_layer_2": 16,
        "yencoded_layer_1": 8,
        "n_classes": 5,
        "input_dim_conti" : X_cluster_train.shape[1]
    }

    restored_model = model_L(params)
    latest = tf.train.latest_checkpoint(f'model_res/model_fitted_hSRA')
    
    restored_model.load_weights(latest)

    test_y_pred = restored_model.predict(X_cluster_test)
    confusion_matrix(data_test_y, lab_train.inverse_transform(test_y_pred), lab_test.categories_)

    # SHAP Values
    print('SHAP VALUES GENERATION...\n')
    
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough 
    explainer = shap.GradientExplainer(restored_model, X_cluster_train)

    shap_values = explainer.shap_values(X_cluster_test, nsamples=50)
    re_shap = np.array(shap_values).reshape(X_cluster_test.shape[0], X_cluster_test.shape[1], 5)
    re_shap_max_avg_abs = np.abs(re_shap.max(axis=2)).mean(axis=0)

    pd.DataFrame({"ID": hsra_cluster[hsra_cluster['hSRA_label']==1]["CHROM:POS"],"SHAP": re_shap_max_avg_abs}).to_csv('../../run_results/SHAP_res/SHAP_hSRA.csv')
