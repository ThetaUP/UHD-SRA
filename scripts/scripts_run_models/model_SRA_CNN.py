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

import matplotlib.pyplot as plt

import seaborn as sns

import neptune as neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback

# import seaborn as sns
# sns.set_theme(rc={"figure.dpi":300, 'savefig.dpi':300})
# sns.set_context('notebook')

tmp = os.environ['TMPDIR']

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
def tf_learn(X, y, label_names, params, run):
    neptune_callback = NeptuneCallback(run=run) 

    # Train metrics
    f1_history_train, auc_history_train, loss_history_train = [], [], []

    # Test metrics
    f1_history_val, auc_history_val, loss_history_val = [], [], []

    # Define SKFold                                 
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Run model
    for i, (train_index, val_index) in enumerate(skf.split(X, label_names)):
        print(f'### FOLD {i}')
        model = model_L(params)

        model_history = model.fit(X[train_index], y[train_index],
                            epochs=params['epoch'],
                            batch_size=params['batch_size'],
                            shuffle=True,
                            verbose=2,
                            callbacks=[neptune_callback],
                            validation_data=(X[val_index], y[val_index])
                            )

        f1_history_train.append(model_history.history['f1_score'])
        auc_history_train.append(model_history.history['auc'])
        loss_history_train.append(model_history.history['loss'])
        
        f1_history_val.append(model_history.history['val_f1_score'])
        auc_history_val.append(model_history.history['val_auc'])
        loss_history_val.append(model_history.history['val_loss'])

        tf.keras.backend.clear_session()

    res_avg = {
           'avg_f1_history_train': np.mean(f1_history_train, axis=0), 
           'avg_auc_history_train': np.mean(auc_history_train, axis=0), 
           'avg_loss_history_train': np.mean(loss_history_train, axis=0), 
           'avg_f1_history_val': np.mean(f1_history_val, axis=0),
           'avg_auc_history_val': np.mean(auc_history_val, axis=0), 
           'avg_loss_history_val': np.mean(loss_history_val, axis=0)
           }

    res_std = {
           'std_f1_history_train': np.std(f1_history_train, axis=0), 
           'std_auc_history_train': np.std(auc_history_train, axis=0), 
           'std_loss_history_train': np.std(loss_history_train, axis=0), 
           'std_f1_history_val': np.std(f1_history_val, axis=0),
           'std_auc_history_val': np.std(auc_history_val, axis=0), 
           'std_loss_history_val': np.std(loss_history_val, axis=0)
           }
    
    res = {**res_avg, **res_std}

    print(res)
    
    for epoch in range(params['epoch']): 
        run["train/f1_history"].append(res['avg_f1_history_train'][epoch])
        run["train/auc_history"].append(res['avg_auc_history_train'][epoch])
        run["train/loss_history"].append(res['avg_loss_history_train'][epoch])

        run["val/f1_history"].append(res['avg_f1_history_val'][epoch])
        run["val/auc_history"].append(res['avg_auc_history_val'][epoch])
        run["val/loss_history"].append(res['avg_loss_history_val'][epoch])

    return model_history, res

# %%
def tf_learn_full(X, y, params):

    directory = f'{tmp}/model_fitted_SRA'
    checkpoint_filepath = directory + '/checkpoint_{epoch:02d}.ckpt'

    cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                mode='min', monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, save_freq='epoch')                                              

    model = model_L(params)

    full_model = model.fit(X, y,
                            epochs=params['epoch'],
                            batch_size=params['batch_size'],
                            shuffle=True,
                            verbose=2,
                            callbacks=[cp],
                            )

    return full_model

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

    print(f'MODELS DIRECTORY {tmp}')
    # Init params
    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    
    run = neptune.init_run(
    project="",
    api_token="",
    tags=["SRA", 'TRAIN']
    )  # credentials

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
    ## SRA
    sra_cluster = pd.read_csv('SRA_weg_max_cluster_res.csv')
    sra_cluster_id = sra_cluster['SRA_label']==1

    X_cluster_train = data_train_X[:, sra_cluster_id]
    print(f'shape of TRAIN X with selected features {X_cluster_train.shape}')

    del data_train_X

    X_cluster_test = data_test_X[:, sra_cluster_id]
    print(f'shape of TEST X with selected features {X_cluster_test.shape}')

    del data_test_X

    gc.collect()

    # Define PARAMS space
    model_name = 'SRA'
    params = {
        "epoch": 30,
        "batch_size": 52,
        "yencoded_layer_3": 32,
        "yencoded_layer_2": 16,
        "yencoded_layer_1": 8,
        "n_classes": 5,
        "input_dim_conti" : X_cluster_train.shape[1]
    }

    run["parameters"] = params
    run["files/script"].upload(f'model_SRA.py')

    # Perform KFold
    history, res = tf_learn(X_cluster_train, data_train_y_OHE, data_train_y, params, run)

    # Train full model
    full_model = tf_learn_full(X_cluster_train, data_train_y_OHE, params)

    restored_model = model_L(params)
    latest = tf.train.latest_checkpoint(f'{tmp}/model_fitted_SRA')
    restored_model.load_weights(latest)

    train_y_pred = restored_model.predict(X_cluster_train)

    print(train_y_pred)
    print(lab_train.inverse_transform(train_y_pred))
    print(lab_train.categories_)
    confusion_matrix(data_train_y, lab_train.inverse_transform(train_y_pred), lab_train.categories_)

    # Evaluate TEST
    del X_cluster_train

    test_y_pred = restored_model.predict(X_cluster_test)
    confusion_matrix(data_test_y, lab_train.inverse_transform(test_y_pred), lab_test.categories_)

    test_history = restored_model.evaluate(X_cluster_test, data_test_y_OHE, batch_size=params['batch_size'], return_dict=True)

    df_new = pd.Series({'Model name': model_name,
                    'Mean LOSS Train': res['avg_loss_history_train'][-1], 
                    'SD LOSS Train': res['std_loss_history_train'][-1], 

                    'Mean f1 Train': res['avg_f1_history_train'][-1], 
                    'SD f1 Train': res['std_f1_history_train'][-1], 

                    'Mean AUC Train': res['avg_auc_history_train'][-1], 
                    'SD AUC Train': res['std_auc_history_train'][-1], 

                    'Mean LOSS Val': res['avg_loss_history_val'][-1], 
                    'SD LOSS Val': res['std_loss_history_val'][-1], 

                    'Mean f1 Val': res['avg_f1_history_val'][-1], 
                    'SD f1 Val': res['std_f1_history_val'][-1], 

                    'Mean AUC Val':res['avg_auc_history_val'][-1],
                    'SD AUC Val': res['std_auc_history_val'][-1],

                    'loss Test': test_history['loss'],
                    'f1 Test': test_history['f1_score'],
                    'AUC test': test_history['auc']})
    
    df = df_new.to_frame().T.to_csv('../../run_results/SRA_table_fit_macro.csv')

    print(df)
# %%
