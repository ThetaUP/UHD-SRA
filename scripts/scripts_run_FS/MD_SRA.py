import pandas as pd
from sklearn.utils import resample
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import joblib
from joblib import dump, load

from sklearn.metrics import log_loss

import os
import gc

def gen_window(X, window_size):
    n_col = X.shape[1]
    n_chunks = n_col // window_size

    # print(X)

    perm_id = np.random.permutation(n_col)
    # print(f'Indexes permutated:\n {perm_id}')

    perm_id_inv = np.argsort(perm_id)
    # print(f'Indexes UNpermutated:\n {perm_id_inv}')

    X_permutate = np.array(X[:, perm_id])
    # print(f'X permuated:\n {X_permutate}')

    idx_chanks = np.array_split(np.arange(n_col), n_chunks)
    # print(f'Chanks:\n {idx_chanks}')

    # print(f'Unpermutated matrix:\n {X_permutate[:, perm_id_inv]}')

    return X_permutate, idx_chanks, perm_id, perm_id_inv

def generate_performance_matrix(i, idx, X, y_true, id_inv, output, PM):
    # Define model
    X_window = X[:, i]

    clf = LogisticRegression(penalty=None, tol=1e-2, solver='saga', n_jobs=2, max_iter=200)

    clf.fit(X_window, y_true)
    # print(X_window.shape)

    snps_to_fill = id_inv[i]

    y_proba = clf.predict_proba(X_window)

    cross_e = 1/(np.array(log_loss(y_true, y_proba), dtype=np.float32))

    output[PM, snps_to_fill] = np.max(clf.coef_, axis=0) * cross_e

    # print(output)
    # print(f'Out results from window: {out}')

if __name__ == "__main__":

    params = {'X_train': '',
              'Y_train': '',
              'temp_folder': '',
              'N_MODELS': 5,
              'windows_size': 250}


    N_MODELS = params['N_MODELS']

    X = np.load(params['X_train'])

    Y = pd.read_csv(params['Y_train']).to_numpy()
    lab = LabelEncoder()
    Y = lab.fit_transform(Y.ravel())
    print(Y.shape)

    folder = params['temp_folder']

    output_filename_memmap = os.path.join(folder, 'output_max_hSRA')
    output = np.memmap(output_filename_memmap, dtype=np.float32, mode='w+', shape=(N_MODELS, X.shape[1]))

    for PM in range(N_MODELS): #PM - performance matrix row

        X_permutate, idx_chanks, perm_id, perm_id_inv = gen_window(X, params['windows_size'])

        data_filename_memmap = os.path.join(folder, 'temp_max_hSRA')
        dump(X_permutate, data_filename_memmap)
        data = load(data_filename_memmap, mmap_mode='r')

        del X_permutate

        gc.collect()

        final = joblib.Parallel(n_jobs=20, backend="threading", verbose=1)(joblib.delayed(generate_performance_matrix)(i, idx, data, Y, perm_id_inv, output, PM) for idx, i in enumerate(idx_chanks))
