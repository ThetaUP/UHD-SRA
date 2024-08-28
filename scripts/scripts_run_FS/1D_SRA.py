import pandas as pd
from sklearn.utils import resample
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from joblib import dump, load
import joblib

from sklearn.metrics import log_loss

import seaborn as sns

import os
import gc

def gen_window(X, window_size):
    n_col = X.shape[1]
    n_chunks = n_col // window_size

    perm_id = np.random.permutation(n_col)
    # print(f'Indexes permutated: {perm_id}')

    perm_id_inv = np.argsort(perm_id)
    # print(f'Indexes UNpermutated: {perm_id_inv}')

    X_permutate = np.array(X[:, perm_id])
    # print(f'X permuated: {X_permutate}')

    idx_chanks = np.array_split(np.arange(n_col), n_chunks)
    # print(f'Chanks: {idx_chanks}')

    return X_permutate, idx_chanks, perm_id, perm_id_inv

def generate_performance_matrix(i, idx, X, y_true, id_inv, output):
    # Define model
    X_window = X[:, i]

    clf = LogisticRegression(penalty=None, tol=1e-2, solver='saga', n_jobs=2, max_iter=200)

    clf.fit(X_window, y_true)
    # print(X_window.shape)

    y_proba = clf.predict_proba(X_window)
    cross_e = 1/(np.array(log_loss(y_true, y_proba), dtype=np.float32))

    out = np.zeros(X.shape[1], dtype=np.float32)
    out[i] = np.max(clf.coef_, axis=0)
    # print(f'Out results from window: {out}')

    out = out[id_inv]

    # print(f'Out results oryginal order: {out}')

    out_full = np.append(out, cross_e)

    # MEMMAP approach
    output[idx, :] = out_full

if __name__ == "__main__":

    params = {'X_train': '',
              'Y_train': '',
              'temp_folder': '',
              'N_MODELS': 5}

    X = np.load(params['X_train'])
    Y = pd.read_csv(params['Y_train']).to_numpy()

    lab = LabelEncoder()
    Y = lab.fit_transform(Y.ravel())
    print(Y.shape)

    X_permutate, idx_chanks, perm_id, perm_id_inv = gen_window(X, 250)

    del X

    folder = params['temp_folder']

    data_filename_memmap = os.path.join(folder, 'data_max_1D')
    dump(X_permutate, data_filename_memmap)
    data = load(data_filename_memmap, mmap_mode='r')

    print(len(idx_chanks))
    print(data.shape[1] + 1)

    del X_permutate

    output_filename_memmap = os.path.join(folder, 'output_max_1D')
    output = np.memmap(output_filename_memmap, dtype=np.float32, mode='w+', shape=(len(idx_chanks), data.shape[1] + 1))

    gc.collect()

    final = joblib.Parallel(n_jobs=30, backend="threading", verbose=1)(joblib.delayed(generate_performance_matrix)(i, idx, data, Y, perm_id_inv, output) for idx, i in enumerate(idx_chanks))