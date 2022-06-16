import json
import glob
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

from sklearn.model_selection import KFold

def normalize_cols(mat, num_train, min_col, max_col=None):
    col_std = []
    max_col = mat.shape[1] if max_col is None else max_col
    for col in range(min_col, max_col):
        col_train_mean = mat[:num_train, col].mean()
        col_train_sd = np.sqrt(np.var(mat[:num_train, col]))
        mat[:, col] -= col_train_mean
        if col_train_sd > 0:
            mat[:, col] /= col_train_sd
        col_std.append([col_train_mean, col_train_sd])
    return np.array(col_std)


def normalize_cols_with_params(mat, col_std, max_col):
    for col in range(max_col):
        col_train_mean = col_std[col, 0]
        col_train_sd = col_std[col, 1]
        mat[:, col] -= col_train_mean
        if col_train_sd > 0:
            mat[:, col] /= col_train_sd


"""
Classification datas
"""
N_FOLDS = 4
MIN_COUNT = 1

files = glob.glob("cumida/*.csv")
tot_files = len(files)
dataset_meta = {}
for file_idx, file_name in enumerate(files):
    np.random.seed(0)
    df = pd.read_csv(file_name)

    if len(df) < 40:
        continue

    dataset = file_name.split("/")[-1].replace(".csv", "")
    print("dataset %d/%d, %s, num samples: %d" % (file_idx, tot_files,  dataset, len(df)))
    mat = df.to_numpy()
    np.random.shuffle(mat)
    num_train = mat.shape[0] * 3//4
    out_train_file = "_output/%s_train.npz" % dataset
    out_test_file = "_output/%s_test.npz" % dataset

    if np.unique(mat[:num_train, 1]).size != np.unique(mat[:, 1]).size:
        print(np.unique(mat[:num_train, 1]).size, np.unique(mat[:, 1]).size)
        print("PROBLEM num train classes not equal total classes")
        continue

    uniq_classes = np.unique(mat[:, 1])
    dataset_meta[dataset] = {"class": uniq_classes.size}
    uniq_class_dict = {cls_str: i for i, cls_str in enumerate(uniq_classes)}
    for i in range(mat.shape[0]):
        mat[i, 1] = uniq_class_dict[mat[i, 1]]

    print("num train classes", np.unique(mat[:num_train, 1]).size)
    print("train class bin", np.bincount(mat[:num_train, 1].astype(int)))
    print("train class bin", np.bincount(mat[:, 1].astype(int)))
    print("num total classes", np.unique(mat[:, 1]).size)

    col_std = normalize_cols(mat, mat.shape[0], min_col = 2, max_col=mat.shape[1])
    print(col_std)

    np.savez_compressed(
        out_train_file,
        x=mat[:num_train, 2:].astype(float),
        y=mat[:num_train, 1:2].astype(int),
        true_y=mat[:num_train, 1:2].astype(int),
    )
    np.savez_compressed(
        out_test_file,
        x=mat[num_train:, 2:].astype(float),
        y=mat[num_train:, 1:2].astype(int),
        true_y=mat[num_train:, 1:2].astype(int),
    )

with open("cumida/datasets_new.json", "w") as f:
    json.dump(dataset_meta, f)
