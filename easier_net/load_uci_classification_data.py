import pandas as pd
import numpy as np
from sklearn.datasets import load_boston


def normalize_cols(mat, num_train, max_col=None):
    col_std = []
    max_col = mat.shape[1] if max_col is None else max_col
    for col in range(max_col):
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
np.random.seed(0)

"""
Arrythmia
"""
out_train_file = "_output/arrhythmia_train.npz"
out_test_file = "_output/arrhythmia_test.npz"
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data",
    names=np.arange(280),
).drop(axis=1, index=13)
print(df)
mat = df.to_numpy()
print(mat)
print("bad ?", np.mean(mat == "?"))
mat[mat == "?"] = 0
for col in range(mat.shape[1] - 1):
    mat[:, col] = mat[:, col].astype(float)
mat[:, -1] = mat[:, -1].astype(int) - 1

uniq_labels = np.unique(mat[:, -1])
label_dict = {uniq_labels[i]: i for i in range(uniq_labels.size)}
for i in range(mat.shape[0]):
    mat[i, -1] = label_dict[mat[i, -1]]

np.random.shuffle(mat)
num_train = mat.shape[0] // 4 * 3
assert np.unique(mat[:num_train, -1:]).size == uniq_labels.size
print(f"UNIQ LABELS {uniq_labels.size}")

normalize_cols(mat, num_train, max_col=mat.shape[1] - 1)

np.savez_compressed(
    out_train_file,
    x=mat[:num_train, :-1].astype(float),
    y=mat[:num_train, -1:].astype(int),
    true_y=mat[:num_train, -1:].astype(int),
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, :-1].astype(float),
    y=mat[num_train:, -1:].astype(int),
    true_y=mat[num_train:, -1:].astype(int),
)

"""
gene cancer
"""
out_train_file = "_output/gene_cancer_class_train.npz"
out_test_file = "_output/gene_cancer_class_test.npz"

df_y = pd.read_csv("../data/TCGA-PANCAN-HiSeq-801x20531/labels.csv", nrows=801,)
y = df_y.iloc[:, 1].to_numpy()
classes = np.unique([str(a) for a in df_y.iloc[:, 1]])
class_dict = {klass: i for i, klass in enumerate(classes)}
y = [class_dict[a] for a in y]
print(y)

df_x = pd.read_csv("../data/TCGA-PANCAN-HiSeq-801x20531/data.csv", nrows=801)
x = df_x.iloc[:801, 1:].to_numpy()
x = np.nan_to_num(x)

mat = np.hstack([x, np.array(y).reshape((-1, 1))])

np.random.shuffle(mat)
num_train = mat.shape[0] // 3 * 2

normalize_cols(mat, num_train, max_col=mat.shape[1] - 1)
print(np.unique(mat[:num_train, -1:]).astype(int).size, classes.size)
assert np.unique(mat[:num_train, -1:]).astype(int).size == classes.size

np.savez_compressed(
    out_train_file,
    x=mat[:num_train, :-1].astype(float),
    y=mat[:num_train, -1:].astype(int),
    true_y=mat[:num_train, -1:].astype(int),
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, :-1].astype(float),
    y=mat[num_train:, -1:].astype(int),
    true_y=mat[num_train:, -1:].astype(int),
)

"""
Soybean (large)
"""
out_train_file = "_output/soybean_train.npz"
out_test_file = "_output/soybean_test.npz"
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data"
)
print(df)
print(df.shape)
mat = df.to_numpy()
classes = np.unique(mat[:, 0])
class_dict = {klass: i for i, klass in enumerate(classes)}
x = mat[:, 1:]
x[x == "?"] = -1
y = [class_dict[a] for a in mat[:, 0]]
mat = np.hstack([x, np.array(y).reshape((-1, 1))]).astype(float)

np.random.shuffle(mat)
num_train = mat.shape[0] // 4 * 3
normalize_cols(mat, num_train, max_col=mat.shape[1] - 1)
print(np.unique(mat[:num_train, -1:]).astype(int).size, classes.size)
assert np.unique(mat[:num_train, -1:]).astype(int).size == classes.size
print(f"UNIQ LABELS {classes.size}")

np.savez_compressed(
    out_train_file,
    x=mat[:num_train, :-1].astype(float),
    y=mat[:num_train, -1:].astype(int),
    true_y=mat[:num_train, -1:].astype(int),
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, :-1].astype(float),
    y=mat[num_train:, -1:].astype(int),
    true_y=mat[num_train:, -1:].astype(int),
)

"""
Semieon
"""
out_train_file = "_output/semieon_train.npz"
out_test_file = "_output/semieon_test.npz"
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data",
    delimiter=" ",
)
mat = df.to_numpy()
x = mat[:, :256]
y = np.where(mat[:, 256:266])[1]
mat = np.hstack([x, y.reshape((-1, 1))])
classes = np.unique(y)

np.random.shuffle(mat)
num_train = mat.shape[0] // 3 * 2
print(np.unique(mat[:num_train, -1:]).astype(int).size, classes.size)
normalize_cols(mat, num_train, max_col=mat.shape[1] - 1)
assert np.unique(mat[:num_train, -1:]).astype(int).size == classes.size

np.savez_compressed(
    out_train_file,
    x=mat[:num_train, :-1].astype(float),
    y=mat[:num_train, -1:].astype(int),
    true_y=mat[:num_train, -1:].astype(int),
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, :-1].astype(float),
    y=mat[num_train:, -1:].astype(int),
    true_y=mat[num_train:, -1:].astype(int),
)

"""
Hill-valley
"""
out_train_file = "_output/hill_valley_train.npz"
out_test_file = "_output/hill_valley_test.npz"
df_train = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_with_noise_Testing.data"
)
df_test = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/hill-valley/Hill_Valley_with_noise_Training.data"
)
mat_train = df_train.to_numpy()
num_train = mat_train.shape[0]
mat_test = df_test.to_numpy()

print("NUM TRA", num_train)
col_std = normalize_cols(mat_train, num_train, max_col=mat_train.shape[1] - 1)
normalize_cols_with_params(mat_test, col_std, max_col=mat_train.shape[1] - 1)

np.savez_compressed(
    out_train_file,
    x=mat_train[:, :-1].astype(float),
    y=mat_train[:, -1:].astype(int),
    true_y=mat_train[:, -1:].astype(int),
)
np.savez_compressed(
    out_test_file,
    x=mat_test[:, :-1].astype(float),
    y=mat_test[:, -1:].astype(int),
    true_y=mat_test[:, -1:].astype(int),
)
