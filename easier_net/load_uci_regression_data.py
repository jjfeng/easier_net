import pandas as pd
import numpy as np
from sklearn.datasets import load_boston


def normalize_cols(mat, num_train):
    for col in range(mat.shape[1]):
        col_train_mean = mat[:num_train, col].mean()
        col_train_sd = np.sqrt(np.var(mat[:num_train, col]))
        mat[:, col] -= col_train_mean
        if col_train_sd > 0:
            mat[:, col] /= col_train_sd


"""
Regression datas
"""
np.random.seed(0)

"""
CT slices
"""
out_train_file = "_output/ct_slices_train.npz"
out_test_file = "_output/ct_slices_test.npz"
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip"
)
print(df)
mat = df.to_numpy()[:, 1:]
np.random.shuffle(mat)
num_train = mat.shape[0] // 3 * 2

normalize_cols(mat, num_train)

# Predict sale price
np.savez_compressed(
    out_train_file,
    x=mat[:num_train, :-1],
    y=mat[:num_train, -1:],
    true_y=mat[:num_train, -1:],
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, :-1],
    y=mat[num_train:, -1:],
    true_y=mat[num_train:, -1:],
)

"""
crime
"""
out_train_file = "_output/crime_train.npz"
out_test_file = "_output/crime_test.npz"
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
    names=range(128),
)
mat = df.to_numpy()[:, 5:]
np.random.shuffle(mat)
num_train = mat.shape[0] // 3 * 2

good_cols = np.where(
    [np.mean(mat[:, col] == "?") < 0.1 for col in range(mat.shape[1])]
)[0]
mat = mat[:, good_cols]
mat[mat == "?"] = 0
mat = mat.astype(float)

print(good_cols)
normalize_cols(mat, num_train)

# Predict sale price
np.savez_compressed(
    out_train_file,
    x=mat[:num_train, :-1],
    y=mat[:num_train, -1:],
    true_y=mat[:num_train, -1:],
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, :-1],
    y=mat[num_train:, -1:],
    true_y=mat[num_train:, -1:],
)

"""
Boston
"""
out_train_file = "_output/boston_train.npz"
out_test_file = "_output/boston_test.npz"
X, y = load_boston(return_X_y=True)
print(y.shape, X.shape)
mat = np.hstack([X, y.reshape((-1, 1))])
np.random.shuffle(mat)
num_train = mat.shape[0] // 3 * 2

normalize_cols(mat, num_train)

# Predict sale price
np.savez_compressed(
    out_train_file,
    x=mat[:num_train, :-1],
    y=mat[:num_train, -1:],
    true_y=mat[:num_train, -1:],
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, :-1],
    y=mat[num_train:, -1:],
    true_y=mat[num_train:, -1:],
)

"""
Residential Building Data Set Data Set

sale pice last column,
construction cost second to last column
"""
out_train_file = "_output/iran_house_train.npz"
out_test_file = "_output/iran_house_test.npz"
df = pd.read_excel(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00437/Residential-Building-Data-Set.xlsx",
    header=1,
)
mat = df.to_numpy()
np.random.shuffle(mat)
num_train = mat.shape[0] // 3 * 2

normalize_cols(mat, num_train)

# Predict sale price
np.savez_compressed(
    out_train_file,
    x=mat[:num_train, :-2],
    y=mat[:num_train, -1:],
    true_y=mat[:num_train, -1:],
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, :-2],
    y=mat[num_train:, -1:],
    true_y=mat[num_train:, -1:],
)

"""
wine
"""
out_train_file = "_output/wine_train.npz"
out_test_file = "_output/wine_test.npz"

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    delimiter=";",
)
mat = df.to_numpy()
np.random.shuffle(mat)
print("tot data", mat.shape[0])
num_train = mat.shape[0] // 3 * 2

normalize_cols(mat, num_train)

np.savez_compressed(
    out_train_file,
    x=mat[:num_train, :-1],
    y=mat[:num_train, -1:],
    true_y=mat[:num_train, -1:],
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, :-1],
    y=mat[num_train:, -1:],
    true_y=mat[num_train:, -1:],
)

"""
gene cancer
"""
out_train_file = "_output/gene_cancer_train.npz"
out_test_file = "_output/gene_cancer_test.npz"

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"
)
# There are only 801 samples. The extra rows are all nan
# First columns is meaningless. Let's try to predict the expr of the second gene in this dataset
df = df.iloc[:801, 2:]
mat = df.to_numpy()
mat = np.nan_to_num(mat)
np.random.shuffle(mat)
num_train = mat.shape[0] // 3 * 2

normalize_cols(mat, num_train)

np.savez_compressed(
    out_train_file,
    x=mat[:num_train, 1:],
    y=mat[:num_train, :1],
    true_y=mat[:num_train, :1],
)
np.savez_compressed(
    out_test_file,
    x=mat[num_train:, 1:],
    y=mat[num_train:, :1],
    true_y=mat[num_train:, :1],
)
