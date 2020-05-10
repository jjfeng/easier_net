import argparse
import sys
import logging
import json
import glob
import copy

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

from evaluate_model import load_easier_net


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--n-inputs", type=int,
    )
    parser.add_argument("--corr", type=float)
    parser.add_argument(
        "--fitted-model-files", type=str,
    )
    parser.add_argument(
        "--out-support-file", type=str,
    )
    parser.set_defaults()
    args = parser.parse_args()

    args.fitted_model_files = glob.glob(args.fitted_model_files)
    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)

    # Get the support probabilities
    supports = np.zeros(args.n_inputs)
    print("NUM FILES", len(args.fitted_model_files))
    for f in args.fitted_model_files:
        model, _ = load_easier_net(f)
        support = model.support()
        for i in support:
            supports[i] += 1
    supports /= len(args.fitted_model_files)
    print(supports)

    data = pd.DataFrame({"input": np.arange(args.n_inputs), "prob_support": supports,})
    data["corr"] = args.corr
    data.to_csv(args.out_support_file)


if __name__ == "__main__":
    main(sys.argv[1:])
