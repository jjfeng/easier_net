import argparse
import sys
import logging
import json
import glob
import copy

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import torch

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

from evaluate_model import load_easier_net


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "all_resnet", type=str,
    )
    parser.add_argument(
        "--data-file", type=str,
    )
    parser.add_argument(
        "--out-importance-plot-file", type=str,
    )
    parser.set_defaults()
    args = parser.parse_args()

    args.all_resnet = glob.glob(args.all_resnet)
    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)

    dataset_dict = np.load(args.data_file)
    x = dataset_dict["x"]

    connect_factors = []
    for f in args.all_resnet:
        print("files", f)
        model, meta_state_dict = load_easier_net(f)

        importance = np.array(model.get_importance(x))
        print(importance)
        max_import = np.max(np.where(importance > 0)[0])

        max_layer = model.get_net_struct()["max_layer"]
        meta_state_dict.pop("state_dicts")
        for i in range(importance.size):
            meta_state_dict_cp = copy.deepcopy(meta_state_dict)
            meta_state_dict_cp["importance"] = importance[i]
            meta_state_dict_cp["layer"] = f"Layer {i + 1}"
            meta_state_dict_cp["max_layer"] = max_layer
            connect_factors.append(meta_state_dict_cp)
    connect_factors = pd.DataFrame(connect_factors)
    print(connect_factors)

    # Make the plot
    sns.set_context("paper", font_scale=1.5)

    plt.clf()
    ax = sns.lineplot(
        x="full_tree_pen", y="importance", hue="layer", data=connect_factors
    )
    ax.set(xlim=(1e-6, 2), xscale="log")
    plt.xlabel(r"Penalty parameter $\lambda_2$")
    plt.ylabel("Proportion of variance")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])

    plt.tight_layout()
    sns.despine()
    plt.savefig(args.out_importance_plot_file)


if __name__ == "__main__":
    main(sys.argv[1:])
