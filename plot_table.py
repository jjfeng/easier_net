import argparse
import json
import sys
import logging
import joblib
import glob

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from common import get_dataset_display_name

METHOD_DICT_NAME = {
    "LogisticRegression": "Logistic Regression",
    "RandomForestClassifier": "Random Forest",
    "RandomForestRegressor": "Random Forest",
    "SIER-net": "SIER-net",
    "EASIER-net": "EASIER-net",
    "plain_nnet": "DropoutNet-Ensemble",
    "XGBClassifier": "XGBoost",
    "XGBRegressor": "XGBoost",
    "Lasso": "Lasso",
}


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "table_file", type=str,
    )
    parser.add_argument(
        "out_plot", type=str,
    )
    parser.add_argument(
        "--ymin", type=float, default=0.0001,
    )
    parser.add_argument(
        "--ymax", type=float, default=1,
    )
    parser.set_defaults()
    args = parser.parse_args()

    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    print(args)

    # Load model results
    res_df = pd.read_csv(args.table_file, index_col=0)
    res_df.columns = ["model_class", "Test loss", "dataset"]
    res_df["Method"] = [METHOD_DICT_NAME[a] for a in res_df.model_class]
    res_df["Dataset"] = [get_dataset_display_name(a) for a in res_df.dataset]
    print(res_df.groupby("Dataset").min().reset_index())
    order = (
        res_df.groupby("Dataset")
        .min()
        .reset_index()
        .sort_values(by="Test loss")["Dataset"]
    )

    palette = sns.color_palette()

    sns.set_context("paper", font_scale=1.2)
    grid = sns.stripplot(
        x="Dataset",
        y="Test loss",
        hue="Method",
        data=res_df,
        jitter=False,
        dodge=False,
        palette=palette,
        order=order,
        hue_order=["EASIER-net", "SIER-net"]
        + [a for a in res_df.Method.unique() if "SIER-net" not in a],
    )
    grid.set(yscale="log", ylim=(args.ymin, args.ymax))
    ax = sns.stripplot(
        x="Dataset",
        y="Test loss",
        data=res_df[res_df.Method == "EASIER-net"],
        # edgecolor='black',
        # linewidth=2,
        order=order,
        color=palette[0],
        marker="X",
        size=8,
        dodge=False,
        jitter=False,
    )
    ax.set(yscale="log", ylim=(args.ymin, args.ymax))
    grid.legend_.__dict__["legendHandles"][0] = Line2D(
        [0],
        [0],
        linewidth=0,
        marker="X",
        color=palette[0],
        label="EASIER-net",
        markerfacecolor="g",
        markersize=10,
    )
    ax.legend(handles=grid.legend_.__dict__["legendHandles"])
    sns.despine()
    plt.tight_layout()
    plt.savefig(args.out_plot)


if __name__ == "__main__":
    main(sys.argv[1:])
