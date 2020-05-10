import os
import argparse
import sys
import logging
import json
import glob
import copy

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

from evaluate_model import load_plain_nn, load_easier_net
from constants import THRES
from common import get_dataset_display_name


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--nn-files", type=str,
    )
    parser.add_argument(
        "--seeds", type=int,
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--out-file", type=str, default="_output/structs.txt")
    parser.add_argument("--plot-support-file", type=str, default="_output/structs.pdf")
    parser.set_defaults()
    args = parser.parse_args()

    return args


def load_network_struct(res_files: str, seeds: int):
    res_dicts = []
    all_supports = []
    for seed in range(seeds):
        res_json_file = res_files % seed
        try:
            with open(res_json_file, "r") as f:
                res_json = json.load(f)
                all_supports.append(res_json["support"])

                max_layer_import = np.max(
                    np.where([res_json[f"importance_{i}"] > THRES for i in range(5)])[0]
                )
                res_json["hidden_size_avg"] = (
                    np.mean(
                        [
                            res_json[f"hidden_count_{i}"]
                            for i in range(1, max_layer_import + 1)
                        ]
                    )
                    if max_layer_import >= 1
                    else 0
                )

                pop_keys = ["test_loss", "support"] + [
                    k
                    for k in res_json.keys()
                    if ("connect" in k or "hidden_count" in k)
                ]
                for k in pop_keys:
                    res_json.pop(k)
                print(res_json)
                res_dicts.append(res_json)
        except FileNotFoundError as e:
            continue
    return pd.DataFrame(res_dicts), np.array(all_supports)


def main(args=sys.argv[1:]):
    args = parse_args(args)

    nn_res, all_supports = load_network_struct(args.nn_files, args.seeds)

    sns.set_context("paper", font_scale=2.5)

    # Collate network structurs
    res_agg = nn_res.aggregate(["mean", "var", "count"])
    print(res_agg)
    mean_agg = res_agg.loc["mean"]
    se_agg = np.sqrt(res_agg.loc["var"] / res_agg.loc["count"])
    final_res = pd.DataFrame(
        {
            k: ["%.2f (%.2f)" % (mean_agg[k], se_agg[k])]
            for k in res_agg.columns
            if np.isfinite(se_agg[k])
        }
    )
    final_res.insert(0, "dataset", args.dataset)
    print(final_res)
    final_res.to_csv(args.out_file)

    # Plot supports
    support_sizes = np.sum(all_supports, axis=1)
    support_size_mean = np.mean(support_sizes)
    support_size_se = np.sqrt(np.var(support_sizes) / support_sizes.shape[0])

    sorted_support = np.flip(np.sort(all_supports.mean(axis=0)))
    support_grid = np.arange(sorted_support.size)
    ax = sns.lineplot(x=support_grid, y=sorted_support)
    ax.set(ylim=(-0.1, 1.1))
    title = get_dataset_display_name(args.dataset)
    plt.title("%s: %.1f (%.1f)" % (title, support_size_mean, support_size_se))
    plt.xlabel("Variable index")
    plt.ylabel("Selection rate")
    plt.tight_layout()
    sns.despine()
    plt.savefig(args.plot_support_file)


if __name__ == "__main__":
    main(sys.argv[1:])
