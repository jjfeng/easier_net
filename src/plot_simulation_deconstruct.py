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

STUPID_FILE_MAP = {
    "test_ensemble_easier_net.csv": "res_easier_net.json",
    "test_ensemble_sparse.csv": "res_sparse.json",
    "test_ensemble_dropout.csv": "res_dropout.json",
    "test_dropout.csv": "res_dropout.json",
    "test_sparse.csv": "res_sparse.json",
    "test_easier_net.csv": "res_easier_net.json",
}


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--sparse-single", type=str,
    )
    parser.add_argument(
        "--sier-net", type=str,
    )
    parser.add_argument(
        "--dropout-single", type=str,
    )
    parser.add_argument(
        "--sparse-ensemble", type=str,
    )
    parser.add_argument(
        "--easier-net", type=str,
    )
    parser.add_argument(
        "--dropout-ensemble", type=str,
    )
    parser.add_argument(
        "--out-file", type=str,
    )
    parser.set_defaults()
    args = parser.parse_args()

    return args


def load_network_struct(res_file: str):
    res_file_split = res_file.rsplit("/", 1)
    if "seed_init" not in res_file:
        res_json_files = glob.glob(
            os.path.join(
                res_file_split[0], "seed_init_*", STUPID_FILE_MAP[res_file_split[1]]
            )
        )
        assert len(res_json_files)
        res_dicts = []
        for res_json_file in res_json_files:
            with open(res_json_file, "r") as f:
                res_json = json.load(f)
                res_dicts.append(
                    {
                        "max_layer": res_json["max_layer"],
                        "hidden_size_avg": res_json["hidden_size_avg"],
                        "support_size": res_json["support_size"],
                    }
                )
        return pd.DataFrame(pd.DataFrame(res_dicts).mean()).T
    else:
        res_json_file = os.path.join(
            res_file_split[0], STUPID_FILE_MAP[res_file_split[1]]
        )
        with open(res_json_file, "r") as f:
            res_json = json.load(f)
        print(res_json_file)
        print(res_json)
        return pd.DataFrame(
            [
                {
                    "max_layer": res_json["max_layer"],
                    "hidden_size_avg": res_json["hidden_size_avg"],
                    "support_size": res_json["support_size"],
                }
            ]
        )


def get_best(res_files, model_str):
    val_res_template = res_files.replace("test", "val")
    val_res_files = glob.glob(val_res_template)
    print(val_res_template)
    print(model_str, "num files found", len(val_res_files))
    val_res = pd.concat(
        [pd.read_csv(f, index_col=0) for f in val_res_files]
    ).reset_index(drop=True)
    best_idx = np.argmin(val_res.test_loss)
    best_res_file = val_res_files[best_idx].replace("val_", "test_")
    test_loss = pd.read_csv(best_res_file, index_col=0).test_loss[0]
    res = pd.DataFrame(
        [
            {
                "model": model_str,
                "test_loss": test_loss,
                "val_loss": val_res.test_loss[best_idx],
            }
        ]
    )
    network_struct = load_network_struct(best_res_file)
    res = pd.concat([res, network_struct], axis=1)
    print(best_res_file)
    return res


def main(args=sys.argv[1:]):
    args = parse_args(args)

    res = pd.concat(
        [
            get_best(args.dropout_single, "Dropout-Single"),
            get_best(args.dropout_ensemble, "Dropout-Ensemble"),
            get_best(args.sparse_single, "Sparse-Single"),
            get_best(args.sparse_ensemble, "Sparse-Ensemble"),
            get_best(args.sier_net, "SIER-net"),
            get_best(args.easier_net, "EASIER-net"),
        ]
    )
    print(res)
    res.to_latex(args.out_file, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
