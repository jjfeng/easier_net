import argparse
import json
import sys
import logging
import joblib
import glob

import numpy as np
import pandas as pd

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from evaluate_model import do_model_inference


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "res_file_template", type=str,
    )
    parser.add_argument(
        "out_file", type=str,
    )
    parser.add_argument("--groupby", type=str, default=None)
    parser.add_argument("--newcol", type=str, default=None)
    parser.add_argument("--pivot", type=str, default=None)
    parser.add_argument("--log-file", type=str, default="_output/eval.txt")
    parser.set_defaults()
    args = parser.parse_args()

    args.res_files = glob.glob(args.res_file_template)
    return args


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)

    # Load model results
    if args.res_files[0].endswith("csv"):
        all_outputs = pd.concat(
            [pd.read_csv(res_file, index_col=0) for res_file in args.res_files]
        ).reset_index(drop=True)
    elif args.res_files[0].endswith("json"):
        all_outputs = pd.DataFrame(
            [json.load(open(res_file, "r")) for res_file in args.res_files]
        )

    reduced_outputs = (
        all_outputs.groupby(args.groupby).mean().reset_index()
        if args.groupby is not None
        else all_outputs
    )
    if args.newcol is not None:
        args.newcol = args.newcol.split(",")
        reduced_outputs[args.newcol[0]] = args.newcol[1]

    final_res = reduced_outputs
    if args.pivot:
        args.pivot = args.pivot.split(",")
        final_res = reduced_outputs.pivot(index=args.pivot[0], columns=args.pivot[1])
        print(final_res.mean(axis=1))

    print(final_res)
    if args.out_file.endswith("csv"):
        final_res.to_csv(open(args.out_file, "w"), index=True)
    elif args.out_file.endswith("tex"):
        final_res.to_latex(
            open(args.out_file, "w"), index=True, float_format="{:0.4f}".format
        )


if __name__ == "__main__":
    main(sys.argv[1:])
