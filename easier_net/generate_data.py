import argparse
import sys
import logging
import pickle

import numpy as np
import itertools

import xgboost as xgb

from data_generator import DataGenerator

def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument("--n-inputs", type=int, default=4)
    parser.add_argument("--n-relevant-inputs", type=int, default=4)
    parser.add_argument("--n-obs", type=int, default=500)
    parser.add_argument("--snr", type=float, default=1)
    parser.add_argument("--x-scale", type=float, default=1)
    parser.add_argument(
        "--mean-func",
        type=str,
        default="curvy",
        choices=["curvy", "line"],
        help="The form of the mean function is only used in regression. For binary classification, we currently use a very simple function for the probability of Y = 1",
    )
    parser.add_argument("--is-classification", action="store_true")
    parser.add_argument("--correlation", type=float, default=0)
    parser.add_argument("--num-corr", type=int, default=0)
    parser.add_argument("--log-file", type=str, default="_output/log_data.txt")
    parser.add_argument("--out-file", type=str, default="_output/data.npz")
    parser.add_argument("--in-model-file", type=str, default=None)
    parser.add_argument("--out-model-file", type=str, default=None)
    parser.set_defaults()
    args = parser.parse_args()
    return args


def classification_default(x):
    return np.maximum(np.minimum(x[:, 0:1], 0.9), 0.1)


class MeanFunc:
    def make_line_regression_func(n_relevant):
        def mean_func(x):
            y = 0
            for i in range(n_relevant // 4):
                start_idx = i * 4
                y += (
                    0.03
                    * (
                        x[:, start_idx + 1 : start_idx + 2]
                        + x[:, start_idx : start_idx + 1]
                    )
                    + (x[:, start_idx + 2 : start_idx + 3])
                    + (x[:, start_idx + 3 : start_idx + 4])
                )
            return y

        return mean_func

    def make_regression_func(n_relevant):
        def mean_func(x):
            y = 0
            for i in range(n_relevant // 4):
                start_idx = i * 4
                y += np.sin(
                    2 * x[:, start_idx + 1 : start_idx + 2]
                    + 2 * x[:, start_idx : start_idx + 1]
                ) + 5 * x[:, start_idx + 2 : start_idx + 3] * np.abs(
                    x[:, start_idx + 3 : start_idx + 4] - 0.25
                )
            return y

        return mean_func


def create_mean_func(args):
    if args.is_classification:
        return classification_default, {"num_true": 1, "func": "classification_default"}
    elif args.mean_func == "curvy":
        regression_func = MeanFunc.make_regression_func(args.n_relevant_inputs)
        return (
            regression_func,
            {"num_true": args.n_relevant_inputs, "func": "make_regression_func"},
        )
    elif args.mean_func == "line":
        regression_func = MeanFunc.make_line_regression_func(args.n_relevant_inputs)
        return (
            regression_func,
            {"num_true": args.n_relevant_inputs, "func": "make_line_regression_func"},
        )
    else:
        raise ValueError("huh")


def load_mean_func(args, model_dict):
    if model_dict["func"] == "classification_default":
        return classification_default
    elif "regression" in model_dict["func"]:
        return getattr(MeanFunc, model_dict["func"])(model_dict["num_true"])


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)
    np.random.seed(args.seed)

    if args.in_model_file is None:
        mean_func, mean_func_dict = create_mean_func(args)
        print("MEAN FUNC", mean_func)
        # make train data
        data_gen = DataGenerator(
            args.n_inputs,
            mean_func,
            x_scale=args.x_scale,
            snr=args.snr,
            is_classification=args.is_classification,
            correlation=args.correlation,
            num_true=mean_func_dict["num_true"],
            num_corr=args.num_corr,
        )
        x, y, true_y, sigma_eps = data_gen.create_data(args.n_obs)
        y_shift = np.mean(y)
        y_scale = np.sqrt(np.var(y))
        print("y_shif", y_shift, "scale", y_scale)
        y = (y - y_shift) / y_scale
        true_y = (true_y - y_shift) / y_scale
    else:
        with open(args.in_model_file, "rb") as f:
            model_dict = pickle.load(f)
        mean_func = load_mean_func(args, model_dict)
        print("MEAN FUNC", mean_func)
        # make train data
        data_gen = DataGenerator(
            model_dict["n_inputs"],
            mean_func,
            x_scale=model_dict["x_scale"],
            y_scale=model_dict["y_scale"],
            y_shift=model_dict["y_shift"],
            sigma_eps=model_dict["sigma_eps"],
            snr=None,
            is_classification=args.is_classification,
            correlation=model_dict["correlation"],
            num_true=model_dict["num_true"],
            num_corr=model_dict["num_corr"],
        )
        x, y, true_y, sigma_eps = data_gen.create_data(args.n_obs)

    print("MEAN Y", np.mean(y))
    print("VAR Y", np.var(y))
    np.savez_compressed(args.out_file, x=x, y=y, true_y=true_y)
    if args.out_model_file is not None:
        with open(args.out_model_file, "wb") as f:
            model_dict = {
                "snr": args.snr,
                "x_scale": args.x_scale,
                "y_scale": y_scale,
                "y_shift": y_shift,
                "sigma_eps": sigma_eps,
                "n_inputs": args.n_inputs,
                "num_corr": args.num_corr,
                "correlation": args.correlation,
            }
            for k, v in mean_func_dict.items():
                model_dict[k] = v
            pickle.dump(model_dict, f)


if __name__ == "__main__":
    main(sys.argv[1:])
