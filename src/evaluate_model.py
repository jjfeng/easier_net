import argparse
import sys
import logging
import json
import joblib

import numpy as np
import pandas as pd

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from network import Net, SierNet
from constants import THRES


def parse_args(args):
    """ parse command line arguments """

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("data_file", type=str)
    parser.add_argument("model_file", type=str)
    parser.add_argument(
        "model_loader_class",
        type=str,
        choices=["xgb", "scikit", "easier_net", "plain_nnet"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random number generator seed for replicability",
        default=12,
    )
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--dump-net-struct", action="store_true", default=False)
    parser.add_argument("--log-file", type=str, default="_output/eval.txt")
    parser.add_argument("--out-file", type=str, default="_output/eval.csv")
    parser.add_argument("--json-file", type=str, default=None)
    parser.set_defaults()
    args = parser.parse_args()
    return args


def load_plain_nn(model_file):
    meta_state_dict = torch.load(model_file)
    model = Net(
        n_layers=meta_state_dict["n_layers"],
        n_input=meta_state_dict["n_inputs"],
        n_hidden=meta_state_dict["n_hidden"],
        n_out=meta_state_dict["n_out"],
        dropout=meta_state_dict["dropout"],
        input_filter_layer=meta_state_dict["input_filter_layer"],
    )
    model.load_state_dict(meta_state_dict["state_dict"])
    return model, meta_state_dict


def load_easier_net(model_file):
    meta_state_dict = torch.load(model_file)
    model = SierNet(
        n_layers=meta_state_dict["n_layers"],
        n_input=meta_state_dict["n_inputs"],
        n_hidden=meta_state_dict["n_hidden"],
        n_out=meta_state_dict["n_out"],
        input_filter_layer=meta_state_dict["input_filter_layer"],
    )
    model.load_state_dict(meta_state_dict["state_dicts"][0])
    return model, meta_state_dict


def do_model_inference(model_loader_class, model_file, num_classes, x):
    if model_loader_class == "xgb":
        if num_classes == 0:
            model = xgb.XGBRegressor()
            model.load_model(model_file)
            outputs = model.predict(x)
        else:
            model = xgb.XGBClassifier()
            model.load_model(model_file)
            outputs = np.log(model.predict_proba(x))
    elif model_loader_class == "scikit":
        model = joblib.load(model_file)
        if num_classes == 0:
            outputs = model.predict(x)
        else:
            outputs = model.predict_log_proba(x)
    else:
        meta_state_dict = torch.load(model_file)
        if model_loader_class == "easier_net":
            model = SierNet(
                n_layers=meta_state_dict["n_layers"],
                n_input=meta_state_dict["n_inputs"],
                n_hidden=meta_state_dict["n_hidden"],
                n_out=meta_state_dict["n_out"],
                input_filter_layer=meta_state_dict["input_filter_layer"],
            )
            model.load_state_dict(meta_state_dict["state_dicts"][0])
        elif model_loader_class == "plain_nnet":
            model = Net(
                n_layers=meta_state_dict["n_layers"],
                n_input=meta_state_dict["n_inputs"],
                n_hidden=meta_state_dict["n_hidden"],
                n_out=meta_state_dict["n_out"],
                dropout=meta_state_dict["dropout"],
                input_filter_layer=meta_state_dict["input_filter_layer"],
            )
            model.load_state_dict(meta_state_dict["state_dict"])
        model.get_net_struct()
        model.eval()
        if model.is_regression:
            outputs = model.predict(x)
        else:
            outputs = model.predict_log_proba(x)
    return outputs, model


def evaluate_model(is_regression, outputs, true_y):
    if is_regression:
        assert outputs.size == true_y.size
        empirical_loss = np.mean(np.power(outputs.flatten() - true_y.flatten(), 2))
        print("OUTPUTS var", np.var(outputs))
        print("OUTPUTS", outputs.flatten()[:10])
        print("TRUE", true_y.flatten()[:10])
        logging.info(f"test MSE LOSS {empirical_loss}")
        print(f"test MSE LOSS {empirical_loss}")
    else:
        log_prob_class = np.array(
            [outputs[i, true_y[i]] for i in range(true_y.shape[0])]
        ).flatten()
        print(np.median(log_prob_class))
        # print("LOG PROB SORT", np.sort(log_prob_class))
        print(np.mean(np.sort(log_prob_class)[10:]))
        # plt.hist(log_prob_class)
        # plt.savefig("_output/fig.png")
        empirical_loss = -np.mean(log_prob_class)
        print("neg log lik %f" % empirical_loss)
        logging.info(f"neg log lik {empirical_loss}")
    return empirical_loss


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(
        format="%(message)s", filename=args.log_file, level=logging.DEBUG
    )
    print(args)
    logging.info(args)
    np.random.seed(args.seed)

    # Load data
    dataset_dict = np.load(args.data_file)
    x = dataset_dict["x"]
    y = dataset_dict["y"]
    true_y = dataset_dict["true_y"]

    # Load model
    outputs, model = do_model_inference(
        args.model_loader_class, args.model_file, args.num_classes, x
    )
    # Evaluate model
    empirical_loss = evaluate_model(args.num_classes == 0, outputs, true_y)

    loss_df = pd.DataFrame(
        {
            "model_class": [
                model.__class__.__name__ if args.model_name is None else args.model_name
            ],
            "test_loss": [empirical_loss],
        }
    )
    loss_df.to_csv(open(args.out_file, "w"))

    if (
        args.dump_net_struct
        and args.json_file
        and args.model_loader_class in ["easier_net", "plain_nnet"]
    ):
        res_dict = {
            "model_class": model.__class__.__name__
            if args.model_name is None
            else args.model_name,
            "test_loss": float(empirical_loss),
        }
        net_struct = model.get_net_struct()
        for k, v in net_struct.items():
            res_dict[k] = float(v)

        res_dict["support"] = model.support().tolist()

        if args.model_loader_class == "easier_net":
            # Also get importance values...
            importance_dict = model.get_importance(x)
            for i, importance in enumerate(importance_dict):
                res_dict[f"importance_{i}"] = float(importance)
                if importance > THRES:
                    res_dict["max_layer"] = i

        with open(args.json_file, "w") as f:
            json.dump(res_dict, f)


if __name__ == "__main__":
    main(sys.argv[1:])
