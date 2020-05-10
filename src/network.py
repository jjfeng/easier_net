import logging
from collections import OrderedDict

import numpy as np
from scipy.stats import bernoulli
from scipy.special import logsumexp

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    PyTorch implementation of a simple dense neural network
    """
    def __init__(
        self,
        n_layers,
        n_input,
        n_hidden=5,
        n_out=1,
        dropout=0,
        make_output=True,
        input_filter_layer=False,
    ):
        """
        @param n_layers: number of hidden layers
        @param n_input: number of input nodes
        @param n_hidden: number of hidden nodes per layer
        @param n_out: number of output nodes
        @param dropout: dropout rate
        @param make_output: whether or not to make the output layer
        @param input_filter_layer: whether to add an input filter layer
        """
        super().__init__()
        self.input_filter_layer = input_filter_layer
        if input_filter_layer:
            self.input_factors = nn.Parameter(0.5 * torch.ones(n_input))
        else:
            self.input_factors = torch.ones(n_input)

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_out = n_out
        self.middle_layers = OrderedDict()
        for i in range(n_layers):
            self.middle_layers[str(i)] = nn.Linear(n_hidden, n_hidden)
        self.middle_layer_seq = nn.Sequential(self.middle_layers)

        self.input_layer = nn.Linear(n_input, n_hidden)
        self.layers = [self.input_layer] + list(self.middle_layers.values())
        if make_output:
            self.output_layer = nn.Linear(n_hidden, n_out)
            self.layers += [self.output_layer]

        self.dropout = dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)

    @property
    def is_regression(self):
        return self.n_out == 1

    def forward(self, x):
        if self.input_filter_layer:
            x = self.input_factors * x
        x = self.input_layer(x)
        for idx, mid_layer in self.middle_layers.items():
            x = mid_layer(F.relu(x))
            if self.dropout > 0 and self.dropout_layer.p > 0:
                x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(torch.Tensor(x)).detach().numpy()

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        raw_log_probs = F.log_softmax(self.forward(torch.Tensor(x)), 1).detach().numpy()
        return raw_log_probs

    def norm(self):
        return sum([torch.sum(torch.abs(p)) for p in self.parameters()])

    def input_factor_norm(self):
        return torch.norm(self.input_factors, p=1) if self.input_filter_layer else 0

    def input_layer_norm(self):
        return torch.sum(torch.abs(self.input_layer.weight), 0)

    def layer_norms(self):
        return (
            [torch.sum(torch.abs(self.input_layer.weight), 0).detach().numpy()]
            + [
                torch.sum(torch.abs(layer.weight), 0).detach().numpy()
                for layer in self.middle_layers.values()
            ]
            + [torch.sum(torch.abs(self.output_layer.weight), 0).detach().numpy()]
        )

    def weight_matrix_norm(self):
        bias_norm = (
            sum([torch.norm(layer.bias, p=1) for layer in self.middle_layers.values()])
            + torch.norm(self.input_layer.bias, p=1)
            + torch.norm(self.output_layer.bias, p=1)
        )
        weight_norm = (
            sum(
                [torch.norm(layer.weight, p=1) for layer in self.middle_layers.values()]
            )
            + torch.norm(self.input_layer.weight, p=1)
            + torch.norm(self.output_layer.weight, p=1)
        )

        if self.n_out >= 2:
            return bias_norm + weight_norm
        else:
            return weight_norm

    def support(self, threshold: float = 1e-10):
        input_norms = self.input_layer_norm()
        return input_norms > threshold

    def get_net_struct(self, thres: float = 1e-10) -> dict:
        in_factors = self.input_factors.detach().numpy()
        logging.info(f"in factors {in_factors}")
        if self.input_filter_layer:
            input_factors = np.abs(self.input_factors.detach().numpy())
            support_size = np.sum(input_factors > thres)
            logging.info(f"input factors sort {np.flip(np.sort(input_factors))}")
            logging.info(f"input factors argsort {np.flip(np.argsort(input_factors))}")
            logging.info(f"input factors num big {np.sum(input_factors > thres)}")
            logging.info(
                f"input factors where big {np.where(input_factors > thres)[0]}"
            )
        hidden_sizes = []
        all_layer_norms = self.layer_norms()
        for layer_idx, layer_norms in enumerate(all_layer_norms):
            argsorted_norms = np.flip(np.argsort(layer_norms))
            sorted_norms = np.flip(np.sort(layer_norms))
            if layer_idx == 0:
                inner_support_size = np.sum(layer_norms > thres)
                logging.info(f"INPUT layer:norms {layer_norms}")
                logging.info(f"INPUT layer:arg sort layer norms {argsorted_norms}")
                logging.info(f"INPUT layer:sorted layer norms {sorted_norms}")
                logging.info(f"INPUT layer:support size {inner_support_size}")
                if not self.input_filter_layer:
                    support_size = inner_support_size
            else:
                num_hidden = np.sum(layer_norms > thres)
                hidden_sizes.append(num_hidden)
                logging.info(f"layer {layer_idx}:norms {layer_norms}")
                logging.info(
                    f"layer {layer_idx}:arg sort layer norms {argsorted_norms}"
                )
                logging.info(f"layer {layer_idx}:sorted layer norms {sorted_norms}")
                logging.info(f"layer {layer_idx}:num hidden {num_hidden}")
        return {
            "max_layer": self.n_layers,
            "support_size": support_size,
            "hidden_size_avg": np.mean(hidden_sizes),
        }


class SierNet(Net):
    """
    PyTorch implementation of the actual model in SIER-net
    """
    def __init__(
        self,
        n_layers: int,
        n_input: int,
        n_hidden: int = 5,
        n_out: int = 1,
        input_filter_layer: bool = True,
    ):
        """
        @param n_layers: number of hidden layers (at initialization time)
        @param n_input: number of input nodes (at initialization time)
        @param n_hidden: number of hidden nodes per layer (at initialization time)
        @param n_out: number of output nodes
        @param input_filter_layer: whether to add an input filter layer
        """
        super().__init__(
            n_layers,
            n_input,
            n_hidden=n_hidden,
            n_out=n_out,
            make_output=False,
            dropout=0,
            input_filter_layer=input_filter_layer,
        )

        self.middle_layers = OrderedDict()
        for i in range(n_layers):
            self.middle_layers[str(i)] = nn.Linear(n_hidden, n_hidden)
        self.middle_layer_seq = nn.Sequential(self.middle_layers)

        self.input_layer = nn.Linear(n_input, n_hidden)

        self.input_to_out_layer = nn.Linear(n_input, n_out)
        self.to_out_layers = OrderedDict()
        for i in range(n_layers):
            self.to_out_layers[str(i)] = nn.Linear(n_hidden, n_out)
        self.to_out_layers_seq = nn.Sequential(self.to_out_layers)

        self.layers = (
            list(self.middle_layers.values())
            + list(self.to_out_layers.values())
            + [self.input_layer]
        )
        self.connection_factors = nn.Parameter(torch.ones(self.n_layers + 1))

    def _forward(self, x):
        x = self.input_factors * x
        intermed_out = [self.input_layer(x)] + [None for _ in range(self.n_layers)]
        for idx, mid_layer in enumerate(self.middle_layers.values()):
            intermed_out[idx + 1] = mid_layer(F.relu(intermed_out[idx]))

        # skip connection layers
        abs_connection_factors = torch.abs(self.connection_factors)
        tot_connection_factor = torch.sum(abs_connection_factors)
        normalized_connection_factors = abs_connection_factors / tot_connection_factor
        layer_outputs = [
            normalized_connection_factors[0] * self.input_to_out_layer(x)
        ] + [
            normalized_connection_factors[idx + 1] * to_out_layer(intermed_out[idx + 1])
            for idx, to_out_layer in enumerate(self.to_out_layers.values())
        ]

        return layer_outputs

    def forward(self, x):
        """
        @return model prediction
        """
        layer_outputs = self._forward(x)
        return sum(layer_outputs)

    def connection_norm(self):
        """
        @return norm of the skip-connection weights
        """
        return torch.norm(self.connection_factors, p=1)

    def input_norm(self):
        """
        @return the lasso penalty on the weights connected to the inputs
        """
        weight_norm = torch.norm(self.input_factors, p=1) + torch.norm(
            self.input_to_out_layer.weight, p=1
        )
        bias_norm = torch.norm(self.input_to_out_layer.bias, p=1)
        if self.is_regression:
            return weight_norm
        else:
            return bias_norm + weight_norm

    def weight_matrix_norm(self):
        """
        @return the lasso penalty on the network weights that are not connected to the inputs
        """
        bias_norm = sum([torch.norm(layer.bias, p=1) for layer in self.layers])
        weight_norm = sum([torch.norm(layer.weight, p=1) for layer in self.layers])
        if self.is_regression:
            return weight_norm
        else:
            return bias_norm + weight_norm

    def layer_norms(self):
        """
        @return the norm of each layer
        """
        return [torch.sum(torch.abs(self.input_layer.weight), 0).detach().numpy()] + [
            torch.sum(torch.abs(layer.weight), 0).detach().numpy()
            for layer in self.middle_layers.values()
        ]

    def to_out_layer_norms(self):
        """
        @return the norm of weights connected to the output, partitioned by the layer
        """
        return [
            torch.sum(torch.abs(self.input_to_out_layer.weight), 0).detach().numpy()
        ] + [
            torch.sum(torch.abs(layer.weight), 0).detach().numpy()
            for layer in self.to_out_layers.values()
        ]

    def support(self, threshold: float = 1e-10):
        """
        @return the support of the model as determined by the input filter layer
        """
        return np.array(
            torch.abs(self.input_factors).detach().numpy() > threshold, dtype=int
        )

    def get_importance(self, x: np.ndarray) -> list:
        """
        @return the proportion of variance contributed by each layer
        """
        layer_outputs = self._forward(torch.Tensor(x))
        final_output_var = np.var(sum(layer_outputs).detach().numpy(), axis=0)
        layer_importances_raw = [
            np.var(layer.detach().numpy(), axis=0) for layer in layer_outputs
        ]
        layer_importances = [
            np.mean(l_import / final_output_var) for l_import in layer_importances_raw
        ]
        return layer_importances

    def get_net_struct(self, thres: float = 0) -> dict:
        """
        @return a summary of the network structure
        """
        connection_fac = np.abs(self.connection_factors.detach().numpy())
        connection_fac /= np.sum(connection_fac)
        logging.info(f"connection factors {connection_fac}")
        input_factors = np.abs(self.input_factors.detach().numpy())
        support_size = np.sum(input_factors > thres)
        support = input_factors > thres
        logging.info(f"input factors sort {np.flip(np.sort(input_factors))}")
        logging.info(f"input factors argsort {np.flip(np.argsort(input_factors))}")
        logging.info(f"input factors num big {np.sum(input_factors > thres)}")
        logging.info(f"input factors where big {np.where(input_factors > thres)[0]}")
        hidden_sizes = []
        all_layer_norms = self.layer_norms()
        all_to_out_layer_norms = self.to_out_layer_norms()
        for layer_idx, layer_norms in enumerate(all_layer_norms):
            # if np.abs(connection_fac[layer_idx]) < thres:
            #    continue

            argsorted_norms = np.flip(np.argsort(layer_norms))
            sorted_norms = np.flip(np.sort(layer_norms))
            my_to_out_layer_norms = (
                all_to_out_layer_norms[layer_idx]
                if layer_idx < len(all_to_out_layer_norms)
                else None
            )
            to_out_layer_norms = (
                np.flip(np.sort(all_to_out_layer_norms[layer_idx]))
                if layer_idx < len(all_to_out_layer_norms)
                else None
            )
            argsorted_to_out_norms = (
                np.flip(np.argsort(all_to_out_layer_norms[layer_idx]))
                if layer_idx < len(all_to_out_layer_norms)
                else None
            )

            if layer_idx == 0:
                inner_support_size = np.sum(layer_norms > thres)
                logging.info(f"INPUT layer:norms {layer_norms}")
                logging.info(f"INPUT layer:arg sort layer norms {argsorted_norms}")
                logging.info(f"INPUT layer:sorted layer norms {sorted_norms}")
                logging.info(f"INPUT layer:support size {inner_support_size}")
                to_out_support_size = np.sum(to_out_layer_norms > thres)
                logging.info(f"INPUT to out layer:norms {to_out_layer_norms}")
                logging.info(
                    f"INPUT to out layer:arg sort layer norms {argsorted_to_out_norms}"
                )
                logging.info(f"INPUT to out layer:support size {to_out_support_size}")
                logging.info(
                    f"INPUT to out layer:support {np.where(my_to_out_layer_norms > thres)[0]}"
                )
            else:
                if layer_idx < len(all_to_out_layer_norms):
                    layer_norms += all_to_out_layer_norms[layer_idx]
                num_hidden = np.sum(layer_norms > thres)
                hidden_sizes.append(num_hidden)
                logging.info(f"layer {layer_idx}:sorted layer norms {sorted_norms}")
                logging.info(f"layer {layer_idx}:num hidden {num_hidden}")

        hidden_sizes = np.array(hidden_sizes)
        max_layer = max(
            0,
            min(
                np.max(np.where(connection_fac > thres)[0]),
                np.max(np.where(hidden_sizes > 0)[0]) + 1
                if np.sum(hidden_sizes) > 0
                else 0,
            ),
        )
        connection_fac[max_layer + 1 :] = 0
        connection_fac /= np.sum(connection_fac)
        res = {
            "max_layer": max_layer,
            "support_size": support_size,
            "hidden_size_avg": np.mean(hidden_sizes[hidden_sizes > 0])
            if np.sum(hidden_sizes > 0)
            else 0,
        }
        for i in range(connection_fac.size):
            res[f"connect_{i}"] = connection_fac[i]
        for i in range(hidden_sizes.size):
            res[f"hidden_count_{i}"] = hidden_sizes[i]
        return res
