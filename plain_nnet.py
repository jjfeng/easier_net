import logging
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from network import Net


class PlainNetEstimator:
    def __init__(
        self,
        n_inputs: int,
        n_layers: int,
        n_hidden: int,
        n_out: int,
        full_tree_pen: float,
        input_pen: float,
        max_iters: int,
        max_prox_iters: int,
        batch_size: int,
        num_classes: int,
        weight: list,
        dropout: float,
        input_filter_layer: bool = False,
    ):
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.full_tree_pen = full_tree_pen
        self.input_pen = input_pen
        self.input_filter_layer = input_filter_layer

        self.max_iters = max_iters
        self.max_prox_iters = max_prox_iters
        self.batch_size = batch_size
        self.dropout = dropout

        self.num_classes = num_classes
        assert num_classes != 1
        self.weight = weight
        self.criterion = (
            nn.CrossEntropyLoss(weight=torch.Tensor(weight))
            if self.num_classes >= 2
            else nn.MSELoss()
        )
        self.score_criterion = (
            nn.CrossEntropyLoss() if self.num_classes >= 2 else nn.MSELoss()
        )

    def run_prox_gd_step(self, trainloader, step_sizes=[1, 0.1, 1e-2, 1e-3, 1e-4]):
        def _soft_threshold(parameter, soft_thres):
            """
            Do soft thresholding (prox operator for lasso)
            """
            soft_thres_mask = torch.abs(parameter.data) < soft_thres
            soft_thres_pos_mask = parameter.data > soft_thres
            soft_thres_neg_mask = parameter.data < -soft_thres
            parameter.data[soft_thres_mask] = 0
            parameter.data[soft_thres_pos_mask] -= soft_thres
            parameter.data[soft_thres_neg_mask] += soft_thres

        optimizer = optim.SGD(self.net.parameters(), lr=1)

        state_dict = copy.deepcopy(self.net.state_dict())
        for i, data in enumerate(trainloader):
            # get the inputs;  is a list of [inputs, labels]
            inputs, labels = data
            labels = labels if self.num_classes == 0 else labels[:, 0]

            for step_size in step_sizes:
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self.net.forward(inputs)
                empirical_loss = self.criterion(outputs, labels)
                loss = (
                    empirical_loss
                    + self.full_tree_pen * self.net.weight_matrix_norm()
                    + self.input_pen * self.net.input_factor_norm()
                )
                empirical_loss.backward()
                # Update params with respect to smooth loss gradient
                for p in self.net.parameters():
                    p.grad *= step_size
                optimizer.step()

                # Now apply prox step
                # Recall that the lasso penalty applies to the bias as well if doing classification
                soft_thres_full_tree = step_size * self.full_tree_pen
                for layer in self.net.layers:
                    _soft_threshold(layer.weight, soft_thres_full_tree)
                    if self.num_classes >= 2:
                        _soft_threshold(layer.bias, soft_thres_full_tree)
                if self.net.input_filter_layer:
                    soft_thres_input = step_size * self.input_pen
                    _soft_threshold(self.net.input_factors, soft_thres_input)

                # reevaluate
                optimizer.zero_grad()
                outputs = self.net.forward(inputs)
                new_empirical_loss = self.criterion(outputs, labels)
                new_loss = (
                    new_empirical_loss
                    + self.full_tree_pen * self.net.weight_matrix_norm()
                    + self.input_pen * self.net.input_factor_norm()
                )
                if new_loss < loss:
                    logging.info(
                        f"success w step size {step_size}. new loss {new_loss} old loss {loss}"
                    )
                    return True, new_loss.item(), new_empirical_loss.item()
                else:
                    logging.info(
                        f"try smaller step size. this step size {step_size} didnt work"
                    )
                    # Reset model parameters
                    self.net.load_state_dict(state_dict)

        return False, loss.item(), empirical_loss.item()

    def run_epoch(self, optimizer, trainloader):
        for i, data in enumerate(trainloader):
            # get the inputs;  is a list of [inputs, labels]
            inputs, labels = data
            labels = labels if self.num_classes == 0 else labels[:, 0]

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = self.net.forward(inputs)
            empirical_loss = self.criterion(outputs, labels)
            loss = (
                empirical_loss
                + self.full_tree_pen * self.net.weight_matrix_norm()
                + self.input_pen * self.net.input_factor_norm()
            )
            loss.backward()
            optimizer.step()
        return loss.item(), empirical_loss.item()

    def fit(self, x: np.ndarray, y: np.ndarray, state_dict: dict = None) -> None:
        # Assemble data
        torch_y = (
            torch.Tensor(y)
            if self.num_classes == 0
            else torch.from_numpy(y.astype(int))
        )
        dataset = TensorDataset(torch.Tensor(x), torch_y)

        trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.net = Net(
            n_layers=self.n_layers,
            n_input=self.n_inputs,
            n_hidden=self.n_hidden,
            n_out=self.n_out,
            dropout=self.dropout,
            input_filter_layer=self.input_filter_layer,
        )
        if state_dict is not None:
            self.net.load_state_dict(state_dict)
            print("loaded state dict")
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        for epoch in range(self.max_iters):  # loop over the set multiple times
            loss, empirical_loss = self.run_epoch(optimizer, trainloader)

            # print statistics
            if epoch % 100 == 0:
                logging.info(f"[{epoch}] loss: {loss}")
                logging.info(f"[{epoch}] empirical: {empirical_loss}")
                print(f"[{epoch}] loss: {loss} empirical: {empirical_loss}")

        trainloader = DataLoader(dataset, batch_size=x.shape[0])
        for i in range(self.max_prox_iters):
            did_step, loss, empirical_loss = self.run_prox_gd_step(trainloader)
            if epoch % 10 == 0:
                logging.info(f"[prox {i}] loss: {loss}")
                logging.info(f"[prox {i}] empirical_loss: {empirical_loss}")
                self.net.get_net_struct()

            if not did_step:
                break

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        torch_y = (
            torch.Tensor(y)
            if self.num_classes == 0
            else torch.from_numpy(y.astype(int))[:, 0]
        )
        return -self.score_criterion(self.net(torch.Tensor(x)), torch_y).item()

    def predict(self, x: np.ndarray) -> np.ndarray:
        output = self.net(torch.Tensor(x)).detach().numpy()
        if self.num_classes == 0:
            return output
        else:
            raise NotImplementedError()
            # return np.exp(output)/np.sum(np.exp(output), axis=1, keepdims=True)

    def get_params(self, deep=True) -> dict:
        return {
            "n_layers": self.n_layers,
            "n_inputs": self.n_inputs,
            "n_hidden": self.n_hidden,
            "n_out": self.n_out,
            "full_tree_pen": self.full_tree_pen,
            "input_pen": self.input_pen,
            "max_iters": self.max_iters,
            "max_prox_iters": self.max_prox_iters,
            "batch_size": self.batch_size,
            "num_classes": self.num_classes,
            "weight": self.weight,
            "dropout": self.dropout,
            "input_filter_layer": self.input_filter_layer,
        }

    def set_params(self, **param_dict):
        print(param_dict)
        if "n_layers" in param_dict:
            self.n_layers = param_dict["n_layers"]
        if "n_inputs" in param_dict:
            self.n_inputs = param_dict["n_inputs"]
        if "n_hidden" in param_dict:
            self.n_hidden = param_dict["n_hidden"]
        if "n_out" in param_dict:
            self.n_out = param_dict["n_out"]
        if "full_tree_pen" in param_dict:
            self.full_tree_pen = param_dict["full_tree_pen"]
        if "input_pen" in param_dict:
            self.input_pen = param_dict["input_pen"]
        if "max_iters" in param_dict:
            self.max_iters = param_dict["max_iters"]
        if "max_prox_iters" in param_dict:
            self.max_prox_iters = param_dict["max_prox_iters"]
        if "batch_size" in param_dict:
            self.batch_size = param_dict["batch_size"]
        if "num_classes" in param_dict:
            self.num_classes = param_dict["num_classes"]
        if "weight" in param_dict:
            self.weight = param_dict["weight"]
        if "dropout" in param_dict:
            self.dropout = param_dict["dropout"]
        if "input_filter_layer" in param_dict:
            self.input_filter_layer = param_dict["input_filter_layer"]
        return self