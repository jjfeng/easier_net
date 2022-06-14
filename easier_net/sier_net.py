import logging
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from .network import SierNet

class SierNetEstimator: 
    """
    This first a single sparse-input hierarchical network (SIER-net).
    """
    def __init__(
        self,
        input_filter_layer:bool,
        n_layers: int ,
        n_hidden: int,
        full_tree_pen: float,
        input_pen: float,
        batch_size: int, 
        num_classes: int,
        weight: list,
        max_iters: int,
        max_prox_iters: int,
        connection_pen: float = 0,
        state_dict: dict = None,
    ):
        self.input_filter_layer = input_filter_layer
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_out = 1 if num_classes == 0 else num_classes
        
        self.full_tree_pen = full_tree_pen
        self.connection_pen = connection_pen
        self.input_pen = input_pen

        self.batch_size = batch_size

        self.num_classes = num_classes
        assert num_classes != 1
        self.weight = weight

        self.max_iters = max_iters
        self.max_prox_iters = max_prox_iters
        self.criterion = (
            nn.CrossEntropyLoss(weight=torch.Tensor(weight))
            if self.num_classes >= 2
            else nn.MSELoss()
        )
        
        self.score_criterion = (
            nn.CrossEntropyLoss() if self.num_classes >= 2 else nn.MSELoss()
        )
        
        if state_dict is not None:
            self._load_state_dict(state_dict, strict=False)
            print("loaded state dict")

    def _load_state_dict(self, state_dict):
        # TODO: fill this out later:
        self.net = SierNet(
            n_input=state_dict["n_inputs"],
            input_filter_layer=self.input_filter_layer,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            n_out=self.n_out,
        )

        self.net.load_state_dict(state_dict, strict=False)

    def run_prox_gd_step(
        self, trainloader, step_sizes=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    ):
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
                    + self.input_pen * self.net.input_norm()
                    + self.full_tree_pen * self.net.weight_matrix_norm()
                    + self.connection_pen * self.net.connection_norm()
                )
                empirical_loss.backward()
                # Update params with respect to smooth loss gradient
                for idx, p in enumerate(self.net.parameters()):
                    p.grad *= step_size
                optimizer.step()

                # Now apply prox step
                # Recall that the lasso penalty applies to the bias as well if doing classification
                soft_thres_full_tree = step_size * self.full_tree_pen
                for layer in self.net.layers:
                    if not self.net.is_regression:
                        _soft_threshold(layer.bias, soft_thres_full_tree)
                    _soft_threshold(layer.weight, soft_thres_full_tree)

                if self.input_pen:
                    soft_thres_input = step_size * self.input_pen
                    _soft_threshold(self.net.input_factors, soft_thres_input)
                    _soft_threshold(
                        self.net.input_to_out_layer.weight, soft_thres_input
                    )
                    if not self.net.is_regression:
                        _soft_threshold(
                            self.net.input_to_out_layer.bias, soft_thres_input
                        )

                soft_thres_connection = step_size * self.connection_pen
                _soft_threshold(self.net.connection_factors, soft_thres_connection)

                # reevaluate
                optimizer.zero_grad()
                outputs = self.net.forward(inputs)
                new_empirical_loss = self.criterion(outputs, labels)
                new_loss = (
                    new_empirical_loss
                    + self.input_pen * self.net.input_norm()
                    + self.full_tree_pen * self.net.weight_matrix_norm()
                    + self.connection_pen * self.net.connection_norm()
                )
                if new_loss < loss:
                    logging.info(
                        f"success w step size {step_size}. new loss {new_loss} old loss {loss}"
                    )
                    return True, new_loss.item(), empirical_loss.item()
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
                + self.input_pen * self.net.input_norm()
                + self.full_tree_pen * self.net.weight_matrix_norm()
                + self.connection_pen * self.net.connection_norm()
            )
            loss.backward()

            optimizer.step()
        return loss.item(), empirical_loss.item()

    def get_pen_loss(self, x, y):
        # Assemble data
        torch_y = (
            torch.Tensor(y)
            if self.num_classes == 0
            else torch.from_numpy(y.astype(int))
        )
        dataset = TensorDataset(torch.Tensor(x), torch_y)

        trainloader = DataLoader(dataset, batch_size=x.shape[0], shuffle=True)
        for i, data in enumerate(trainloader):
            # get the inputs;  is a list of [inputs, labels]
            inputs, labels = data
            labels = labels if self.num_classes == 0 else labels[:, 0]

            outputs = self.net.forward(inputs)
            empirical_loss = self.criterion(outputs, labels)
            loss = (
                empirical_loss
                + self.input_pen * self.net.input_norm()
                + self.full_tree_pen * self.net.weight_matrix_norm()
                + self.connection_pen * self.net.connection_norm()
            )
        return loss.item()

    def get_fisher(self, optimizer, trainloader):
        # param_list = [p for layer in self.layers for p in layer.parameters()] + [self.input_factors] + self.input_to_out_layer.parameters() + [self.connection_factors]
        fisher = 0
        num_obs = len(trainloader)
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
                + self.input_pen * self.net.input_norm()
                + self.full_tree_pen * self.net.weight_matrix_norm()
                + self.connection_pen * self.net.connection_norm()
            )
            loss.backward()
            gradient = np.concatenate(
                [p.grad.detach().numpy().flatten() for p in self.net.parameters()]
            ).reshape((-1, 1))
            fisher += np.matmul(gradient, gradient.T)
            if i % 100 == 0:
                print("OBS", i)

        # input_eye = np.eye(self.n_inputs)
        # log_determinant = np.linalg.slogdet(fisher/num_obs + self.input_pen * input_eye + self.full_tree_pen * full_tree_eye)
        print("trying", fisher.shape[0])
        log_determinant = np.linalg.slogdet(
            fisher / num_obs + self.input_pen * np.eye(fisher.shape[0])
        )
        print("DET", log_determinant)
        logging.info(f"LOG DET {log_determinant}")

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ):
        self.n_inputs = x.shape[1]
        self.net = SierNet(
            n_input=self.n_inputs,
            input_filter_layer=self.input_filter_layer,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            n_out=self.n_out,
        )

        torch_y = (
            torch.Tensor(y)
            if self.num_classes == 0
            else torch.from_numpy(y.astype(int))
        )
        dataset = TensorDataset(torch.Tensor(x), torch_y)

        trainloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        for epoch in range(self.max_iters):  # loop over the set multiple times
            loss, empirical_loss = self.run_epoch(optimizer, trainloader)

            # print statistics
            if epoch % 100 == 0:
                logging.info(f"[{epoch}] loss: {loss}")
                logging.info(f"[{epoch}] empirical: {empirical_loss}")
                print(f"[{epoch}] loss: {loss} empirical: {empirical_loss}")

                self.net.get_net_struct()

        trainloader = DataLoader(dataset, batch_size=x.shape[0])
        for i in range(self.max_prox_iters):
            did_step, loss, empirical_loss = self.run_prox_gd_step(trainloader)
            if i % 100 == 0:
                logging.info(f"[prox {i}] loss: {loss}")
                logging.info(f"[prox {i}] empirical_loss: {empirical_loss}")
                print(f"[prox {i}] loss: {loss} empirical: {empirical_loss}")
                self.net.get_net_struct()

            if not did_step:
                break

        return self #added for .fit

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

    def get_params(self, deep=False) -> dict:
        return {
            "n_layers": self.n_layers,
            "input_filter_layer": self.input_filter_layer,
            "n_inputs": self.n_inputs,
            "n_hidden": self.n_hidden,
            "full_tree_pen": self.full_tree_pen,
            "input_pen": self.input_pen,
            "connection_pen": self.connection_pen,
            "batch_size": self.batch_size,
            "num_classes": self.num_classes,
            "max_iters": self.max_iters,
            "max_prox_iters": self.max_prox_iters,
            "weight": self.weight,
        }

    def set_params(self, **param_dict):
        print("SET PARAMS", param_dict)
        if "n_layers" in param_dict:
            self.n_layers = param_dict["n_layers"]
        if "input_filter_layer" in param_dict:
            self.input_filter_layer = param_dict["input_filter_layer"]
        if "n_hidden" in param_dict:
            self.n_hidden = param_dict["n_hidden"]
        if "connection_pen" in param_dict:
            self.connection_pen = param_dict["connection_pen"]
        if "full_tree_pen" in param_dict:
            self.full_tree_pen = param_dict["full_tree_pen"]
        if "input_pen" in param_dict:
            self.input_pen = param_dict["input_pen"]
        if "batch_size" in param_dict:
            self.batch_size = param_dict["batch_size"]
        if "num_classes" in param_dict:
            self.num_classes = param_dict["num_classes"]
        if "max_iters" in param_dict:
            self.max_iters = param_dict["max_iters"]
        if "max_prox_iters" in param_dict:
            self.max_prox_iters = param_dict["max_prox_iters"]            
        if "weight" in param_dict:
            self.weight = param_dict["weight"]

        return self
