from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


import pickle
import sier_net 


class EasierNetEstimator:
    """ This is the Ensembling by Averaging Sparse-Input Hierarchical Networks(EASIER-NET)"""
    def __init__(
        self,
        n_estimators: int, #Size of ensemble; number of SIER-nets being ensembled.
        input_filter_layer: bool, #Scales the inputs by parameter β
        n_layers: int, #Number of hidden layers
        n_hidden: int, #Number of hidden nodes in each layer
        full_tree_pen: float, #λ2; controls the number of active layers and hidden nodes 
        input_pen: float, #λ1; Controls the input sparsity         
        batch_size: int, #size of mini-batches for Adam
        num_classes: int, #Equal to 0 if doing a regression, otherwise number of classes for binary / multi-classification
        weight: list,
        max_iters: int, #Number of epochs to run Adam     
        max_prox_iters: int, #Number of epochs to run batch proximal gradient descent
        model_fit_params_file: str =None, #A json file that specifies what the hyperparameters are. If given, this will override the arguments passed in.
    ):

        self.num_classes=num_classes
        self.input_filter_layer=input_filter_layer
        self.n_layers=n_layers
        self.n_hidden=n_hidden
        self.n_estimators=n_estimators
        self.max_iters=max_iters
        self.max_prox_iters=max_prox_iters
        self.full_tree_pen=full_tree_pen
        self.input_pen=input_pen
        self.batch_size=batch_size
        self.model_fit_params_file=model_fit_params_file
        self.score_criterion = (
            nn.CrossEntropyLoss() if self.num_classes >= 2 else nn.MSELoss()
        )   
        self.weight = weight

        self.estimators = self._generate_estimators()    
        if self.model_fit_params_file is not None:
            self.load_model(model_fit_params_file)
            

        assert num_classes != 1
        assert input_filter_layer
    
    def _generate_estimators(self) -> List[sier_net.SierNetEstimator]:
        estimators=[]
        for i in range(self.n_estimators):
            est=sier_net.SierNetEstimator(
                input_filter_layer=self.input_filter_layer,
                n_layers= self.n_layers,
                n_hidden= self.n_hidden,
                full_tree_pen=self.full_tree_pen,
                input_pen=self.input_pen,
                batch_size=self.batch_size,
                num_classes=self.num_classes,
                weight=self.weight,
                max_iters=self.max_iters,
                max_prox_iters=self.max_prox_iters,
            ) 
            
            estimators.append(est)

        return estimators


    def write_model(self, filename): #save and write model to file
        #save as .pt file
        meta_state_dict = self.get_params()
        meta_state_dict["state_dicts"] = [
            est.net.state_dict() for est in self.estimators
        ]
        for state_dict in meta_state_dict["state_dicts"]:
            state_dict["n_inputs"] = self.n_inputs

        torch.save(meta_state_dict, filename)

    def load_model(self, model_fit_params_file):
        meta_state_dict = torch.load(model_fit_params_file, map_location='cpu') #to avoid gpu ram surge
        state_dict_list = meta_state_dict["state_dicts"]
        for (sier_net, state_dict) in zip(self.estimators, state_dict_list):
            sier_net._load_state_dict(state_dict)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ):
        self.n_inputs=x.shape[1]
        for est in self.estimators:
            est.fit(x, y)
       
        return self 

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        nparr_to_torch = lambda nparr: (
            torch.Tensor(nparr)
            if self.num_classes == 0
            else torch.from_numpy(nparr.astype(int))[:, 0]
        )
        torch_y_pred = nparr_to_torch(y_pred)
        torch_y = nparr_to_torch(y)

        return -self.score_criterion(torch_y_pred, torch_y).item()

    def predict(self, x: np.ndarray) -> np.ndarray: 
        all_outputs=[]
        for est in self.estimators:
            output = est.net(torch.Tensor(x)).detach().numpy()
            all_outputs.append(output)
              
        final_output_regression = np.mean(all_outputs, axis=0)
        if self.num_classes == 0: #if this is regression, return avg
            return final_output_regression
        _, final_output_class = torch.max(final_output_regression, dim = 1) #max_indices give class label
        return final_output_class
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("return probabilities here please")


    def get_params(self, deep=False) -> dict:
        return{
            "num_classes": self.num_classes,
            "input_filter_layer": self.input_filter_layer,
            "n_layers": self.n_layers,
            "n_hidden": self.n_hidden,
            "n_estimators": self.n_estimators,
            "input_pen": self.input_pen,
            "full_tree_pen": self.full_tree_pen,
            "max_iters": self.max_iters,
            "max_prox_iters": self.max_prox_iters,
            "batch_size": self.batch_size,
        }

    def set_params(self, **param_dict):
        if "num_classes" in param_dict:
            self.num_classes = param_dict["num_classes"]
        if "input_filter_layer" in param_dict:
            self.input_filter_layer = param_dict["input_filter_layer"]
        if "n_layers" in param_dict:
            self.n_layers = param_dict["n_layers"]
        if "n_hidden" in param_dict:
            self.n_hidden = param_dict["n_hidden"]
        if "n_estimators" in param_dict:
            self.n_estimators = param_dict["n_estimators"]
        if "input_pen" in param_dict:
            self.input_pen = param_dict["input_pen"]
        if "full_tree_pen" in param_dict:
            self.full_tree_pen = param_dict["full_tree_pen"]
        if "max_iters" in param_dict:
            self.max_iters = param_dict["max_iters"]
        if "max_prox_iters" in param_dict:
            self.max_prox_iters = param_dict["max_prox_iters"]             
        if "batch_size" in param_dict:
            self.batch_size = param_dict["batch_size"]       

        return self
