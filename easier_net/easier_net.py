from ensurepip import bootstrap
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


import pickle
import sier_net 
from sklearn.base import clone


class EasierNetEstimator:
    """ This is the Ensembling by Averaging Sparse-Input Hierarchical Networks(EASIER-NET)"""
    def __init__(
        self,
        num_classes=0, #Equal to 0 if doing a regression, otherwise number of classes for binary / multi-classification
        input_filter_layer=True, #Scales the inputs by parameter β
        n_layers=2, #Number of hidden layers
        n_hidden=10, #Number of hidden nodes in each layer
        n_estimators=5, #Size of ensemble; number of SIER-nets being ensembled.
        max_iters=40, #Number of epochs to run Adam     
        max_prox_iters=0, #Number of epochs to run batch proximal gradient descent
        input_pen=0, #λ1; Controls the input sparsity         
        full_tree_pen=0.001, #λ2; controls the number of active layers and hidden nodes 
        num_batches=3, #number of mini-batches for Adam
        # n_jobs=16, #TODO: ..
        model_fit_params_file=None, #A json file that specifies what the hyperparameters are. If given, this will override the arguments passed in.
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
        self.num_batches=num_batches
        # self.n_jobs=n_jobs
        self.model_fit_params_file=model_fit_params_file
        self.score_criterion = (
            nn.CrossEntropyLoss() if self.num_classes >= 2 else nn.MSELoss()
        )       
        if self.model_fit_params_file is not None:
            self.estimators = self.load_model(model_fit_params_file) 
        else:
            self.estimators=self._generate_estimators()

        assert num_classes != 1
        assert input_filter_layer
    
    def _generate_estimators(self):
        estimators=[]
        for est in range(self.n_estimators): #TODO: i -> est?
            est=sier_net.SierNetEstimator(
                # random_state=i,
                num_classes=self.num_classes,
                input_filter_layer=self.input_filter_layer,
                n_hidden= self.n_hidden,
                # n_estimators=self.n_estimators,
                max_iters=self.max_iters,
                max_prox_iters=self.max_prox_iters,
                full_tree_pen=self.full_tree_pen,
                input_pen=self.input_pen,
                # num_batches=self.num_batches,
                # n_jobs=self.n_jobs,
            ) 
            
            estimators.append(est)

        return estimators

            #TODO: fix? this import sier_net -> from sier_net import SierNetEstimator()


    def write_model(self, filename): #save and write model to file
        #from fit_easier_net
        meta_state_dict = self.estimators[0].get_params()
        meta_state_dict["state_dicts"] = [
            est.net.state_dict() for est in self.estimators
        ]
        torch.save(meta_state_dict, filename)


#TODO: delete this old fx 
    # def load_model(self):
    #     estimators=sier_net.SierNetEstimator()
    #     #definitely wrong, need to load for each state_dicts
    #     estimators.load_state_dict(torch.load(self.model_fit_params_file, map_location='cpu'), strict=False) #to avoid gpu ram surge
 #TODO: new - still need to fix a bit

    def load_model(self, model_fit_params_file):
        estimators = []
        state_dict_list = torch.load(model_fit_params_file, map_location='cpu') #to avoid gpu ram surge
        for state_dict in state_dict_list:
            sier_net = sier_net.SierNetEstimator(
                num_classes=self.num_classes,
                input_filter_layer=self.input_filter_layer,
                n_hidden= self.n_hidden,
                # n_estimators=self.n_estimators,
                max_iters=self.max_iters,
                max_prox_iters=self.max_prox_iters,
                full_tree_pen=self.full_tree_pen,
                input_pen=self.input_pen,
                # num_batches=self.num_batches,
                # n_jobs=self.n_jobs,
            )
            sier_net.load_state_dict(state_dict)
            estimators.append(sier_net)
        return estimators

        #loading json way
        # estimators = []
        # for i in range(self.n_estimators):
        #     filename = 'models/model_' + str(i) + '.json'
        #     est = model_from_json(self.model_fit_params_file)
        #     estimators.append(est)
        #     print(f"[INFO]>>loaded {filename}.")
        # return estimators


    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ):

        for est in self.estimators:
            est.fit(x, y)
       
        return self 

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        self.fit(x,y)
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
            "num_batches": self.num_batches,
            # "n_jobs": self.n_jobs,
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
        if "num_batches" in param_dict:
            self.num_batches = param_dict["num_batches"]  
        # if "n_jobs" in param_dict:
        #     self.n_jobs = param_dict["n_jobs"]      

        return self
