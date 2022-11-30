import enum
from typing import Union, Sequence
import torch
from torch import optim

device = torch.device("cuda")

class HyperparameterType(enum.Enum):
    """
    Class which defines a type of a hyperparameter to be tuned.
    """
    FLOAT = 1,
    INTEGER = 2,
    CATEGORICAL = 3
    BOOLEAN = 4

class Hyperparameter():
    def __init__(self, name: str, hp_type: HyperparameterType, min: Union[None, int, float] =0, max: Union[None, int, float] =0, categories: Sequence[Union[None, bool, int, float, str]] =None) -> None:
        self.name = name
        self.type = hp_type
        self.min = min
        self.max = max
        self.categories = categories
    
    def suggest_value(self, trial):
        if self.type == HyperparameterType.FLOAT:
            return trial.suggest_float(self.name, self.min, self.max)
        if self.type == HyperparameterType.INTEGER:
            return trial.suggest_int(self.name, self.min, self.max)
        if self.type == HyperparameterType.CATEGORICAL:
            return trial.suggest_categorical(self.name, self.categories)
        if self.type == HyperparameterType.BOOLEAN:
            return trial.suggest_categorical(self.name, [True, False])


class PytorchHyperparameterOptimizier():
    """
    Class conserning neural network optimizers in PyTorch, as hyperparameter to be optimized during the process.
    """
    def __init__(self, model, hyperparameters) -> None:
        self.model = model.to(device)
        self.hyperparameters = hyperparameters

    def create_optimizer(self, trial):
        """
        Returns pytorch optimizer object.
        """
        optimizer_name = trial.suggest_categorical("optimizer", self.hyperparameters["optimizer"])
        
        if optimizer_name == "Adam":
            opt_params = {}
            betas = [0,0]
            for hp in self.hyperparameters["optimizer"]["Adam"]:
                if hp.type == HyperparameterType.FLOAT and hp.name == 'beta1':
                    betas[0] = hp.suggest_value(trial)
                elif hp.type == HyperparameterType.FLOAT and hp.name == 'beta2':
                    betas[1] = hp.suggest_value(trial)
                else:
                    opt_params[hp.name] = hp.suggest_value(trial)

            if betas[0] == 0:
                betas[0] = 0.9
            elif betas[1] == 0:
                betas[1] = 0.999
            else:
                opt_params['betas'] = tuple(betas)

            print(betas)
            
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), **opt_params) #TODO See how to pass parameters, kwargs

        if optimizer_name == "AdamW":
            opt_params = {}
            betas = [0,0]
            for hp in self.hyperparameters["optimizer"]["AdamW"]:
                if hp.type == HyperparameterType.FLOAT and hp.name == 'beta1':
                    betas[0] = hp.suggest_value(trial)
                elif hp.type == HyperparameterType.FLOAT and hp.name == 'beta2':
                    betas[1] = hp.suggest_value(trial)
                else:
                    opt_params[hp.name] = hp.suggest_value(trial)

            if betas[0] == 0:
                betas[0] = 0.9
            elif betas[1] == 0:
                betas[1] = 0.999
            else:
                opt_params['betas'] = tuple(betas)
            
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), **opt_params)

        if optimizer_name == "SparseAdam":
            #   Implements lazy version of Adam algorithm suitable for sparse tensors.
            #   In this variant, only moments that show up in the gradient get updated, and only those portions of the gradient get applied to the parameters.

            opt_params = {}
            betas = [0,0]
            for hp in self.hyperparameters["optimizer"]["SparseAdam"]:
                if hp.type == HyperparameterType.FLOAT and hp.name == 'beta1':
                    betas[0] = hp.suggest_value(trial)
                elif hp.type == HyperparameterType.FLOAT and hp.name == 'beta2':
                    betas[1] = hp.suggest_value(trial)
                else:
                    opt_params[hp.name] = hp.suggest_value(trial)

            if betas[0] == 0:
                betas[0] = 0.9
            elif betas[1] == 0:
                betas[1] = 0.999
            else:
                opt_params['betas'] = tuple(betas)
            
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), **opt_params)

        if optimizer_name == "Adamax":
            opt_params = {}
            betas = [0,0]
            for hp in self.hyperparameters["optimizer"]["Adamax"]:
                if hp.type == HyperparameterType.FLOAT and hp.name == 'beta1':
                    betas[0] = hp.suggest_value(trial)
                elif hp.type == HyperparameterType.FLOAT and hp.name == 'beta2':
                    betas[1] = hp.suggest_value(trial)
                else:
                    opt_params[hp.name] = hp.suggest_value(trial)

            if betas[0] == 0:
                betas[0] = 0.9
            elif betas[1] == 0:
                betas[1] = 0.999
            else:
                opt_params['betas'] = tuple(betas)
            
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), **opt_params)

        if optimizer_name == "RMSprop":
            opt_params = {}
            for hp in self.hyperparameters["optimizer"]["RMSprop"]:
                opt_params[hp.name] = hp.suggest_value(trial)
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), **opt_params)

        if optimizer_name == "Adadelta":
            opt_params = {}
            for hp in self.hyperparameters["optimizer"]["Adadelta"]:
                opt_params[hp.name] = hp.suggest_value(trial)
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), **opt_params)

        if optimizer_name == "Adagrad":
            opt_params = {}
            for hp in self.hyperparameters["optimizer"]["Adagrad"]:
                opt_params[hp.name] = hp.suggest_value(trial)
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), **opt_params)

        if optimizer_name == "SGD":
            opt_params = {}
            for hp in self.hyperparameters["optimizer"]["SGD"]:
                opt_params[hp.name] = hp.suggest_value(trial)
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), **opt_params)

        return optimizer