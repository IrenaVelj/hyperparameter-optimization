# Neural Architecture Search

import torch
import optuna
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from optuna.trial import TrialState


#TODO: Eliminate as much as we could from hardcoded variables/layers
input_size = 784
CLASSES = 10
batch_size = 8
n_epochs = 50
device = torch.device("cuda")
N_TRAIN_EXAMPLES = batch_size*40
N_VALID_EXAMPLES = batch_size * 20

# Activation functions map
# Activations that exist in Torch https://pytorch.org/docs/stable/nn.html
activation_map =    {'Threshold': nn.Threshold(0.1, 20), # TODO: Think how to integrate activation functions which have additional parameters to be passed, like here.
                    'ReLU': nn.ReLU(),
                    'RReLU': nn.RReLU(),
                    'Hardtanh': nn.Hardtanh(),
                    'ReLU6': nn.ReLU6(),
                    'Sigmoid': nn.Sigmoid(),
                    'Hardsigmoid': nn.Hardsigmoid(),
                    'Tanh': nn.Tanh(),
                    'SiLU': nn.SiLU(),
                    'Mish': nn.Mish(),
                    'Hardswish': nn.Hardswish(),
                    'ELU': nn.ELU(),
                    'CELU': nn.CELU(),
                    'SELU': nn.SELU(),
                    'GLU': nn.GLU(),
                    'GELU': nn.GELU(),
                    'Hardshrink': nn.Hardshrink(),
                    'LeakyReLU': nn.LeakyReLU(),
                    'LogSigmoid': nn.LogSigmoid(),
                    'Softplus': nn.Softplus(),
                    'Softshrink': nn.Softshrink(),
                    'MultiheadAttention': nn.MultiheadAttention(1000, 4),   # TODO: Think how to integrate activation functions which have additional parameters to be passed, like here.
                    'PReLU': nn.PReLU(),
                    'Softsign': nn.Softsign(),
                    'Tanhshrink': nn.Tanhshrink(),
                    'Softmin': nn.Softmin(),
                    'Softmax': nn.Softmax(),
                    'Softmax2d': nn.Softmax2d(),
                    'LogSoftmax': nn.LogSoftmax()
                }

#TODO: Take batch size from user
def get_data(batch_size):
    train_dataset = torchvision.datasets.MNIST(root = "./data", train = True, transform = transforms.ToTensor() ,download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./data", train = False, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size ,shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader



def define_model(trial, param):
    """ 
    Model builder. It builds a model for each different set of samples varibles.
    Params:
        - Param => dict. It contains the sampled variables from optuna.

    """
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = param['n_layers']
    layers = []
    in_features = 28 * 28 #TODO: Take from as input from user
    for i in range(n_layers):
        out_features = param['hidden_size_{}'.format(i+1)]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation_map[param['activation']])
        p = param['dropout_{}'.format(i+1)]
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

class NeuralArchitectureOptimizer():
    def __init__(self, train_loader, test_loader, model_fcn, architecture_parameters) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_fcn = model_fcn
        self.architecture_parameters = architecture_parameters

    def create_architecture(self, trial):
        param = {}
        #TODO: Try to find a more optimal solution to float/int/categorical 
        for architecture_param in self.architecture_parameters.keys():
            if(type(architecture_parameters[architecture_param]) is list):
                if (type(architecture_parameters[architecture_param][0]) is int):
                    param[architecture_param] = trial.suggest_int(architecture_param, self.architecture_parameters[architecture_param][0], self.architecture_parameters[architecture_param][1])
                elif (type(architecture_parameters[architecture_param][0]) is float):
                    param[architecture_param] = trial.suggest_float(architecture_param, self.architecture_parameters[architecture_param][0], self.architecture_parameters[architecture_param][1])
                else:
                    param[architecture_param] = trial.suggest_categorical(architecture_param, self.architecture_parameters[architecture_param])
            else:
                param[architecture_param] = architecture_parameters[architecture_param]

        model = self.model_fcn(trial, param)
        return model
        
    def objective(self, trial):
        criterion =  nn.CrossEntropyLoss()

        model = self.create_architecture(trial).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Training loop
        for epoch in range (n_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # Limiting training data for faster epochs.
                if i * batch_size >= N_TRAIN_EXAMPLES:
                    break

                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)

                #forward
                outputs = model(images)
                loss = criterion(outputs, labels)

                #backwards
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for i, (images, labels) in enumerate(self.test_loader):
                # Limiting validation data.
                if i * batch_size >= N_VALID_EXAMPLES:
                    break
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1) #returns values of max and its indes // WE ARE INTERESTED IN INDICES
                n_samples +=  labels.shape[0]
                n_correct += (predictions == labels).sum().item()
            acc = 100 * n_correct/n_samples

        return acc

    def study_and_optimize(self):
        study = optuna.create_study(direction="maximize", study_name="nas-trial-nnlo")
        study.optimize(self.objective, n_trials=150, timeout=600)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


if __name__ == "__main__":

    train_loader, test_loader = get_data(batch_size)
    architecture_parameters = {'hidden_size_1': [10, 20],
                                'hidden_size_2': 5,
                                'hidden_size_3': [10, 20],
                                'hidden_size_4': [10, 20],
                                'dropout_1': [0.1,0.5],
                                'dropout_2': 0.3,
                                'dropout_3': [0.1,0.5],
                                'dropout_4': [0.1,0.5],
                                'activation': ['ReLU6', 'SELU'],
                                'n_layers': [1, 4]
                                }

    nn_architecture_search = NeuralArchitectureOptimizer(train_loader, test_loader, define_model, architecture_parameters)
    nn_architecture_search.study_and_optimize()
