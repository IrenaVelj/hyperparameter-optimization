import torch
import optuna
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from optuna.trial import TrialState

batch_size = 8  # TODO Add batch size to be hyperparameter to be optimized
device = torch.device("cuda")
N_TRAIN_EXAMPLES = batch_size*40
N_VALID_EXAMPLES = batch_size * 20

class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, num_classes=10):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    def forward (self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

def define_model():
    return NeuralNet()

def get_data(batch_size):
    train_dataset = torchvision.datasets.MNIST(root = "./data", train = True, transform = transforms.ToTensor() ,download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./data", train = False, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size ,shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader

class HyperparameterOptimizier():
    def __init__(self, model, train_loader, test_loader, direction, hyperparameters, study_name="study-test", num_trials=20, timeout=300, n_epochs=22) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.direction = direction
        self.hyperparameters = hyperparameters
        self.study_name = study_name
        self.num_trials = num_trials
        self.timeout = timeout
        self.n_epochs = n_epochs

    def create_optimizer(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer", self.hyperparameters["optimizer"])
        
        if optimizer_name == "Adam":
            # Learning rate
            if "adam_lr" in self.hyperparameters.keys(): 
                # Specific range for this specific optimizer
                adam_lr = trial.suggest_float("lr", self.hyperparameters["adam_lr"][0], self.hyperparameters["adam_lr"][1])
            elif "lr" in self.hyperparameters.keys():
                # Generic range for all optimizers (user wants to use the same range for all optimizers)
                adam_lr = trial.suggest_float("lr", self.hyperparameters["lr"][0], self.hyperparameters["lr"][1])
            else:
                # If there is no range set by the user, we use the default one
                adam_lr = 1e-3
            # Betas - coefficient used for computing running avarages of gradient and its square
            if "adam_beta1" in self.hyperparameters.keys():
                adam_beta1 = trial.suggest_float("beta1", self.hyperparameters["adam_beta1"][0], self.hyperparameters["adam_beta1"][1])
            else:
                adam_beta1 = 0.9
            if "adam_beta2" in self.hyperparameters.keys():
                adam_beta2 = trial.suggest_float("beta2", self.hyperparameters["adam_beta2"][0], self.hyperparameters["adam_beta2"][1])
            else:
                adam_beta2 = 0.999
            # Weight decay (L2 penalty)
            if "adam_weight_decay" in self.hyperparameters.keys():
                adam_weight_decay = trial.suggest_float("weight_decay", self.hyperparameters["adam_weight_decay"][0], self.hyperparameters["adam_weight_decay"][1])
            else:
                adam_weight_decay = 0
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), adam_lr, (adam_beta1, adam_beta2), adam_weight_decay)

        elif optimizer_name == "AdamW":
            # Learning rate
            if "AdamW_lr" in self.hyperparameters.keys(): 
                # Specific range for this specific optimizer
                AdamW_lr = trial.suggest_float("lr", self.hyperparameters["AdamW_lr"][0], self.hyperparameters["AdamW_lr"][1])
            elif "lr" in self.hyperparameters.keys():
                # Generic range for all optimizers (user wants to use the same range for all optimizers)
                AdamW_lr = trial.suggest_float("lr", self.hyperparameters["lr"][0], self.hyperparameters["lr"][1])
            else:
                # If there is no range set by the user, we use the default one
                AdamW_lr = 1e-3
            # Betas - coefficient used for computing running avarages of gradient and its square
            if "AdamW_beta1" in self.hyperparameters.keys():
                AdamW_beta1 = trial.suggest_float("beta1", self.hyperparameters["AdamW_beta1"][0], self.hyperparameters["AdamW_beta1"][1])
            else:
                AdamW_beta1 = 0.9
            if "AdamW_beta2" in self.hyperparameters.keys():
                AdamW_beta2 = trial.suggest_float("beta2", self.hyperparameters["AdamW_beta2"][0], self.hyperparameters["AdamW_beta2"][1])
            else:
                AdamW_beta2 = 0.999
            # Weight decay (L2 penalty)
            if "AdamW_weight_decay" in self.hyperparameters.keys():
                AdamW_weight_decay = trial.suggest_float("weight_decay", self.hyperparameters["AdamW_weight_decay"][0], self.hyperparameters["AdamW_weight_decay"][1])
            else:
                AdamW_weight_decay = 0.01
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), AdamW_lr, (AdamW_beta1, AdamW_beta2), AdamW_weight_decay)

        elif optimizer_name == "SparseAdam":
            #   Implements lazy version of Adam algorithm suitable for sparse tensors.
            #   In this variant, only moments that show up in the gradient get updated, and only those portions of the gradient get applied to the parameters.

            # Learning rate
            if "SparseAdam_lr" in self.hyperparameters.keys(): 
                # Specific range for this specific optimizer
                SparseAdam_lr = trial.suggest_float("lr", self.hyperparameters["SparseAdam_lr"][0], self.hyperparameters["SparseAdam_lr"][1])
            elif "lr" in self.hyperparameters.keys():
                # Generic range for all optimizers (user wants to use the same range for all optimizers)
                SparseAdam_lr = trial.suggest_float("lr", self.hyperparameters["lr"][0], self.hyperparameters["lr"][1])
            else:
                # If there is no range set by the user, we use the default one
                SparseAdam_lr = 1e-3
            # Betas - coefficient used for computing running avarages of gradient and its square
            if "SparseAdam_beta1" in self.hyperparameters.keys():
                SparseAdam_beta1 = trial.suggest_float("beta1", self.hyperparameters["SparseAdam_beta1"][0], self.hyperparameters["SparseAdam_beta1"][1])
            else:
                SparseAdam_beta1 = 0.9
            if "SparseAdam_beta2" in self.hyperparameters.keys():
                SparseAdam_beta2 = trial.suggest_float("beta2", self.hyperparameters["SparseAdam_beta2"][0], self.hyperparameters["SparseAdam_beta2"][1])
            else:
                SparseAdam_beta2 = 0.999
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), SparseAdam_lr, (SparseAdam_beta1, SparseAdam_beta2))

        elif optimizer_name == "Adadelta":
            if "Adadelta_lr" in self.hyperparameters.keys():
                Adadelta_lr = trial.suggest_float("lr", self.hyperparameters["Adadelta_lr"][0], self.hyperparameters["Adadelta_lr"][1])
            elif "lr" in self.hyperparameters.keys():
                Adadelta_lr = trial.suggest_float("lr", self.hyperparameters["lr"][0], self.hyperparameters["lr"][1])
            else:
                Adadelta_lr = 1.0
            if "Adadelta_rho" in self.hyperparameters.keys():
                Adadelta_rho = trial.suggest_float("rho", self.hyperparameters["Adadelta_rho"][0], self.hyperparameters["Adadelta_rho"][1])
            else:
                Adadelta_rho = 0.9
            # Weight decay (L2 penalty)
            if "Adadelta_weight_decay" in self.hyperparameters.keys():
                Adadelta_weight_decay = trial.suggest_float("weight_decay", self.hyperparameters["Adadelta_weight_decay"][0], self.hyperparameters["Adadelta_weight_decay"][1])
            else:
                Adadelta_weight_decay = 0
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), Adadelta_lr, Adadelta_rho, Adadelta_weight_decay)

        elif optimizer_name == "Adagrad":
            if "Adagrad_lr" in self.hyperparameters.keys():
                Adagrad_lr = trial.suggest_float("lr", self.hyperparameters["Adagrad_lr"][0], self.hyperparameters["Adagrad_lr"][1])
            elif "lr" in self.hyperparameters.keys():
                Adagrad_lr = trial.suggest_float("lr", self.hyperparameters["lr"][0], self.hyperparameters["lr"][1])
            else:
                Adagrad_lr = 0.01
            if "Adagrad_lr_decay" in self.hyperparameters.keys():
                Adagrad_lr_decay = trial.suggest_float("lr_decay", self.hyperparameters["Adagrad_lr_decay"][0], self.hyperparameters["Adagrad_lr_decay"][1])
            else:
                Adagrad_lr_decay = 0
            if "Adagrad_weight_decay" in self.hyperparameters.keys():
                Adagrad_weight_decay = trial.suggest_float("weight_decay", self.hyperparameters["Adagrad_weight_decay"][0], self.hyperparameters["Adagrad_weight_decay"][1])
            else:
                Adagrad_weight_decay = 0
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), Adagrad_lr, Adagrad_lr_decay, Adagrad_weight_decay)

        elif optimizer_name == "Adamax":
            if "Adamax_lr" in self.hyperparameters.keys():
                Adamax_lr = trial.suggest_float("lr", self.hyperparameters["Adamax_lr"][0], self.hyperparameters["Adamax_lr"][1])
            elif "lr" in self.hyperparameters.keys():
                Adamax_lr = trial.suggest_float("lr", self.hyperparameters["lr"][0], self.hyperparameters["lr"][1])
            else:
                Adamax_lr = 0.002
            if "Adamax_weight_decay" in self.hyperparameters.keys():
                Adamax_weight_decay = trial.suggest_float("weight_decay", self.hyperparameters["Adamax_weight_decay"][0], self.hyperparameters["Adamax_weight_decay"][1])
            else:
                Adamax_weight_decay = 0
            # Betas - coefficient used for computing running avarages of gradient and its square
            if "Adamax_beta1" in self.hyperparameters.keys():
                Adamax_beta1 = trial.suggest_float("beta1", self.hyperparameters["Adamax_beta1"][0], self.hyperparameters["Adamax_beta1"][1])
            else:
                Adamax_beta1 = 0.9
            if "Adamax_beta2" in self.hyperparameters.keys():
                Adamax_beta2 = trial.suggest_float("beta2", self.hyperparameters["Adamax_beta2"][0], self.hyperparameters["Adamax_beta2"][1])
            else:
                Adamax_beta2 = 0.999
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), Adamax_lr, (Adamax_beta1, Adamax_beta2), Adamax_weight_decay)

        elif optimizer_name == "RMSprop":
            if "RMSprop_lr" in self.hyperparameters.keys():
                RMSprop_lr = trial.suggest_float("lr", self.hyperparameters["RMSprop_lr"][0], self.hyperparameters["RMSprop_lr"][1])
            elif "lr" in self.hyperparameters.keys():
                RMSprop_lr = trial.suggest_float("lr", self.hyperparameters["lr"][0], self.hyperparameters["lr"][1])
            else:
                RMSprop_lr = 1e-3
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), RMSprop_lr)

        elif optimizer_name == "SGD":
            if "SGD_lr" in self.hyperparameters.keys():
                SGD_lr = trial.suggest_float("lr", self.hyperparameters["SGD_lr"][0], self.hyperparameters["SGD_lr"][1])
            elif "lr" in self.hyperparameters.keys():
                SGD_lr = trial.suggest_float("lr", self.hyperparameters["lr"][0], self.hyperparameters["lr"][1])
            else: 
                SGD_lr = 1e-1
            if "SGD_momentum" in self.hyperparameters.keys():
                SGD_momentum = trial.suggest_float("momentum", self.hyperparameters["SGD_momentum"][0], self.hyperparameters["SGD_momentum"][1])
            elif "momentum" in self.hyperparameters.keys():
                SGD_momentum = trial.suggest_float("momentum", self.hyperparameters["momentum"][0], self.hyperparameters["momentum"][1])
            else:
                SGD_momentum = 0
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), SGD_lr, SGD_momentum)

        else:
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), 1e-1)

        return optimizer

    def objective(self, trial):
        criterion =  nn.CrossEntropyLoss()

        optimizer = self.create_optimizer(trial)
        
        # Training loop
        if "n_epochs" in self.hyperparameters.keys():
            n_epochs = trial.suggest_int("n_epochs", self.hyperparameters["n_epochs"][0], self.hyperparameters["n_epochs"][1])
        else:
            n_epochs = self.n_epochs

        for epoch in range (n_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # Limiting training data for faster epochs.
                if i * batch_size >= N_TRAIN_EXAMPLES:
                    break

                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)

                #forward
                outputs = self.model(images)
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
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1) #returns values of max and its indes // WE ARE INTERESTED IN INDICES
                n_samples +=  labels.shape[0]
                n_correct += (predictions == labels).sum().item()
            acc = 100 * n_correct/n_samples

        return acc

    def study_and_optimize(self):
        study = optuna.create_study(direction=self.direction, study_name=self.study_name)
        study.optimize(func=self.objective, n_trials=self.num_trials, timeout=self.timeout)

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
    hyperparameter_optimizier = HyperparameterOptimizier(define_model(), 
                                                        train_loader, 
                                                        test_loader, 
                                                        direction = "maximize",
                                                        hyperparameters={"optimizer": ['Adam', 'RMSprop', 'SGD', 'Adadelta', 'AdamW', 'Adamax'],
                                                                        "lr": [1e-5, 1e-1],
                                                                        "RMSprop_lr": [1e-4,1e-2],
                                                                        "SGD_lr": [1e-4, 1e-1],
                                                                        "SGD_momentum": [0.9, 0.99],
                                                                        "adam_beta1": [0.89, 0.91],
                                                                        "adam_beta2": [0.990, 0.999],
                                                                        "AdamW_beta1": [0.89, 0.91],
                                                                        "AdamW_beta2": [0.990, 0.999],
                                                                        "Adamax_beta1": [0.895, 0.910],
                                                                        "Adamax_beta2": [0.995, 0.998],
                                                                        "n_epochs": [10, 20] 
                                                                        },
                                                        study_name = "nnlo-hyperopt-test",
                                                        num_trials = 30,
                                                        timeout = 600)

    hyperparameter_optimizier.study_and_optimize()

    
