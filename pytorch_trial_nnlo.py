import torch
import optuna
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from optuna.trial import TrialState


input_size = 784
hidden_size = 100
n_classes = 10
batch_size = 8
n_epochs = 10
device = torch.device("cuda")
N_TRAIN_EXAMPLES = batch_size*40
N_VALID_EXAMPLES = batch_size * 20

class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
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
    return NeuralNet(input_size, hidden_size, n_classes)

def get_data(batch_size):
    train_dataset = torchvision.datasets.MNIST(root = "./data", train = True, transform = transforms.ToTensor() ,download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./data", train = False, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size ,shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader

class HyperparameterOptimizier():
    def __init__(self, model, train_loader, test_loader, hyperparameters) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.hyperparameters = hyperparameters

    def create_optimizer(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer", self.hyperparameters["optimizer"])

        if optimizer_name == "Adam":
            if "adam_lr" in self.hyperparameters.keys(): 
                # Specific range for this specific optimizer
                adam_lr = trial.suggest_float("lr", self.hyperparameters["adam_lr"][0], self.hyperparameters["adam_lr"][1])
            elif "lr" in self.hyperparameters.keys():
                # Generic range for all optimizers (user wants to use the same range for all optimizers)
                adam_lr = trial.suggest_float("lr", self.hyperparameters["lr"][0], self.hyperparameters["lr"][1])
            else:
                # If there is no range set by the user
                adam_lr = 1e-3
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), adam_lr)
        elif optimizer_name == "RMSprop":
            if "RMSprop_lr" in self.hyperparameters.keys():
                RMSprop_lr = trial.suggest_float("lr", self.hyperparameters["RMSprop_lr"][0], self.hyperparameters["RMSprop_lr"][1])
            else:
                RMSprop_lr = 1e-3
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), RMSprop_lr)

        else:
            optimizer = getattr(optim, optimizer_name)(self.model.parameters(), 1e-1)

        return optimizer

    def objective(self, trial):
        criterion =  nn.CrossEntropyLoss()
        # optimizer_name = trial.suggest_categorical("optimizer", self.hyperparameters["optimizer"])
        # # optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=1e-4) 
        # learning_rate = trial.suggest_float("learning_rate", self.hyperparameters["learning_rate"][0], self.hyperparameters["learning_rate"][1])
        optimizer = self.create_optimizer(trial)
        
        # Training loop
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
        study = optuna.create_study(direction="maximize", study_name="pytorch-trial-nnlo")
        study.optimize(self.objective, n_trials=30, timeout=600)

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
                                                        hyperparameters={"optimizer": ['Adam', 'RMSprop', 'SGD'],
                                                                        "lr": [1e-5, 1e-1],
                                                                        "RMSprop_lr": [1e-4,1e-2]})

    # nn_architecture_search = ... TODO: Make a class for NAS

    hyperparameter_optimizier.study_and_optimize()

    
