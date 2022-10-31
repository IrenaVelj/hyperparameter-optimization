# Neural Architecture Search

import torch
import optuna
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from optuna.trial import TrialState


input_size = 784
n_classes = 10
batch_size = 8
n_epochs = 10
device = torch.device("cuda")
N_TRAIN_EXAMPLES = batch_size*40
N_VALID_EXAMPLES = batch_size * 20

def get_data(batch_size):
    train_dataset = torchvision.datasets.MNIST(root = "./data", train = True, transform = transforms.ToTensor() ,download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./data", train = False, transform = transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size ,shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, test_loader

class NeuralNet(nn.Module):
    def __init__(self, trial, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    def forward (self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def define_model(trial, param):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    return NeuralNet(trial, input_size, param, n_classes)

class NeuralArchitectureOptimizer():
    def __init__(self, train_loader, test_loader, model_fcn, architecture_parameters) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_fcn = model_fcn
        self.architecture_parameters = architecture_parameters

    def create_architecture(self, trial):
        for architecture_param in self.architecture_parameters.keys():
            param = trial.suggest_int(architecture_param, self.architecture_parameters[architecture_param][0], self.architecture_parameters[architecture_param][1])
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
    
    nn_architecture_search = NeuralArchitectureOptimizer(train_loader, test_loader, define_model, architecture_parameters={'hidden_size': [90, 100]} )

    nn_architecture_search.study_and_optimize()
