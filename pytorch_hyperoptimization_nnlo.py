import torch
import optuna
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import enum
from optuna.trial import TrialState

batch_size = 8  # TODO Add batch size to be hyperparameter to be optimized
device = torch.device("cuda")
N_TRAIN_EXAMPLES = batch_size*40
N_VALID_EXAMPLES = batch_size * 20

class HyperparameterType(enum.Enum):
    """
    Class which defines a type of a hyperparameter to be tuned.
    """
    FLOAT = 1,
    INTEGER = 2,
    CATEGORICAL = 3
    BOOLEAN = 4

class Hyperparameter():
    def __init__(self, name, hp_type, min=0, max=0, categories=None) -> None:
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

    def objective(self, trial):
        criterion =  nn.CrossEntropyLoss()

        optimizer = self.create_optimizer(trial)
        
        # Training loop
        if "n_epochs" in self.hyperparameters.keys():
            n_epochs = self.hyperparameters["n_epochs"].suggest_value(trial)
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
        study = optuna.create_study(storage=None,
                                    sampler=None,
                                    pruner=None,
                                    study_name=self.study_name,
                                    direction=self.direction,
                                    load_if_exists=False,
                                    directions=None)

        study.optimize(func=self.objective,
                        n_trials=self.num_trials, 
                        timeout=self.timeout,
                        n_jobs=1,
                        catch=(),
                        callbacks=None,
                        gc_after_trial=False,
                        show_progress_bar=True  # Fancy progress bar :D
                        )


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
                                                        hyperparameters={"optimizer": {
                                                                                        'Adam': [Hyperparameter('lr', HyperparameterType.FLOAT, min=1e-4, max=1e-3), 
                                                                                                Hyperparameter('beta1', HyperparameterType.FLOAT, min=0.88, max = 0.89),
                                                                                                Hyperparameter('beta2', HyperparameterType.FLOAT, min=0.991, max = 0.993),
                                                                                                Hyperparameter('amsgrad', HyperparameterType.BOOLEAN)],
                                                                                       'RMSprop': [Hyperparameter('lr', HyperparameterType.FLOAT, min=1e-4, max=1e-3),
                                                                                                   Hyperparameter('alpha', HyperparameterType.FLOAT, min=0.97, max=0.99)],
                                                                                        'Adamax': [Hyperparameter('lr', HyperparameterType.FLOAT, min=1e-4, max=1e-3),
                                                                                                    Hyperparameter('beta1', HyperparameterType.FLOAT, min=0.88, max = 0.89),
                                                                                                    Hyperparameter('beta2', HyperparameterType.FLOAT, min=0.991, max = 0.993)],
                                                                                        'SGD': [Hyperparameter('lr', HyperparameterType.FLOAT, min=1e-4, max=1e-3)]
                                                                                        },
                                                                        "n_epochs": Hyperparameter('n_epochs', HyperparameterType.INTEGER, min=15, max=20)
                                                                        },
                                                        study_name = "nnlo-hyperopt-test",
                                                        num_trials = 50,
                                                        timeout = 600)

    hyperparameter_optimizier.study_and_optimize()


