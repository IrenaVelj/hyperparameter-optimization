from optimize.optimizer import Optimizer

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    return x**2

if __name__ == "__main__":
    print("hello")
    optimizer = Optimizer(study_name=None,    #   Study’s name. If this argument is set to None, a unique name is generated automatically.
                            storage=None,
                            sampler=None,  #   A sampler object that implements background algorithm for value suggestion. If None is specified, TPESampler is used during single-objective optimization and NSGAIISampler during multi-objective optimization.
                            pruner=None,   #    A pruner object that decides early stopping of unpromising trials. If None is specified, MedianPruner is used as the default.
                            direction="minimize",    #   Direction of optimization. Set minimize for minimization and maximize for maximization. You can also pass the corresponding StudyDirection object.
                            load_if_exists=False,   #   Resume. Flag to control the behavior to handle a conflict of study names. In the case where a study named study_name already exists in the storage, a DuplicatedStudyError is raised if load_if_exists is set to False. Otherwise, the creation of the study is skipped, and the existing one is returned.
                            directions=None)

    optimizer.search(objective_fcn=objective, n_trials=10)