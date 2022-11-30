from optimize.optimizer import Optimizer

def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    return x**2

if __name__ == "__main__":
    print("hello")
    optimizer = Optimizer(study_name=None,
                            storage=None,
                            sampler=None, 
                            pruner=None,  
                            direction="minimize",   
                            load_if_exists=False,   
                            directions=None)

    optimizer.search(objective_fcn=objective, n_trials=10)