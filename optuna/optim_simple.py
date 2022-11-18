"""A simple optimization problem:
    Define objective function to be optimized. Let's minimize (x - 2)^2
    Suggest hyperparameter values using trial object. Here, a float value of x is suggested from -10 to 10
    Create a study object and invoke the optimize method over 100 trials
"""

import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}