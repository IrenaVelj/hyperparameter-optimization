import optuna
from typing import Union, Optional, Sequence, Callable, Tuple, Type, List


class Optimizer():
    def __init__(
        self,
        study_name: Optional[str] = None,    #   Study’s name. If this argument is set to None, a unique name is generated automatically.
        storage: Union[str, optuna.storages._base.BaseStorage, None] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,  #   A sampler object that implements background algorithm for value suggestion. If None is specified, TPESampler is used during single-objective optimization and NSGAIISampler during multi-objective optimization.
        pruner: Optional[optuna.pruners.BasePruner] = None,   #    A pruner object that decides early stopping of unpromising trials. If None is specified, MedianPruner is used as the default.
        direction: Optional[Union[str, optuna.study.StudyDirection]] = None,    #   Direction of optimization. Set minimize for minimization and maximize for maximization. You can also pass the corresponding StudyDirection object.
        load_if_exists: bool = False,   #   Resume. Flag to control the behavior to handle a conflict of study names. In the case where a study named study_name already exists in the storage, a DuplicatedStudyError is raised if load_if_exists is set to False. Otherwise, the creation of the study is skipped, and the existing one is returned.
        directions: Optional[Sequence[Union[str, optuna.study.StudyDirection]]] = None,  #   A sequence of directions during multi-objective optimization. direction and directions must not be specified at the same time
    ) -> None:
        self.study_name = study_name
        self.storage = storage
        self.sampler = sampler
        self.pruner = pruner
        self.direction = direction
        self.load_if_exists = load_if_exists
        self.directions = directions
        self.study = optuna.create_study(storage=self.storage, 
                                    sampler=self.sampler,
                                    pruner=self.pruner,
                                    study_name=self.study_name, 
                                    direction=self.direction, 
                                    load_if_exists=self.load_if_exists, 
                                    directions=self.directions)


    def search(self, 
                objective_fcn: Callable[[optuna.trial.Trial], Union[float, Sequence[float]]],
                n_trials: Optional[int] = None, 
                timeout: Union[None, float]= None, 
                n_jobs: int = 1, 
                catch: Tuple[Type[Exception], ...] = (), 
                callbacks: Optional[List[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]]] = None, 
                gc_after_trial: bool = False, 
                show_progress_bar: bool = False):

        return self.study.optimize(func=objective_fcn,
                                    n_trials=n_trials,
                                    timeout=timeout, 
                                    n_jobs=n_jobs,
                                    catch=catch, 
                                    callbacks=callbacks,
                                    gc_after_trial=gc_after_trial, 
                                    show_progress_bar=show_progress_bar)
