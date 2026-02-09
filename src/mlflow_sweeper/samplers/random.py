import itertools
import logging
from typing import Sequence

from filelock import FileLock
from optuna import Study
from optuna.samplers import RandomSampler as BaseRandomSampler
from optuna.trial import FrozenTrial, TrialState

from mlflow_sweeper.optimize import PREEMPTED_KEY, PREEMPTION_REASON_KEY, PREEMPTION_REASON_NO_MORE_TRIALS


RETRY_COUNT_KEY = "retry_count"


logger = logging.getLogger(__name__)


class RandomSampler(BaseRandomSampler):
    """Random sampler with optional n_runs cap, grid_params expansion, and retry logic."""

    def __init__(
        self,
        n_runs: int | None = None,
        seed: int | None = None,
        max_retry_count: int = 3,
        grid_search_space: dict[str, list] | None = None,
    ) -> None:
        super().__init__(seed)
        self._n_runs = n_runs
        self._max_retry_count = max_retry_count
        self._grid_search_space = grid_search_space or {}

        if self._grid_search_space:
            keys = sorted(self._grid_search_space.keys())
            values = [self._grid_search_space[k] for k in keys]
            self._grid_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        else:
            self._grid_combinations = []

    def _count_random_groups(self, study: Study) -> int:
        """Count non-voided trials that have the 'random_group' system attr and are in COMPLETE/FAIL/RUNNING state."""
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)
        voided_trial_ids = study._storage.get_study_user_attrs(study._study_id).get('voided_trial_ids', [])

        count = 0
        for t in trials:
            if t._trial_id in voided_trial_ids:
                continue
            if "random_group" not in t.system_attrs:
                continue
            if t.state in (TrialState.COMPLETE, TrialState.FAIL, TrialState.RUNNING):
                count += 1
        return count

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        if "fixed_params" in trial.system_attrs:
            return

        if self._n_runs is not None:
            with FileLock(f"output/{study.study_name}.lock"):
                count = self._count_random_groups(study)
                if count >= self._n_runs:
                    study._storage.set_trial_user_attr(trial._trial_id, PREEMPTED_KEY, True)
                    study._storage.set_trial_user_attr(trial._trial_id, PREEMPTION_REASON_KEY, PREEMPTION_REASON_NO_MORE_TRIALS)
                    return
                study._storage.set_trial_system_attr(trial._trial_id, "random_group", True)
        else:
            study._storage.set_trial_system_attr(trial._trial_id, "random_group", True)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        is_enqueued = "fixed_params" in trial.system_attrs

        # 1. Retry failed trials
        if state == TrialState.FAIL:
            retry_count = trial.user_attrs.get(RETRY_COUNT_KEY, 0)
            if retry_count < self._max_retry_count:
                study.enqueue_trial(dict(trial.params), user_attrs={RETRY_COUNT_KEY: retry_count + 1})

        # 2. Enqueue grid combos (only for primary trials)
        if not is_enqueued and self._grid_search_space:
            this_grid_combo = {k: trial.params[k] for k in self._grid_search_space}
            random_params = {k: v for k, v in trial.params.items() if k not in self._grid_search_space}

            for combo in self._grid_combinations:
                if combo == this_grid_combo:
                    continue
                enqueue_params = {**random_params, **combo}
                study.enqueue_trial(enqueue_params)

    def is_exhausted(self, study: Study) -> bool:
        if self._n_runs is None:
            return False
        return self._count_random_groups(study) >= self._n_runs
