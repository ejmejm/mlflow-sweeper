from collections import Counter
import logging
from typing import Mapping, Sequence, Tuple

from filelock import FileLock
from optuna import Study
from optuna.samplers import GridSampler as BaseGridSampler
from optuna.samplers._grid import GridValueType
from optuna.trial import FrozenTrial, TrialState

from mlflow_sweeper.optimize import PREEMPTED_KEY, PREEMPTION_REASON_KEY, PREEMPTION_REASON_NO_MORE_TRIALS


RETRY_COUNT_KEY = "retry_count"


logger = logging.getLogger(__name__)


class GridSampler(BaseGridSampler):
    """Version of the GridSampler that does not run duplicate trials."""

    def __init__(
        self,
        search_space: Mapping[str, Sequence[GridValueType]],
        seed: int | None = None,
        max_retry_count: int = 3,
    ) -> None:
        super().__init__(search_space, seed)
        self._max_retry_count = max_retry_count

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # Instead of returning param values, GridSampler puts the target grid id as a system attr,
        # and the values are returned from `sample_independent`. This is because the distribution
        # object is hard to get at the beginning of trial, while we need the access to the object
        # to validate the sampled value.

        # When the trial is created by RetryFailedTrialCallback or enqueue_trial, we should not
        # assign a new grid_id.
        if "grid_id" in trial.system_attrs or "fixed_params" in trial.system_attrs:
            return

        # TODO: Change this file lock to a better location or a database lock.
        with FileLock(f"output/{study.study_name}.lock"):
            pending_grid_ids = self._get_pending_grid_ids(study)
            
            if len(pending_grid_ids) == 0:
                # This case may occur with distributed optimization or trial queue. If there is no
                # target grid, `GridSampler` will preempt the trial and set a flag to stop the study.
                study._storage.set_trial_user_attr(trial._trial_id, PREEMPTED_KEY, True)
                study._storage.set_trial_user_attr(trial._trial_id, PREEMPTION_REASON_KEY, PREEMPTION_REASON_NO_MORE_TRIALS)
                return
        
            # In distributed optimization, multiple workers may simultaneously pick up the same grid.
            # To make the conflict less frequent, the grid is chosen randomly.
            grid_id = int(self._rng.rng.choice(pending_grid_ids))

            study._storage.set_trial_system_attr(trial._trial_id, "search_space", self._search_space)
            study._storage.set_trial_system_attr(trial._trial_id, "grid_id", grid_id)

    def _get_pending_grid_ids(self, study: Study) -> list[int]:
        """Get a list of grid ids that are pending."""
        grid_visitation_counts = Counter()
        resolved_grid_ids = set()

        # We directly query the storage to get trials here instead of `study.get_trials`,
        # since some pruners such as `HyperbandPruner` use the study transformed
        # to filter trials. See https://github.com/optuna/optuna/issues/2327 for details.
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)
        voided_trial_ids = study._storage.get_study_user_attrs(study._study_id).get('voided_trial_ids', [])

        for t in trials:
            not_voided = t._trial_id not in voided_trial_ids
            has_grid_id = "grid_id" in t.system_attrs
            
            if not_voided and has_grid_id and self._same_search_space(t.system_attrs.get("search_space", {})):
                if t.state.is_finished():
                    grid_visitation_counts[t.system_attrs["grid_id"]] += 1
                if t.state in (TrialState.RUNNING, TrialState.COMPLETE):
                    resolved_grid_ids.add(t.system_attrs["grid_id"])
    
        pending_grid_ids = set(range(self._n_min_trials)) - resolved_grid_ids
        pending_grid_ids = [
            grid_id for grid_id in pending_grid_ids
            if grid_visitation_counts[grid_id] <= self._max_retry_count
        ]
        
        return pending_grid_ids

    def is_exhausted(self, study: Study) -> bool:
        # Override the base is_exhausted which only checks if all grid IDs have
        # been visited, ignoring retry logic and voided trials.
        return len(self._get_pending_grid_ids(study)) == 0

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        # Override the base GridSampler.after_trial to prevent it from
        # prematurely stopping the study. We intentionally do nothing here
        # because Optuna calls after_trial BEFORE writing the trial's final
        # state to storage (_tell_with_warning). That means a just-failed
        # trial still appears as RUNNING, causing _get_pending_grid_ids to
        # incorrectly treat its grid_id as resolved.  Instead, we let
        # before_trial's preemption path handle the stop: by the next
        # iteration the state has been flushed, so the computation is correct.
        pass