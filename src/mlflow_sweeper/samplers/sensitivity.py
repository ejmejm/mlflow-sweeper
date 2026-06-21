from typing import Any, Mapping, Sequence

from optuna.samplers._grid import GridValueType
from optuna.samplers._lazy_random_state import LazyRandomState

from mlflow_sweeper.samplers.grid import GridSampler


class SensitivitySampler(GridSampler):
    """One-parameter-at-a-time sensitivity sweep off a baseline.

    A sensitivity sweep holds every parameter at its default value and varies one
    parameter at a time over its candidate values, instead of taking the full
    cartesian product like a grid sweep. The set of trials is one baseline run (all
    defaults) plus, for each varied parameter, one run per non-default candidate value.

    This subclasses the project ``GridSampler`` so that grid-id assignment, file-lock
    coordination, retry accounting, preemption, and ``is_exhausted`` are inherited
    unchanged. Only ``__init__`` differs: instead of building ``_all_grids`` from the
    cartesian product, it enumerates the one-at-a-time trial list. Optuna's
    ``sample_independent`` reads ``_all_grids[grid_id][_param_names.index(name)]`` and is
    agnostic to how ``_all_grids`` was constructed, so the rest works as-is.
    """

    def __init__(
        self,
        search_space: Mapping[str, Sequence[GridValueType]],
        defaults: Mapping[str, Any],
        seed: int | None = None,
        max_retry_count: int = 3,
    ) -> None:
        # NOTE: deliberately not calling super().__init__ — the base builds the
        # cartesian product. We replicate its attribute contract by hand, swapping the
        # grid enumeration for the sensitivity one.
        for param_name, param_values in search_space.items():
            for value in param_values:
                self._check_value(param_name, value)

        self._search_space = {}
        for param_name, param_values in sorted(search_space.items()):
            self._search_space[param_name] = list(param_values)

        self._param_names = sorted(search_space.keys())
        self._all_grids = self._build_sensitivity_grids(defaults)
        self._n_min_trials = len(self._all_grids)
        self._rng = LazyRandomState(seed or 0)
        self._rng.rng.shuffle(self._all_grids)  # type: ignore[arg-type]
        self._max_retry_count = max_retry_count

    def _build_sensitivity_grids(self, defaults: Mapping[str, Any]) -> list[tuple]:
        """Enumerate the one-at-a-time trial points.

        Tuples are ordered by ``self._param_names`` (sorted) so that
        ``sample_independent``'s ``_param_names.index(name)`` lookup is correct. The
        baseline (all defaults) is included once; each variation differs from the
        baseline in exactly one coordinate. Candidate values equal to the default are
        skipped because the baseline already covers them.
        """
        baseline = tuple(defaults[name] for name in self._param_names)
        grids: list[tuple] = [baseline]
        for idx, name in enumerate(self._param_names):
            for value in self._search_space[name]:
                if self._grid_value_equal(value, defaults[name]):
                    continue
                point = list(baseline)
                point[idx] = value
                grids.append(tuple(point))
        return grids
