from __future__ import annotations

from matplotlib import pyplot as plt

from mcr.avinfra_persuasion.bp.signals import TypedStateDependentMaskSignalPolicy
from mcr.avinfra_persuasion.datastructures import MetricName
from mcr.avinfra_persuasion.experiments.experiment_3_0 import (
    build_typed_state_dependent_game_three,
)
from mcr.avinfra_persuasion.experiments.plotting import (
    plot_state_mask_policy,
    plot_typed_state_policy_gradient_field,
)


def test_experiment_3_0_typed_state_dependent_evaluate_policy_runs() -> None:
    game = build_typed_state_dependent_game_three(seed=3, n_humans=1, n_avs=1)

    evaluation = game.evaluate_policy()

    assert isinstance(game.sender.signal_policy, TypedStateDependentMaskSignalPolicy)
    assert game.sender.signal_policy.type_names == frozenset({"human", "av"})
    assert len(evaluation["breakdown_rows"]) == 64
    assert "masks_by_type" in evaluation["breakdown_rows"][0]


def test_experiment_3_0_plots_typed_policy_and_gradient_field() -> None:
    game = build_typed_state_dependent_game_three(seed=3, n_humans=1, n_avs=1)
    result = game.solve(max_iter=1)
    _, axes = plt.subplots(1, 2)

    policy_ax = plot_state_mask_policy(result, ax=axes[0])
    gradient_ax = plot_typed_state_policy_gradient_field(
        MetricName.HAZARD,
        MetricName.TRAVEL_TIME,
        game,
        state_name=game._state_order[0],
        type_name=game._type_order[0],
        result=result,
        ax=axes[1],
        grid_size=2,
        show_colorbar=False,
    )

    assert policy_ax is axes[0]
    assert gradient_ax is axes[1]
    plt.close(axes[0].figure)
