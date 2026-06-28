"""
Finite-prior toy experiment for typed lottery-policy private signals.

The sender commits to one state-dependent mask distribution for each anonymous
receiver type. A type is the receiver mode plus OD pair, e.g.
``human_0_0_1_1`` or ``av_0_0_1_1``. Lottery-policy evaluation then draws
private masks within each type and compresses indistinguishable same-type
assignments into mask-count profiles.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..bp.game import Preference
from ..bp.receivers import PriorRouteChoiceReceiver
from ..bp.senders import Objective, ScalarSender
from ..bp.signals import TypedStateDependentMaskSignalPolicy
from ..datastructures import (
    Demand,
    FinitePrior,
    Individual,
    MetricName,
    Node,
    World,
)
from ..networks.toy_3 import create_sample_graph
from .experiment_3_0 import build_informative_prior
from .games.osmrsptslp import OSMRSPTSLPGame
from .helpers import format_mask


TypedMaskPolicy = Mapping[str, Mapping[str, Mapping[frozenset[MetricName], float]]]


def _node_key(node: Node) -> str:
    if isinstance(node, tuple):
        return "_".join(str(part) for part in node)
    return str(node).replace(" ", "_")


def receiver_type(mode: str, demand: Demand) -> str:
    return (
        f"{mode}_{_node_key(demand.origin)}_"
        f"{_node_key(demand.destination)}"
    )


def _receiver_mode(individual: Individual) -> str:
    return "human" if individual.id.startswith("h") else "av"

#################################
# This is only for initialization
#################################
def _mask_distribution_for(
    *,
    state_name: str,
    receiver_type_name: str,
) -> dict[frozenset[MetricName], float]:
    none = frozenset()
    travel_time = frozenset({MetricName.TRAVEL_TIME})
    hazard = frozenset({MetricName.HAZARD})
    both = frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD})
    is_fast = state_name.startswith("fast")
    is_risky = state_name.endswith("risky")
    mode = receiver_type_name.split("_", maxsplit=1)[0]

    if mode == "human":
        none_probability = 0.05
        travel_time_probability = (
            0.70
            if is_fast and not is_risky
            else 0.55
            if is_fast
            else 0.25
            if not is_risky
            else 0.15
        )
        hazard_probability = 0.15 if is_risky else 0.05
        return {
            none: none_probability,
            travel_time: travel_time_probability,
            hazard: hazard_probability,
            both: 1.0
            - none_probability
            - travel_time_probability
            - hazard_probability,
        }

    if mode == "av":
        none_probability = 0.05
        travel_time_probability = 0.10 if is_risky else 0.35
        hazard_probability = 0.55 if is_risky else 0.15
        return {
            none: none_probability,
            travel_time: travel_time_probability,
            hazard: hazard_probability,
            both: 1.0
            - none_probability
            - travel_time_probability
            - hazard_probability,
        }

    raise ValueError(f"Unknown receiver mode in type {receiver_type_name!r}.")


def initial_lottery_signal_distributions(
    *,
    state_names: set[str] | frozenset[str],
    type_names: set[str] | frozenset[str],
) -> dict[str, dict[str, dict[frozenset[MetricName], float]]]:
    return {
        state_name: {
            type_name: _mask_distribution_for(
                state_name=state_name,
                receiver_type_name=type_name,
            )
            for type_name in sorted(type_names)
        }
        for state_name in sorted(state_names)
    }
#################################

def build_lottery_policy_game_four(
    seed: int = 1,
    n_humans: int = 2,
    n_avs: int = 2,
) -> OSMRSPTSLPGame:
    network = create_sample_graph(instrumented="tlbr")
    origin: Node = (0, 0)
    target: Node = (1, 1)
    demand = Demand(origin, target)
    individuals = frozenset(
        [Individual(id=f"h{i}", demand=demand) for i in range(n_humans)]
        + [Individual(id=f"a{i}", demand=demand) for i in range(n_avs)]
    )
    world = World(network=network, individuals=individuals)
    prior = build_informative_prior(world)

    human_preference = Preference(
        elements={
            MetricName.TRAVEL_TIME,
            MetricName.COST,
            MetricName.EMISSIONS,
        },
        relations={
            (MetricName.COST, MetricName.TRAVEL_TIME),
            (MetricName.EMISSIONS, MetricName.TRAVEL_TIME),
        },
    )
    av_preference = Preference(
        elements={
            MetricName.TRAVEL_TIME,
            MetricName.HAZARD,
            MetricName.COST,
        },
        relations={
            (MetricName.COST, MetricName.HAZARD),
            (MetricName.TRAVEL_TIME, MetricName.HAZARD),
        },
    )
    sender_preference = Preference(
        elements={MetricName.TRAVEL_TIME},
        relations=set(),
    )
    receiver_types = frozenset(
        receiver_type(_receiver_mode(individual), individual.demand)
        for individual in individuals
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=sender_preference,
        objective=Objective.MINIMIZE,
        signal_policy=TypedStateDependentMaskSignalPolicy(
            seed=seed,
            type_names=receiver_types,
            state_names=frozenset(prior.support),
            considered_metrics=frozenset(
                {
                    MetricName.TRAVEL_TIME,
                    MetricName.HAZARD,
                }
            ),
            state_type_probabilities=initial_lottery_signal_distributions(
                state_names=frozenset(prior.support),
                type_names=receiver_types,
            ),
        ),
    )
    receivers = [
        PriorRouteChoiceReceiver(
            individual=individual,
            rtype=receiver_type(_receiver_mode(individual), individual.demand),
            preference=human_preference
            if individual.id.startswith("h")
            else av_preference,
            prior=prior,
            world=world,
            sender=sender,
            n_scenarios=len(prior.support),
        )
        for individual in sorted(individuals, key=lambda individual: individual.id)
    ]
    return OSMRSPTSLPGame(
        sender=sender,
        receivers=receivers,
        world=world,
        public_prior=prior,
        seed=seed,
    )


def build_informative_game_four(
    seed: int = 1,
    n_humans: int = 2,
    n_avs: int = 2,
) -> OSMRSPTSLPGame:
    return build_lottery_policy_game_four(
        seed=seed,
        n_humans=n_humans,
        n_avs=n_avs,
    )


def learn_lottery_policy_game_four(
    seed: int = 1,
    n_humans: int = 2,
    n_avs: int = 2,
    max_iter: int = 2,
    step_size: float = 0.1,
    finite_diff_epsilon: float = 1e-4,
    convergence_tol: float = 1e-8,
    convergence_patience: int = 10,
) -> tuple[OSMRSPTSLPGame, dict[str, Any]]:
    game = build_lottery_policy_game_four(
        seed=seed,
        n_humans=n_humans,
        n_avs=n_avs,
    )
    initial_evaluation = game.evaluate_policy()
    result = game.solve(
        max_iter=max_iter,
        step_size=step_size,
        finite_diff_epsilon=finite_diff_epsilon,
        convergence_tol=convergence_tol,
        convergence_patience=convergence_patience,
    )
    result["initial_expected_sender_metric"] = initial_evaluation[
        "expected_sender_metric"
    ]
    return game, result


def format_mask_count_profile(
    mask_counts: Mapping[tuple[MetricName, ...], int],
) -> str:
    return ", ".join(
        f"{format_mask(mask)} x {count}"
        for mask, count in sorted(
            mask_counts.items(),
            key=lambda item: (
                len(item[0]),
                tuple(metric.value for metric in item[0]),
            ),
        )
    )


def format_lottery_profile(
    mask_counts_by_type: Mapping[str, Mapping[tuple[MetricName, ...], int]],
) -> str:
    return "; ".join(
        f"{type_name}: {format_mask_count_profile(mask_counts)}"
        for type_name, mask_counts in sorted(mask_counts_by_type.items())
    )


def print_typed_mask_distributions(
    title: str,
    probabilities: TypedMaskPolicy,
) -> None:
    print(title)
    for state_name, type_distributions in probabilities.items():
        print(f"  {state_name}:")
        for type_name, distribution in type_distributions.items():
            print(f"    {type_name}:")
            for mask, probability in sorted(
                distribution.items(),
                key=lambda item: (
                    len(item[0]),
                    tuple(
                        metric.value
                        for metric in sorted(item[0], key=lambda metric: metric.value)
                    ),
                ),
            ):
                print(f"      {format_mask(mask)}: {probability:.3f}")


def plot_lottery_signal_policy(
    result: Mapping[str, Any],
    ax: Axes | None = None,
) -> Axes:
    final_probabilities = result.get("final_probabilities")
    if not isinstance(final_probabilities, Mapping) or not final_probabilities:
        raise ValueError(
            "result must contain a non-empty 'final_probabilities' mapping."
        )

    state_names = tuple(sorted(str(state_name) for state_name in final_probabilities))
    first_state = final_probabilities[state_names[0]]
    if not isinstance(first_state, Mapping) or not first_state:
        raise ValueError("Each state must contain at least one receiver type.")
    type_names = tuple(sorted(str(type_name) for type_name in first_state))
    first_distribution = first_state[type_names[0]]
    if not isinstance(first_distribution, Mapping) or not first_distribution:
        raise ValueError("Each state/type distribution must be non-empty.")

    masks = tuple(
        sorted(
            first_distribution,
            key=lambda mask: (
                len(mask),
                tuple(
                    metric.value
                    for metric in sorted(mask, key=lambda metric: metric.value)
                ),
            ),
        )
    )
    row_labels = tuple(
        f"{state_name}\n{type_name}"
        for state_name in state_names
        for type_name in type_names
    )
    values = np.asarray(
        [
            [
                float(final_probabilities[state_name][type_name][mask])
                for mask in masks
            ]
            for state_name in state_names
            for type_name in type_names
        ],
        dtype=float,
    )

    width = max(7.0, 1.25 * len(masks))
    height = max(4.0, 0.55 * len(row_labels) + 1.7)
    ax = ax or plt.subplots(figsize=(width, height))[1]
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(masks)))
    ax.set_xticklabels([format_mask(mask) for mask in masks], rotation=25, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Signal mask")
    ax.set_ylabel("State and receiver type")
    ax.set_title("Learned lottery signal policy")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            text_color = "#111111" if value > 0.62 else "#f5f5f5"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("P(mask | state, receiver type)")
    return ax


if __name__ == "__main__":
    game, result = learn_lottery_policy_game_four(
        seed=1,
        n_humans=2,
        n_avs=2,
        max_iter=20,
    )

    print("Receiver types:")
    for type_name in sorted({receiver.rtype for receiver in game.receivers}):
        print(f"  {type_name}")

    print("Learning summary:")
    print("  iterations:", result["iterations"])
    print("  converged:", result["converged"])
    print(
        "  initial expected sender metric:",
        f"{result['initial_expected_sender_metric']:.4f}",
    )
    print(
        "  learned expected sender metric:",
        f"{result['expected_sender_metric']:.4f}",
    )

    print_typed_mask_distributions(
        "Learned lottery-policy distributions:",
        result["final_probabilities"],
    )

    print("Learned lottery count-profile breakdown sample:")
    for row in result["breakdown_rows"][:8]:
        print(
            f"  scenario={row['scenario_name']} "
            f"(p={row['scenario_probability']:.3f}) | "
            f"counts=({format_lottery_profile(row['mask_counts_by_type'])}) "
            f"(p={row['mask_probability']:.3f}) | "
            f"path_counts={row['path_counts']} | "
            f"sender_metric={row['sender_metric_value']:.3f} | "
            f"contribution={row['weighted_contribution']:.3f}"
        )

    print("Final expected sender metric:", f"{result['expected_sender_metric']:.4f}")

    _, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plot_lottery_signal_policy(result, ax=axes[0])
    plt.tight_layout()
    plt.show()
