"""
Finite-prior toy experiment with typed, state-dependent private signals.

The sender observes the realized state and commits to one mask distribution per
receiver type. Receivers of the same type observe the same type-specific signal,
while human and AV receivers may receive different masked metric observations in
the same realized state.
"""

from __future__ import annotations

from collections.abc import Mapping

from matplotlib import pyplot as plt

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
    Scenario,
    World,
)
from ..networks.toy_3 import create_sample_graph
from .games.osmrspts import OSMRSPTSGame
from .helpers import format_mask
from .plotting import plot_state_mask_policy, plot_typed_state_policy_gradient_fields


def build_informative_prior(world: World) -> FinitePrior:
    source_right = ((0, 0), (0, 1))
    right_target = ((0, 1), (1, 1))
    source_to_bottom = ((0, 0), (1, 0))
    bottom_to_target = ((1, 0), (1, 1))

    base_scenario = Scenario.from_world("base", world)

    def make_scenario(
        name: str,
        *,
        top_travel_time: float,
        top_hazard: float,
        node_policing: float,
    ) -> Scenario:
        return base_scenario.with_overrides(
            name=name,
            arc_overrides={
                MetricName.TRAVEL_TIME: {
                    source_right: top_travel_time,
                    right_target: top_travel_time,
                    source_to_bottom: 1.0,
                    bottom_to_target: 1.0,
                },
                MetricName.HAZARD: {
                    source_right: top_hazard,
                    right_target: top_hazard,
                    source_to_bottom: 0.5,
                    bottom_to_target: 0.5,
                },
                MetricName.COST: {
                    source_right: 1.0,
                    right_target: 1.0,
                    source_to_bottom: 0.0,
                    bottom_to_target: 0.0,
                },
                MetricName.EMISSIONS: {
                    source_right: 1.0,
                    right_target: 1.0,
                    source_to_bottom: 0.5,
                    bottom_to_target: 0.5,
                },
            },
            node_overrides={
                MetricName.POLICING: {
                    (0, 0): node_policing,
                    (1, 0): 0.5,
                    (1, 1): node_policing,
                }
            },
        )

    support = {
        "fast_safe": make_scenario(
            "fast_safe",
            top_travel_time=0.25,
            top_hazard=0.0,
            node_policing=1.0,
        ),
        "fast_risky": make_scenario(
            "fast_risky",
            top_travel_time=0.25,
            top_hazard=2.0,
            node_policing=0.0,
        ),
        "slow_safe": make_scenario(
            "slow_safe",
            top_travel_time=1.25,
            top_hazard=0.0,
            node_policing=1.0,
        ),
        "slow_risky": make_scenario(
            "slow_risky",
            top_travel_time=1.25,
            top_hazard=2.0,
            node_policing=0.0,
        ),
    }

    return FinitePrior(
        name="prior",
        support=support,
        probabilities={
            "fast_safe": 0.4,
            "fast_risky": 0.2,
            "slow_safe": 0.3,
            "slow_risky": 0.1,
        },
    )


def initial_typed_signal_distributions(
    state_names: set[str] | frozenset[str],
) -> dict[str, dict[str, dict[frozenset[MetricName], float]]]:
    none = frozenset()
    travel_time = frozenset({MetricName.TRAVEL_TIME})
    hazard = frozenset({MetricName.HAZARD})
    both = frozenset({MetricName.TRAVEL_TIME, MetricName.HAZARD})
    distributions: dict[str, dict[str, dict[frozenset[MetricName], float]]] = {}

    for state_name in state_names:
        is_fast = state_name.startswith("fast")
        is_risky = state_name.endswith("risky")
        distributions[state_name] = {
            "human": {
                none: 0.05,
                travel_time: 0.70 if is_fast else 0.20,
                hazard: 0.05 if is_fast else 0.10,
                both: 0.20 if is_fast else 0.65,
            },
            "av": {
                none: 0.05,
                travel_time: 0.10 if is_risky else 0.35,
                hazard: 0.65 if is_risky else 0.15,
                both: 0.20 if is_risky else 0.45,
            },
        }

    return distributions


def build_typed_state_dependent_game_three(
    seed: int = 1,
    n_humans: int = 10,
    n_avs: int = 10,
) -> OSMRSPTSGame:
    network = create_sample_graph(instrumented="tl")
    origin: Node = (0, 0)
    target: Node = (1, 1)
    individuals = frozenset(
        [Individual(id=f"h{i}", demand=Demand(origin, target)) for i in range(n_humans)]
        + [Individual(id=f"a{i}", demand=Demand(origin, target)) for i in range(n_avs)]
    )
    world = World(network=network, individuals=individuals)
    prior = build_informative_prior(world)

    human_preference = Preference(
        elements={
            MetricName.TRAVEL_TIME,
            # MetricName.COST,
            # MetricName.EMISSIONS,
        },
        relations={
            # (MetricName.COST, MetricName.TRAVEL_TIME),
            # (MetricName.EMISSIONS, MetricName.TRAVEL_TIME),
        },
    )
    av_preference = Preference(
        elements={
            MetricName.TRAVEL_TIME,
            # MetricName.HAZARD,
            # MetricName.COST,
        },
        relations={
            # (MetricName.COST, MetricName.HAZARD),
            # (MetricName.TRAVEL_TIME, MetricName.HAZARD),
        },
    )
    sender_preference = Preference(
        elements={MetricName.TRAVEL_TIME},
        relations=set(),
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=sender_preference,
        objective=Objective.MINIMIZE,
        signal_policy=TypedStateDependentMaskSignalPolicy(
            seed=seed,
            type_names=frozenset({"human", "av"}),
            state_names=frozenset(prior.support),
            considered_metrics=frozenset(
                {
                    MetricName.TRAVEL_TIME,
                    MetricName.HAZARD,
                }
            ),
            state_type_probabilities=initial_typed_signal_distributions(
                frozenset(prior.support)
            ),
        ),
    )
    receivers = [
        PriorRouteChoiceReceiver(
            individual=individual,
            rtype="human" if individual.id.startswith("h") else "av",
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
    return OSMRSPTSGame(
        sender=sender,
        receivers=receivers,
        world=world,
        public_prior=prior,
        seed=seed,
    )


def build_informative_game_three(
    seed: int = 1,
    n_humans: int = 10,
    n_avs: int = 10,
) -> OSMRSPTSGame:
    return build_typed_state_dependent_game_three(
        seed=seed,
        n_humans=n_humans,
        n_avs=n_avs,
    )


def format_typed_mask_profile(
    masks_by_type: Mapping[str, tuple[MetricName, ...]],
) -> str:
    return ", ".join(
        f"{type_name}={format_mask(mask)}"
        for type_name, mask in sorted(masks_by_type.items())
    )


if __name__ == "__main__":
    game = build_typed_state_dependent_game_three(seed=1, n_humans=5, n_avs=5)
    result = game.solve(max_iter=1)

    print("Converged:", result["converged"])
    print("Iterations:", result["iterations"])
    print("Final typed state-dependent mask distributions:")
    for state_name, type_distributions in result["final_probabilities"].items():
        print(f"  {state_name}:")
        for type_name, distribution in type_distributions.items():
            print(f"    {type_name}:")
            for mask in sorted(
                distribution,
                key=lambda mask: (
                    len(mask),
                    tuple(
                        metric.value
                        for metric in sorted(mask, key=lambda metric: metric.value)
                    ),
                ),
            ):
                print(f"      {format_mask(mask)}: {distribution[mask]:.4f}")

    print("Per-scenario / per-type-mask breakdown:")
    for row in result["breakdown_rows"]:
        print(
            f"  scenario={row['scenario_name']} "
            f"(p={row['scenario_probability']:.3f}) | "
            f"masks=({format_typed_mask_profile(row['masks_by_type'])}) "
            f"(p={row['mask_probability']:.3f}) | "
            f"path_counts={row['path_counts']} | "
            f"sender_metric={row['sender_metric_value']:.3f} | "
            f"contribution={row['weighted_contribution']:.3f}"
        )

    print("Final expected sender metric:", f"{result['expected_sender_metric']:.4f}")
    plot_state_mask_policy(result)
    plot_typed_state_policy_gradient_fields(
        MetricName.HAZARD,
        MetricName.TRAVEL_TIME,
        game,
        result=result,
        grid_size=10,
    )
    plt.show()
