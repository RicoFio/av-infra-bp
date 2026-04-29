"""
Finite-prior toy experiment with a family of receivers and a per-metric policy.

The prior support is an epistemic model of the world used for signaling and
belief updates. Once all receivers choose routes, the actual realized network
metrics are computed endogenously from the joint path profile via
``world.get_realized_metrics(...)``.

sender has full commitment power
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from ..bp.game import Preference
from ..bp.receivers import PriorRouteChoiceReceiver
from ..bp.senders import ScalarSender, Objective
from ..bp.signals import StateDependentMaskSignalPolicy
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
from .helpers import format_mask
from .plotting import plot_state_mask_policy, plot_state_policy_gradient_fields
from .games.osmrsp import OSMRSPGame
from .experiment_2_0 import build_informative_prior


def build_informative_game_two(
    seed: int = 1,
    n_humans: int = 10,
    n_avs: int = 10,
) -> OSMRSPGame:
    network = create_sample_graph(instrumented="tr")
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
            # MetricName.HAZARD,
            # MetricName.COST,
            MetricName.TRAVEL_TIME,
        },
        relations={
            # (MetricName.COST, MetricName.HAZARD),
        },
    )
    # av_preference.draw_hasse_diagram()
    # plt.show()
    sender_preference = Preference(
        elements={MetricName.TRAVEL_TIME},
        relations=set(),
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=sender_preference,
        objective=Objective.MINIMIZE,
        signal_policy=StateDependentMaskSignalPolicy(
            seed=seed,
            state_names=frozenset(prior.support),
            considered_metrics=frozenset(
                {
                    MetricName.TRAVEL_TIME,
                    MetricName.HAZARD,
                },
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
    return OSMRSPGame(
        sender=sender,
        receivers=receivers,
        world=world,
        public_prior=prior,
        seed=seed,
    )


if __name__ == "__main__":
    game = build_informative_game_two(seed=1, n_humans=5, n_avs=5)
    result = game.solve(max_iter=200)

    print("Converged:", result["converged"])
    print("Iterations:", result["iterations"])
    print("Final state-dependent mask distributions:")
    for state_name, distribution in result["final_probabilities"].items():
        print(f"  {state_name}:")
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
            print(f"    {format_mask(mask)}: {distribution[mask]:.4f}")

    print("Per-scenario / per-mask breakdown:")
    for row in result["breakdown_rows"]:
        print(
            f"  scenario={row['scenario_name']} "
            f"(p={row['scenario_probability']:.3f}) | "
            f"mask={format_mask(row['mask'])} "
            f"(p={row['mask_probability']:.3f}) | "
            f"path_counts={row['path_counts']} | "
            f"sender_metric={row['sender_metric_value']:.3f} | "
            f"contribution={row['weighted_contribution']:.3f}"
        )

    print("Final expected sender metric:", f"{result['expected_sender_metric']:.4f}")

    plot_state_mask_policy(result)
    plot_state_policy_gradient_fields(
        MetricName.HAZARD,
        MetricName.TRAVEL_TIME,
        game,
        result=result,
        grid_size=20,
    )
    plt.show()
