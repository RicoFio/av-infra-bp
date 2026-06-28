from typing import Mapping
from matplotlib import pyplot as plt

from .orders import PartialOrder, Relation
from .bp.receivers import Receiver, Preference
from .routing.routing_solvers import RoutingSolverConfig
from .datastructures import (
    World,
    FinitePrior,
    METRIC_SET,
    MetricName,
    Individual,
    Demand,
    Scenario,
)
from .networks.toy_2 import make_toy_world as make_toy2_world
from .plotting import plot_scenario, plot_infrastructure, plot_pareto_frontier


def compare_individual_choices(world: World, individuals: Mapping[str, Receiver]):
    path_choices = {
        n: i.get_path_choice().path for n, i in individuals.items()
    }
    
    solution = individuals["human"]._compute_paths()
    # fig, ax = plt.subplots(1,1)
    # ax.set_xlim(4,12)
    # ax.set_ylim(-1,6)
    # fig.suptitle("MOSP", fontweight="bold")
    # plot_pareto_frontier(
    #     solution,
    #     x_metric=MetricName.TRAVEL_TIME,
    #     y_metric=MetricName.HAZARD,
    #     objective_names=solution.objective_names,
    #     ax=ax,
    #     connect=False,
    #     highlighting={
    #         "human": individuals['human'].get_path_choice().label,
    #         "av": individuals['av'].get_path_choice().label
    #     }
    # )
    
    # _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    # plot_infrastructure(
    #     world.network,
    # )
    
    plot_scenario(
        world=world,
        # We can sample here since we have one scenario in the support with p=1
        scenario=individuals['human'].prior.sample(1)[0],
        paths={},
        # `_, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)` is creating a new figure and a
        # set of subplots.
        # ax=ax,
        arc_metric=MetricName.HAZARD,
        node_metric=None
    )
    
    plot_scenario(
        world=world,
        # We can sample here since we have one scenario in the support with p=1
        scenario=individuals['human'].prior.sample(1)[0],
        paths={},
        # ax=ax,
        arc_metric=MetricName.TRAVEL_TIME,
        node_metric=None
    )
    
    plot_scenario(
        world=world,
        # We can sample here since we have one scenario in the support with p=1
        scenario=individuals['human'].prior.sample(1)[0],
        paths=path_choices,
        # ax=ax,
        arc_metric=None,
        node_metric=None
    )
    
    plt.show()


if __name__ == "__main__":
    # Build the world
    world = make_toy2_world()
    origin_node = (0, 0)
    destination_node = (4, 4)
    routing_solver_config = RoutingSolverConfig(backend="mosp")

    prior = FinitePrior(
        "toy_setup",
        support={"toy_scenario": Scenario.from_world("toy_scenario", world=world)},
        probabilities={"toy_scenario": 1.0},
    )

    # Build individuals
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
    print("Human preference")
    # human_preference.draw_hasse_diagram()
    human_receiver = Receiver(
        individual=Individual(
            id="human",
            demand=Demand(
                origin=origin_node,
                destination=destination_node,
            ),
        ),
        rtype="human",
        preference=human_preference,
        prior=prior,
        world=world,
        routing_solver_config=routing_solver_config,
    )

    av_preference = Preference(
        elements={
            MetricName.TRAVEL_TIME,
            MetricName.LEFT_TURNS,
            MetricName.HAZARD,
        },
        relations={
            (MetricName.TRAVEL_TIME, MetricName.LEFT_TURNS),
            (MetricName.TRAVEL_TIME, MetricName.HAZARD),
        },
    )
    print("AV preference")
    # av_preference.draw_hasse_diagram()
    av_receiver = Receiver(
        individual=Individual(
            id="av",
            demand=Demand(
                origin=origin_node,
                destination=destination_node,
            ),
        ),
        rtype="av",
        preference=av_preference,
        prior=prior,
        world=world,
        routing_solver_config=routing_solver_config,
    )

    individuals = {
        human_receiver.id: human_receiver,
        av_receiver.id: av_receiver
    }

    compare_individual_choices(world, individuals)
