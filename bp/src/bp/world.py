"""World model: infrastructure, population, scenarios, and priors.

The solver-agnostic core of the project, in one file:

- `InfrastructureGraph` — the physical network and its nominal per-arc/per-node
  attributes, plus the congestion (BPR) response of travel time and emissions.
- `Demand` / `Individual` / `World` — who travels where on that network.
- `Scenario` — one realization of every uncertain metric (a "state of the world").
- `Prior` / `FinitePrior` / `SampledPrior` — the uncertainty over scenarios.

`random_grid_world` builds a small, fully reproducible grid instance for
experiments. Nothing here depends on a particular optimization backend; that
lives elsewhere. Only `numpy` and `networkx` are required.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Protocol

import networkx as nx
import numpy as np

# A node is either a label or a grid coordinate; an arc is an ordered node pair.
Node = str | tuple[int, int]
Arc = tuple[Node, Node]


class MetricName(str, Enum):
    """The quantities individuals care about and the planner reasons over."""

    TRAVEL_TIME = "travel_time"
    DISCOMFORT = "discomfort"
    HAZARD = "hazard"
    COST = "cost"
    EMISSIONS = "emissions"
    POLICING = "policing"

    def __str__(self) -> str:
        return self.value


class InfrastructureGraph:
    """A directed network whose edges/nodes carry the nominal infrastructure attributes.

    The wrapped `networkx.DiGraph` is the single source of truth. Build one with
    `from_networkx`, which fills in sensible defaults for any missing attribute.
    Per-arc attributes: ``travel_time``, ``capacity``, ``distance``, ``discomfort``,
    ``hazard``, ``cost``, ``instrumented``. Per-node attribute: ``policing``.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
    ) -> None:
        self.graph = graph
        self.bpr_alpha = bpr_alpha
        self.bpr_beta = bpr_beta

    @classmethod
    def from_networkx(
        cls,
        G: nx.DiGraph,
        *,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
    ) -> InfrastructureGraph:
        """Build infrastructure from a directed graph, filling missing attributes.

        Only ``travel_time`` is required on every edge. Defaults: ``capacity`` 1,
        ``distance`` = ``travel_time``, ``discomfort`` 0.5 if instrumented else 1,
        ``hazard``/``cost`` 0, ``instrumented`` False, node ``policing`` 0.
        """
        graph = nx.DiGraph()
        for node in G.nodes:
            graph.add_node(node, policing=float(G.nodes[node].get("policing", 0.0)))
        for u, v, data in G.edges(data=True):
            if "travel_time" not in data:
                raise ValueError(f"Arc {(u, v)!r} is missing 'travel_time'.")
            travel_time = float(data["travel_time"])
            instrumented = bool(data.get("instrumented", False))
            capacity = float(data.get("capacity", 1.0))
            if capacity <= 0:
                raise ValueError(f"Arc {(u, v)!r} has non-positive capacity.")
            graph.add_edge(
                u,
                v,
                travel_time=travel_time,
                capacity=capacity,
                distance=float(data.get("distance", travel_time)),
                discomfort=float(data.get("discomfort", 0.5 if instrumented else 1.0)),
                hazard=float(data.get("hazard", 0.0)),
                cost=float(data.get("cost", data.get("toll", 0.0))),
                instrumented=instrumented,
            )
        return cls(graph, bpr_alpha=bpr_alpha, bpr_beta=bpr_beta)

    @property
    def V(self) -> set[Node]:
        """Nodes."""
        return set(self.graph.nodes)

    @property
    def A(self) -> set[Arc]:
        """Arcs."""
        return set(self.graph.edges)

    @property
    def I(self) -> set[Arc]:
        """Instrumented arcs (the ones the planner can signal about)."""
        return {arc for arc in self.graph.edges if self.graph.edges[arc]["instrumented"]}

    def _arc_attr(self, name: str) -> dict[Arc, float]:
        return {arc: self.graph.edges[arc][name] for arc in self.graph.edges}

    @property
    def travel_time(self) -> dict[Arc, float]:
        return self._arc_attr("travel_time")

    @property
    def capacity(self) -> dict[Arc, float]:
        return self._arc_attr("capacity")

    @property
    def distance(self) -> dict[Arc, float]:
        return self._arc_attr("distance")

    @property
    def discomfort(self) -> dict[Arc, float]:
        return self._arc_attr("discomfort")

    @property
    def hazard(self) -> dict[Arc, float]:
        return self._arc_attr("hazard")

    @property
    def cost(self) -> dict[Arc, float]:
        return self._arc_attr("cost")

    @property
    def policing(self) -> dict[Node, float]:
        return {node: self.graph.nodes[node]["policing"] for node in self.graph.nodes}

    @cached_property
    def ordered_arcs(self) -> tuple[Arc, ...]:
        """Arcs in a fixed order, so vectorized quantities line up across calls."""
        return tuple(sorted(self.graph.edges, key=str))

    def _arc_array(self, name: str) -> np.ndarray:
        return np.array(
            [self.graph.edges[arc][name] for arc in self.ordered_arcs], dtype=float
        )

    def actual_travel_times(self, volumes: Mapping[Arc, float]) -> dict[Arc, float]:
        """Congestion-adjusted per-arc travel times via the BPR formula."""
        flow = np.array([volumes.get(arc, 0.0) for arc in self.ordered_arcs], dtype=float)
        free_flow = self._arc_array("travel_time")
        ratio = flow / self._arc_array("capacity")
        congested = free_flow * (1 + self.bpr_alpha * ratio**self.bpr_beta)
        return dict(zip(self.ordered_arcs, congested.tolist()))

    def actual_emissions(self, volumes: Mapping[Arc, float]) -> dict[Arc, float]:
        """Per-arc emissions: distance scaled by the realized congestion factor.

        A placeholder distance-weighted proxy — base emissions grow with arc
        distance and with the ratio of congested to free-flow travel time. Swap
        this out once a calibrated emissions model exists.
        """
        congested = np.array(
            [self.actual_travel_times(volumes)[arc] for arc in self.ordered_arcs],
            dtype=float,
        )
        free_flow = self._arc_array("travel_time")
        slowdown = np.divide(
            congested, free_flow, out=np.ones_like(congested), where=free_flow > 0
        )
        emissions = self._arc_array("distance") * slowdown
        return dict(zip(self.ordered_arcs, emissions.tolist()))


@dataclass(frozen=True)
class Demand:
    """A desire to travel from `origin` to `destination`."""

    origin: Node
    destination: Node

    def __str__(self) -> str:
        return f"{self.origin}_{self.destination}"


@dataclass(frozen=True)
class Individual:
    """A single traveler with an identifier and a demand."""

    id: str
    demand: Demand


class World:
    """An infrastructure together with the population traveling on it."""

    def __init__(
        self,
        network: InfrastructureGraph,
        individuals: Sequence[Individual] | frozenset[Individual] = (),
    ) -> None:
        self.network = network
        self.individuals = frozenset(individuals)
        for individual in self.individuals:
            for endpoint in (individual.demand.origin, individual.demand.destination):
                if endpoint not in network.V:
                    raise ValueError(
                        f"Individual {individual.id!r} references node {endpoint!r} "
                        "outside the network."
                    )

    @property
    def V(self) -> set[Node]:
        return self.network.V

    @property
    def A(self) -> set[Arc]:
        return self.network.A

    @property
    def I(self) -> set[Arc]:
        return self.network.I

    @property
    def ordered_nodes(self) -> tuple[Node, ...]:
        return tuple(sorted(self.network.V, key=str))

    @property
    def ordered_arcs(self) -> tuple[Arc, ...]:
        return self.network.ordered_arcs

    @property
    def total_population(self) -> int:
        return len(self.individuals)

    def population_at_node(self, node: Node) -> int:
        """Number of individuals whose trip starts at `node`."""
        return sum(1 for ind in self.individuals if ind.demand.origin == node)

    def total_demand(self, origin: Node, destination: Node) -> int:
        """Number of individuals traveling from `origin` to `destination`."""
        return sum(
            1
            for ind in self.individuals
            if ind.demand.origin == origin and ind.demand.destination == destination
        )

    def realize(
        self,
        routes: Mapping[Individual, Sequence[Arc]],
        name: str = "realized",
        base_scenario: Scenario | None = None,
    ) -> Scenario:
        """Turn chosen routes into a realized scenario.

        Routes determine arc volumes, which drive congested travel times and
        emissions. The remaining (non-congestion) metrics are copied from
        `base_scenario` if given, otherwise from the nominal infrastructure.
        """
        if set(routes) != self.individuals:
            raise ValueError("routes must cover exactly the world's individuals.")
        volumes = {arc: 0.0 for arc in self.A}
        for individual, route in routes.items():
            if not route:
                raise ValueError(f"Individual {individual.id!r} has an empty route.")
            for arc in route:
                if arc not in volumes:
                    raise ValueError(
                        f"Individual {individual.id!r} uses arc {arc!r} outside the network."
                    )
                volumes[arc] += 1.0

        base = base_scenario
        net = self.network
        return Scenario(
            name=name,
            travel_time=net.actual_travel_times(volumes),
            emissions=net.actual_emissions(volumes),
            discomfort=dict(base.discomfort if base else net.discomfort),
            hazard=dict(base.hazard if base else net.hazard),
            cost=dict(base.cost if base else net.cost),
            policing=dict(base.policing if base else net.policing),
        )


@dataclass(frozen=True)
class Scenario:
    """One realization of every uncertain metric — a state of the world."""

    name: str
    travel_time: Mapping[Arc, float]
    discomfort: Mapping[Arc, float]
    hazard: Mapping[Arc, float]
    cost: Mapping[Arc, float]
    emissions: Mapping[Arc, float]
    policing: Mapping[Node, float]

    @classmethod
    def from_world(cls, name: str, world: World) -> Scenario:
        """The nominal scenario: every metric at its infrastructure value, no congestion."""
        net = world.network
        empty = {arc: 0.0 for arc in net.A}
        return cls(
            name=name,
            travel_time=dict(net.travel_time),
            discomfort=dict(net.discomfort),
            hazard=dict(net.hazard),
            cost=dict(net.cost),
            emissions=net.actual_emissions(empty),
            policing=dict(net.policing),
        )


class Prior(Protocol):
    """A distribution over scenarios that can be sampled."""

    name: str

    def sample(self, n_samples: int, seed: int | None = None) -> list[Scenario]: ...


@dataclass(frozen=True)
class FinitePrior:
    """A prior over a finite set of named scenarios."""

    name: str
    support: Mapping[str, Scenario]
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if set(self.support) != set(self.probabilities):
            raise ValueError("support and probabilities must share the same keys.")
        weights = np.array([self.probabilities[k] for k in self.support], dtype=float)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("probabilities must be non-negative and sum to a positive value.")

    def sample(self, n_samples: int, seed: int | None = None) -> list[Scenario]:
        rng = np.random.default_rng(seed)
        names = list(self.support)
        weights = np.array([self.probabilities[name] for name in names], dtype=float)
        chosen = rng.choice(names, size=n_samples, replace=True, p=weights / weights.sum())
        return [self.support[name] for name in chosen]


@dataclass(frozen=True)
class SampledPrior:
    """A prior defined by a sampler function over a random generator."""

    name: str
    sampler: Callable[[np.random.Generator, int], list[Scenario]]

    def sample(self, n_samples: int, seed: int | None = None) -> list[Scenario]:
        scenarios = self.sampler(np.random.default_rng(seed), n_samples)
        if len(scenarios) != n_samples:
            raise ValueError("sampler must return exactly n_samples scenarios.")
        return scenarios


def random_grid_world(
    rows: int,
    cols: int,
    demands: Mapping[tuple[Node, Node], int],
    *,
    seed: int | None = None,
    travel_time_range: tuple[float, float] = (1.0, 10.0),
    instrumented_fraction: float = 0.0,
) -> World:
    """Build a reproducible random world on a `rows` x `cols` grid.

    Nodes are grid coordinates ``(row, col)``; arcs connect 4-neighbours in both
    directions. Every arc gets independent random metrics drawn from `seed`, and
    each node a random policing value, so the same `seed` reproduces the same
    world exactly.

    `demands` maps an ``(origin, destination)`` pair of grid coordinates to the
    number of individuals making that trip. `instrumented_fraction` is the
    probability that any given arc is instrumented.

    Capacities, discomfort, hazard and cost are drawn from fixed toy ranges;
    rescale them downstream when calibrating to a real instance.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1.")
    if not 0.0 <= instrumented_fraction <= 1.0:
        raise ValueError("instrumented_fraction must be in [0, 1].")
    low, high = travel_time_range
    if not 0 < low <= high:
        raise ValueError("travel_time_range must satisfy 0 < low <= high.")

    rng = np.random.default_rng(seed)

    grid = nx.grid_2d_graph(rows, cols).to_directed()
    graph = nx.DiGraph()
    graph.add_nodes_from(grid.nodes)
    # Draw in a fixed (sorted) order so the seed alone pins down the instance.
    for arc in sorted(grid.edges, key=str):
        graph.add_edge(
            *arc,
            travel_time=float(rng.uniform(low, high)),
            capacity=float(rng.uniform(1.0, 5.0)),
            discomfort=float(rng.uniform(0.0, 1.0)),
            hazard=float(rng.uniform(0.0, 1.0)),
            cost=float(rng.uniform(0.0, 1.0)),
            instrumented=bool(rng.random() < instrumented_fraction),
        )
    for node in sorted(graph.nodes, key=str):
        graph.nodes[node]["policing"] = float(rng.random())

    network = InfrastructureGraph.from_networkx(graph)

    individuals: list[Individual] = []
    for (origin, destination), count in sorted(demands.items(), key=str):
        if origin not in graph.nodes or destination not in graph.nodes:
            raise ValueError(f"Demand {origin!r} -> {destination!r} is off the grid.")
        if count < 0:
            raise ValueError("Demand counts must be non-negative.")
        for k in range(count):
            individuals.append(
                Individual(f"{origin}->{destination}#{k}", Demand(origin, destination))
            )

    return World(network, individuals)
