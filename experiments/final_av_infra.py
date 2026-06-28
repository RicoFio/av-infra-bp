from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import networkx as nx
import numpy as np

from mcr.avinfra_persuasion.orders.partial_order import PartialOrder


Node = tuple[int, int]
Edge = tuple[Node, Node]
Route = tuple[Edge, ...]
Metric = str

METRICS: tuple[Metric, ...] = ("TRAVEL_TIME", "HAZARD", "COST")
METRIC_TO_IDX = {metric: idx for idx, metric in enumerate(METRICS)}


def make_grid_world(n_side: int = 5, seed: int = 0) -> nx.DiGraph:
    """
    Build a small directed grid with the edge attributes used by this demo.

    Each undirected grid edge is represented in both directions. Attribute
    values are deterministic for a given seed so experiments and tests are
    reproducible.
    """
    if n_side < 2:
        raise ValueError("n_side must be at least 2.")

    rng = np.random.default_rng(seed)
    undirected = nx.grid_2d_graph(n_side, n_side)
    graph = nx.DiGraph()
    graph.add_nodes_from(undirected.nodes)

    for u, v in sorted(undirected.edges, key=str):
        for tail, head in ((u, v), (v, u)):
            graph.add_edge(
                tail,
                head,
                TRAVEL_TIME=float(1.0 + rng.uniform(0.0, 0.75)),
                CAPACITY=float(rng.integers(2, 6)),
                HAZARD=float(rng.uniform(0.0, 2.0)),
                COST=float(rng.uniform(0.0, 2.0)),
            )

    return graph


@dataclass
class Timer:
    time: int = 0

    def step(self) -> None:
        self.time += 1

    def reset(self) -> None:
        self.time = 0


@dataclass
class Network:
    graph: nx.DiGraph
    _vehicles: dict[Edge, int]
    bpr_alpha: float
    bpr_beta: float
    toll_min: float
    toll_max: float
    no_toll_ratio: float = 1.0
    toll_saturation_ratio: float = 2.0

    def __post_init__(self) -> None:
        _validate_graph_metrics(self.graph)

        if self.toll_min < 0:
            raise ValueError("toll_min must be nonnegative.")
        if self.toll_max < self.toll_min:
            raise ValueError("toll_max must be greater than or equal to toll_min.")
        if self.toll_saturation_ratio <= self.no_toll_ratio:
            raise ValueError("toll_saturation_ratio must be greater than no_toll_ratio.")

        missing_edges = set(_edge_list(self.graph)) - set(self._vehicles)
        if missing_edges:
            raise ValueError(f"Missing vehicle counters for edges: {missing_edges!r}")

    def register_vehicle(self, edge: Edge) -> None:
        self._vehicles[edge] += 1

    def deregister_vehicle(self, edge: Edge) -> None:
        if self._vehicles[edge] <= 0:
            raise ValueError(f"Cannot deregister vehicle from empty edge {edge!r}.")
        self._vehicles[edge] -= 1

    def reset_vehicles(self) -> None:
        for edge in self._vehicles:
            self._vehicles[edge] = 0

    def _edge_data(self, edge: Edge) -> dict[str, float]:
        return self.graph.edges[edge]

    def _bpr_function(self, t0: float, volume: int, capacity: float) -> float:
        ratio = volume / capacity
        return float(t0 * (1 + self.bpr_alpha * ratio**self.bpr_beta))

    def _bpr_toll(self, volume: float, capacity: float) -> float:
        ratio = volume / capacity
        if ratio <= self.no_toll_ratio:
            return 0.0

        numerator = ratio**self.bpr_beta - self.no_toll_ratio**self.bpr_beta
        denominator = (
            self.toll_saturation_ratio**self.bpr_beta
            - self.no_toll_ratio**self.bpr_beta
        )
        scaled_congestion = numerator / denominator
        toll = self.toll_min + (self.toll_max - self.toll_min) * scaled_congestion
        return float(np.clip(toll, self.toll_min, self.toll_max))

    def _get_congestion_tt(self, edge: Edge) -> float:
        edge_data = self._edge_data(edge)
        return self._bpr_function(
            t0=float(edge_data["TRAVEL_TIME"]),
            volume=self._vehicles[edge],
            capacity=float(edge_data["CAPACITY"]),
        )

    def _get_congestion_hazard(self, edge: Edge) -> float:
        edge_data = self._edge_data(edge)
        return float(
            edge_data["HAZARD"]
            + np.floor(self._vehicles[edge] / float(edge_data["CAPACITY"]))
        )

    def _get_toll(self, edge: Edge) -> float:
        edge_data = self._edge_data(edge)
        return self._bpr_toll(
            volume=self._vehicles[edge],
            capacity=float(edge_data["CAPACITY"]),
        )

    def _get_congestion_cost(self, edge: Edge) -> float:
        return float(self._edge_data(edge)["COST"] + self._get_toll(edge))

    def get_current_metrics(self, edge: Edge) -> dict[Metric, float]:
        return {
            "TRAVEL_TIME": self._get_congestion_tt(edge),
            "HAZARD": self._get_congestion_hazard(edge),
            "COST": self._get_congestion_cost(edge),
        }

    def get_route_metrics(self, route: Route) -> dict[Metric, float]:
        edge_metrics = [self.get_current_metrics(edge) for edge in route]
        return {
            "TRAVEL_TIME": sum(m["TRAVEL_TIME"] for m in edge_metrics),
            "HAZARD": max((m["HAZARD"] for m in edge_metrics), default=0.0),
            "COST": sum(m["COST"] for m in edge_metrics),
        }

    def get_state_array(self, edge_list: list[Edge]) -> np.ndarray:
        state = np.zeros((len(edge_list), len(METRICS)))
        for edge_idx, edge in enumerate(edge_list):
            current = self.get_current_metrics(edge)
            for metric_idx, metric in enumerate(METRICS):
                state[edge_idx, metric_idx] = current[metric]
        return state


@dataclass
class Prior:
    pass


@dataclass
class DynamicHistoricalPrior(Prior):
    """
    Empirical prior over network states from baseline simulated days.

    states[k, t, e, m] is the metric value for scenario k, timer step t, edge e,
    and metric m. Receivers use the slice at their departure time.
    """

    states: np.ndarray
    edge_list: list[Edge]
    n_bins: int = 3
    weights: np.ndarray | None = None

    edge_to_idx: dict[Edge, int] = field(init=False)
    bins: dict[Metric, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=float)
        if self.states.ndim != 4:
            raise ValueError("states must have shape scenarios x timesteps x edges x metrics.")
        if self.states.shape[2] != len(self.edge_list):
            raise ValueError("states edge dimension must match edge_list.")
        if self.states.shape[3] != len(METRICS):
            raise ValueError("states metric dimension must match METRICS.")
        if self.n_bins < 1:
            raise ValueError("n_bins must be positive.")

        self.n_scenarios = int(self.states.shape[0])
        self.max_timesteps = int(self.states.shape[1])
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edge_list)}

        if self.weights is None:
            self.weights = np.ones(self.n_scenarios) / self.n_scenarios
        else:
            self.weights = np.asarray(self.weights, dtype=float)
            if self.weights.shape != (self.n_scenarios,):
                raise ValueError("weights must have one value per scenario.")
            total = float(self.weights.sum())
            if total <= 0:
                raise ValueError("weights must sum to a positive value.")
            self.weights = self.weights / total

        self.bins = self._make_bins()

    def _t(self, timestep: int) -> int:
        return min(max(int(timestep), 0), self.max_timesteps - 1)

    def _make_bins(self) -> dict[Metric, np.ndarray]:
        bins: dict[Metric, np.ndarray] = {}
        for metric, metric_idx in METRIC_TO_IDX.items():
            values = self.states[..., metric_idx].reshape(-1)
            probabilities = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            bins[metric] = np.quantile(values, probabilities)
        return bins

    def bin_index(self, metric: Metric, value: float) -> int:
        cuts = self.bins[metric]
        return int(
            np.clip(
                np.searchsorted(cuts, value, side="right"),
                0,
                self.n_bins - 1,
            )
        )

    def metric_value(
        self,
        scenario_idx: int,
        timestep: int,
        edge: Edge,
        metric: Metric,
    ) -> float:
        t = self._t(timestep)
        edge_idx = self.edge_to_idx[edge]
        metric_idx = METRIC_TO_IDX[metric]
        return float(self.states[scenario_idx, t, edge_idx, metric_idx])

    def route_metrics(
        self,
        scenario_idx: int,
        timestep: int,
        route: Route,
    ) -> dict[Metric, float]:
        totals = {metric: 0.0 for metric in METRICS}
        for edge in route:
            totals["TRAVEL_TIME"] += self.metric_value(
                scenario_idx, timestep, edge, "TRAVEL_TIME"
            )
            totals["HAZARD"] = max(
                totals["HAZARD"],
                self.metric_value(scenario_idx, timestep, edge, "HAZARD"),
            )
            totals["COST"] += self.metric_value(scenario_idx, timestep, edge, "COST")
        return totals


@dataclass
class NetworkBelief:
    prior: DynamicHistoricalPrior
    timestep: int
    weights: np.ndarray

    @staticmethod
    def from_prior(prior: DynamicHistoricalPrior, timestep: int) -> "NetworkBelief":
        return NetworkBelief(
            prior=prior,
            timestep=timestep,
            weights=prior.weights.copy(),
        )

    @staticmethod
    def from_signal(
        prior: DynamicHistoricalPrior,
        timestep: int,
        signal: dict[Edge, dict[Metric, int]],
        sender_policy: "SignalPolicy",
        receiver_type: str,
    ) -> "NetworkBelief":
        likelihoods = np.array(
            [
                sender_policy.likelihood(
                    prior=prior,
                    timestep=timestep,
                    signal=signal,
                    scenario_idx=scenario_idx,
                    receiver_type=receiver_type,
                )
                for scenario_idx in range(prior.n_scenarios)
            ],
            dtype=float,
        )
        unnormalized = prior.weights * likelihoods
        normalizer = float(unnormalized.sum())

        if normalizer <= 0:
            weights = prior.weights.copy()
        else:
            weights = unnormalized / normalizer

        return NetworkBelief(prior=prior, timestep=timestep, weights=weights)

    def expected_route_metrics(self, route: Route) -> dict[Metric, float]:
        expected = {metric: 0.0 for metric in METRICS}
        for scenario_idx, weight in enumerate(self.weights):
            route_metrics = self.prior.route_metrics(
                scenario_idx=scenario_idx,
                timestep=self.timestep,
                route=route,
            )
            for metric in METRICS:
                expected[metric] += float(weight) * route_metrics[metric]
        return expected


@dataclass
class SignalPolicy:
    """
    Public truthful binned signal policy.

    The sender reports the true bin for each metric on each owned edge. The
    likelihood is the corresponding deterministic public observation model.
    """

    owned_edges: set[Edge]
    metrics: tuple[Metric, ...] = METRICS

    def sample_signal(
        self,
        network: Network,
        prior: DynamicHistoricalPrior,
        receiver_type: str,
    ) -> dict[Edge, dict[Metric, int]]:
        signal: dict[Edge, dict[Metric, int]] = {}
        for edge in sorted(self.owned_edges, key=str):
            current = network.get_current_metrics(edge)
            signal[edge] = {
                metric: prior.bin_index(metric, current[metric])
                for metric in self.metrics
            }
        return signal

    def likelihood(
        self,
        prior: DynamicHistoricalPrior,
        timestep: int,
        signal: dict[Edge, dict[Metric, int]],
        scenario_idx: int,
        receiver_type: str,
    ) -> float:
        for edge, reported_metrics in signal.items():
            for metric, reported_bin in reported_metrics.items():
                value = prior.metric_value(
                    scenario_idx=scenario_idx,
                    timestep=timestep,
                    edge=edge,
                    metric=metric,
                )
                if prior.bin_index(metric, value) != reported_bin:
                    return 0.0
        return 1.0


@dataclass
class Sender:
    preference: PartialOrder
    signal_policy: SignalPolicy
    timer: Timer = field(default_factory=Timer)

    def send_signal(
        self,
        network: Network,
        prior: DynamicHistoricalPrior,
        receiver_type: str,
    ) -> dict[Edge, dict[Metric, int]]:
        return self.signal_policy.sample_signal(
            network=network,
            prior=prior,
            receiver_type=receiver_type,
        )


@dataclass
class Receiver:
    preference: PartialOrder
    driver: str
    sender: Sender
    origin: Node
    destination: Node
    departure_time: int
    routes: dict[str, Route]
    prior: DynamicHistoricalPrior
    timer: Timer = field(default_factory=Timer)

    chosen_route: str | None = None
    belief: NetworkBelief | None = None
    active_edge: Edge | None = None
    route_position: int = 0
    remaining_time: int = 0
    finished: bool = False
    experienced_metrics: dict[Metric, float] = field(
        default_factory=lambda: {metric: 0.0 for metric in METRICS}
    )

    def __post_init__(self) -> None:
        unknown_metrics = set(self.preference.elements) - set(METRICS)
        if unknown_metrics:
            raise ValueError(f"Unknown preference metrics: {unknown_metrics!r}")
        if not self.routes:
            raise ValueError("Receiver must have at least one route candidate.")

    @property
    def type(self) -> str:
        return f"{self.driver}:{self.origin}->{self.destination}"

    def reset_for_day(self) -> None:
        self.chosen_route = None
        self.belief = None
        self.active_edge = None
        self.route_position = 0
        self.remaining_time = 0
        self.finished = False
        self.experienced_metrics = {metric: 0.0 for metric in METRICS}

    def update_belief(
        self,
        signal: dict[Edge, dict[Metric, int]],
        sender_policy: SignalPolicy,
    ) -> None:
        self.belief = NetworkBelief.from_signal(
            prior=self.prior,
            timestep=self.timer.time,
            signal=signal,
            sender_policy=sender_policy,
            receiver_type=self.type,
        )

    def choose_route(self) -> str:
        if self.chosen_route is not None:
            return self.chosen_route

        if self.belief is None:
            self.belief = NetworkBelief.from_prior(
                prior=self.prior,
                timestep=self.timer.time,
            )

        scored_routes = {
            label: self.belief.expected_route_metrics(route)
            for label, route in self.routes.items()
        }
        self.chosen_route = min(
            scored_routes,
            key=lambda label: (
                _preference_key(self.preference, scored_routes[label]),
                _preference_key(self.sender.preference, scored_routes[label]),
                label,
            ),
        )
        return self.chosen_route

    def enter_network(self, network: Network) -> None:
        if self.chosen_route is None:
            raise RuntimeError("Receiver must choose a route before entering.")
        self.route_position = 0
        self.finished = False
        self._enter_next_edge(network)

    def _enter_next_edge(self, network: Network) -> None:
        route = self.routes[self.chosen_route]
        if self.route_position >= len(route):
            self.finished = True
            self.active_edge = None
            return

        edge = route[self.route_position]
        self.route_position += 1
        network.register_vehicle(edge)
        self.active_edge = edge

        current = network.get_current_metrics(edge)
        self.remaining_time = max(1, int(np.ceil(current["TRAVEL_TIME"])))
        self.experienced_metrics["TRAVEL_TIME"] += current["TRAVEL_TIME"]
        self.experienced_metrics["COST"] += current["COST"]
        self.experienced_metrics["HAZARD"] = max(
            self.experienced_metrics["HAZARD"],
            current["HAZARD"],
        )

    def step(self, network: Network) -> None:
        if self.finished or self.active_edge is None:
            return

        self.remaining_time -= 1
        if self.remaining_time <= 0:
            network.deregister_vehicle(self.active_edge)
            self.active_edge = None
            self._enter_next_edge(network)


@dataclass
class _BaselineVehicle:
    route: Route
    active_edge: Edge | None = None
    route_position: int = 0
    remaining_time: int = 0
    finished: bool = False

    def enter_network(self, network: Network) -> None:
        self.route_position = 0
        self.finished = False
        self._enter_next_edge(network)

    def _enter_next_edge(self, network: Network) -> None:
        if self.route_position >= len(self.route):
            self.finished = True
            self.active_edge = None
            return

        edge = self.route[self.route_position]
        self.route_position += 1
        network.register_vehicle(edge)
        self.active_edge = edge
        self.remaining_time = max(
            1,
            int(np.ceil(network.get_current_metrics(edge)["TRAVEL_TIME"])),
        )

    def step(self, network: Network) -> None:
        if self.finished or self.active_edge is None:
            return

        self.remaining_time -= 1
        if self.remaining_time <= 0:
            network.deregister_vehicle(self.active_edge)
            self.active_edge = None
            self._enter_next_edge(network)


@dataclass
class World:
    sender: Sender
    receivers: list[Receiver]
    network: Network
    prior: DynamicHistoricalPrior
    timer: Timer = field(default_factory=Timer)

    def __post_init__(self) -> None:
        self.sender.timer = self.timer
        for receiver in self.receivers:
            receiver.timer = self.timer

    def departing_receivers(self) -> list[Receiver]:
        return [
            receiver
            for receiver in self.receivers
            if receiver.departure_time == self.timer.time
            and receiver.chosen_route is None
        ]

    def active_receivers(self) -> list[Receiver]:
        return [
            receiver
            for receiver in self.receivers
            if receiver.active_edge is not None and not receiver.finished
        ]

    def step(self) -> dict[str, float | int]:
        for receiver in self.departing_receivers():
            signal = self.sender.send_signal(
                network=self.network,
                prior=self.prior,
                receiver_type=receiver.type,
            )
            receiver.update_belief(
                signal=signal,
                sender_policy=self.sender.signal_policy,
            )
            receiver.choose_route()
            receiver.enter_network(self.network)

        for receiver in self.active_receivers():
            receiver.step(self.network)

        outcome = self.evaluate_world()
        self.timer.step()
        return outcome

    def play_day(self, max_timesteps: int) -> dict[str, float | int]:
        self.timer.reset()
        self.network.reset_vehicles()
        for receiver in self.receivers:
            receiver.reset_for_day()

        while self.timer.time < max_timesteps:
            self.step()

        return self.evaluate_world()

    def evaluate_world(self) -> dict[str, float | int]:
        started = [
            receiver
            for receiver in self.receivers
            if receiver.chosen_route is not None
        ]
        return {
            "TOTAL_TRAVEL_TIME": sum(
                receiver.experienced_metrics["TRAVEL_TIME"]
                for receiver in started
            ),
            "TOTAL_COST": sum(
                receiver.experienced_metrics["COST"]
                for receiver in started
            ),
            "MAX_HAZARD": max(
                (receiver.experienced_metrics["HAZARD"] for receiver in started),
                default=0.0,
            ),
            "N_STARTED": len(started),
            "N_FINISHED": sum(receiver.finished for receiver in started),
        }


def build_route_candidates(
    graph: nx.DiGraph,
    origin: Node,
    destination: Node,
    k_routes: int = 5,
) -> dict[str, Route]:
    if k_routes < 1:
        raise ValueError("k_routes must be positive.")
    if origin == destination:
        return {"route_0": tuple()}

    routes: list[Route] = []
    seen: set[Route] = set()

    for metric in METRICS:
        try:
            node_paths = nx.shortest_simple_paths(
                graph,
                source=origin,
                target=destination,
                weight=metric,
            )
            for node_path in islice(node_paths, k_routes):
                route = _node_path_to_route(node_path)
                if route not in seen:
                    seen.add(route)
                    routes.append(route)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if not routes:
        raise ValueError(f"No route candidates found from {origin!r} to {destination!r}.")

    return {f"route_{idx}": route for idx, route in enumerate(routes)}


def build_historical_prior_from_random_route_selection(
    graph: nx.DiGraph,
    receiver_specs: list[dict[str, Any]],
    max_timesteps: int,
    n_scenarios: int,
    bpr_alpha: float,
    bpr_beta: float,
    toll_min: float,
    toll_max: float,
    n_bins: int = 3,
    seed: int | None = None,
) -> DynamicHistoricalPrior:
    rng = np.random.default_rng(seed)
    edge_list = _edge_list(graph)
    states = np.zeros((n_scenarios, max_timesteps, len(edge_list), len(METRICS)))

    for scenario_idx in range(n_scenarios):
        network = _make_network(
            graph=graph,
            edge_list=edge_list,
            bpr_alpha=bpr_alpha,
            bpr_beta=bpr_beta,
            toll_min=toll_min,
            toll_max=toll_max,
        )
        active: list[_BaselineVehicle] = []

        for timestep in range(max_timesteps):
            states[scenario_idx, timestep, :, :] = network.get_state_array(edge_list)

            for spec in receiver_specs:
                if int(spec["departure_time"]) != timestep:
                    continue
                routes = spec["routes"]
                route_names = sorted(routes)
                chosen_name = str(rng.choice(route_names))
                vehicle = _BaselineVehicle(route=routes[chosen_name])
                vehicle.enter_network(network)
                active.append(vehicle)

            for vehicle in active:
                vehicle.step(network)

            active = [vehicle for vehicle in active if not vehicle.finished]

    return DynamicHistoricalPrior(
        states=states,
        edge_list=edge_list,
        n_bins=n_bins,
    )


def setup(
    graph: nx.DiGraph,
    demand: list[dict[str, Any]],
    *,
    max_timesteps: int = 30,
    n_scenarios: int = 50,
    n_bins: int = 3,
    k_routes: int = 5,
    seed: int = 0,
) -> World:
    _validate_graph_metrics(graph)
    edge_list = _edge_list(graph)

    routes_by_od: dict[tuple[Node, Node], dict[str, Route]] = {}
    for spec in demand:
        od = spec["origin"], spec["destination"]
        if od not in routes_by_od:
            routes_by_od[od] = build_route_candidates(
                graph=graph,
                origin=spec["origin"],
                destination=spec["destination"],
                k_routes=k_routes,
            )

    receiver_specs = [
        {
            "departure_time": int(spec["departure_time"]),
            "routes": routes_by_od[(spec["origin"], spec["destination"])],
        }
        for spec in demand
    ]
    prior = build_historical_prior_from_random_route_selection(
        graph=graph,
        receiver_specs=receiver_specs,
        max_timesteps=max_timesteps,
        n_scenarios=n_scenarios,
        bpr_alpha=0.15,
        bpr_beta=4.0,
        toll_min=1.0,
        toll_max=10.0,
        n_bins=n_bins,
        seed=seed,
    )

    timer = Timer()
    sender = Sender(
        preference=_sender_preference(),
        signal_policy=SignalPolicy(owned_edges=set(edge_list)),
        timer=timer,
    )
    receivers = [
        Receiver(
            preference=_driver_preference(str(spec["driver"])),
            driver=str(spec["driver"]),
            sender=sender,
            origin=spec["origin"],
            destination=spec["destination"],
            departure_time=int(spec["departure_time"]),
            routes=routes_by_od[(spec["origin"], spec["destination"])],
            prior=prior,
            timer=timer,
        )
        for spec in demand
    ]
    network = _make_network(
        graph=graph,
        edge_list=edge_list,
        bpr_alpha=0.15,
        bpr_beta=4.0,
        toll_min=1.0,
        toll_max=10.0,
    )

    return World(
        sender=sender,
        receivers=receivers,
        network=network,
        prior=prior,
        timer=timer,
    )


def _driver_preference(driver: str) -> PartialOrder:
    if driver == "human":
        return PartialOrder(
            elements=set(METRICS),
            relations={
                ("COST", "TRAVEL_TIME"),
                ("HAZARD", "TRAVEL_TIME"),
            },
        )
    if driver == "av":
        return PartialOrder(
            elements=set(METRICS),
            relations={
                ("COST", "HAZARD"),
                ("TRAVEL_TIME", "HAZARD"),
            },
        )
    raise ValueError(f"Unknown driver type: {driver!r}")


def _sender_preference() -> PartialOrder:
    return PartialOrder(elements={"TRAVEL_TIME"}, relations=set())


def _preference_key(preference: PartialOrder, metrics: dict[Metric, float]) -> tuple[float, ...]:
    key: list[float] = []
    for layer in _preference_layers(preference):
        key.extend(metrics[metric] for metric in layer)
    return tuple(key)


def _preference_layers(preference: PartialOrder) -> tuple[tuple[Metric, ...], ...]:
    remaining = set(preference.elements)
    layers: list[tuple[Metric, ...]] = []

    while remaining:
        sub_preference = preference.build_sub_preorder(set(remaining))
        layer = tuple(
            sorted(
                sub_preference.maximal_elements(),
                key=lambda metric: METRIC_TO_IDX[metric],
            )
        )
        layers.append(layer)
        remaining.difference_update(layer)

    return tuple(layers)


def _make_network(
    graph: nx.DiGraph,
    edge_list: list[Edge],
    bpr_alpha: float,
    bpr_beta: float,
    toll_min: float,
    toll_max: float,
) -> Network:
    return Network(
        graph=graph,
        _vehicles={edge: 0 for edge in edge_list},
        bpr_alpha=bpr_alpha,
        bpr_beta=bpr_beta,
        toll_min=toll_min,
        toll_max=toll_max,
    )


def _edge_list(graph: nx.DiGraph) -> list[Edge]:
    return list(sorted(graph.edges, key=str))


def _node_path_to_route(node_path: list[Node]) -> Route:
    return tuple(zip(node_path, node_path[1:]))


def _validate_graph_metrics(graph: nx.DiGraph) -> None:
    missing: list[tuple[Edge, str]] = []
    for u, v, data in graph.edges(data=True):
        for attr in (*METRICS, "CAPACITY"):
            if attr not in data:
                missing.append(((u, v), attr))
    if missing:
        raise ValueError(f"Graph edges are missing required attributes: {missing!r}")


if __name__ == "__main__":
    demo_graph = make_grid_world(n_side=5, seed=0)
    demo_demand = [
        {
            "departure_time": 4,
            "origin": (0, 0),
            "destination": (4, 4),
            "driver": "human",
        },
        {
            "departure_time": 10,
            "origin": (0, 0),
            "destination": (2, 2),
            "driver": "av",
        },
    ]

    world = setup(demo_graph, demo_demand, seed=123)
    print(world.play_day(max_timesteps=30))
