from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "final_av_infra.py"
SPEC = importlib.util.spec_from_file_location("final_av_infra", MODULE_PATH)
assert SPEC is not None
final_av_infra = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = final_av_infra
SPEC.loader.exec_module(final_av_infra)

METRICS = final_av_infra.METRICS
NetworkBelief = final_av_infra.NetworkBelief
make_grid_world = final_av_infra.make_grid_world
setup = final_av_infra.setup


def _demo_demand() -> list[dict]:
    return [
        {
            "departure_time": 1,
            "origin": (0, 0),
            "destination": (2, 2),
            "driver": "human",
        },
        {
            "departure_time": 3,
            "origin": (0, 0),
            "destination": (1, 1),
            "driver": "av",
        },
    ]


def test_make_grid_world_creates_directed_metric_graph() -> None:
    graph = make_grid_world(n_side=3, seed=1)

    assert graph.is_directed()
    assert len(graph.nodes) == 9
    assert graph.has_edge((0, 0), (0, 1))
    assert graph.has_edge((0, 1), (0, 0))

    for _, _, data in graph.edges(data=True):
        assert all(metric in data for metric in METRICS)
        assert "CAPACITY" in data
        assert data["TRAVEL_TIME"] > 0
        assert data["CAPACITY"] > 0


def test_setup_builds_world_and_historical_prior() -> None:
    graph = make_grid_world(n_side=3, seed=2)
    world = setup(
        graph,
        _demo_demand(),
        max_timesteps=12,
        n_scenarios=8,
        k_routes=3,
        seed=2,
    )

    assert len(world.receivers) == 2
    assert world.prior.states.shape == (8, 12, len(graph.edges), len(METRICS))
    assert set(world.network._vehicles) == set(world.prior.edge_list)
    assert all(receiver.routes for receiver in world.receivers)


def test_truthful_signal_updates_to_normalized_belief() -> None:
    graph = make_grid_world(n_side=3, seed=3)
    world = setup(
        graph,
        _demo_demand(),
        max_timesteps=12,
        n_scenarios=8,
        k_routes=3,
        seed=3,
    )
    receiver = world.receivers[0]

    signal = world.sender.send_signal(
        network=world.network,
        prior=world.prior,
        receiver_type=receiver.type,
    )
    belief = NetworkBelief.from_signal(
        prior=world.prior,
        timestep=world.timer.time,
        signal=signal,
        sender_policy=world.sender.signal_policy,
        receiver_type=receiver.type,
    )

    assert signal
    assert np.isclose(belief.weights.sum(), 1.0)
    assert np.all(belief.weights >= 0.0)


def test_world_play_day_runs_to_completion() -> None:
    graph = make_grid_world(n_side=3, seed=4)
    world = setup(
        graph,
        _demo_demand(),
        max_timesteps=20,
        n_scenarios=10,
        k_routes=3,
        seed=4,
    )

    outcome = world.play_day(max_timesteps=20)

    assert outcome["N_STARTED"] == 2
    assert outcome["N_FINISHED"] == 2
    assert outcome["TOTAL_TRAVEL_TIME"] > 0
    assert outcome["TOTAL_COST"] >= 0
    assert outcome["MAX_HAZARD"] >= 0
