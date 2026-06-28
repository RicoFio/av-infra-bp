"""
Wu/Amin two-route information-design benchmark.

This experiment intentionally keeps the Wu/Amin Wardrop equilibrium formulas
local to this file. The OSMR part that is being exercised is the state-dependent
mask policy plus the finite-difference optimizer over that policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeAlias

from ..bp.game import Preference
from ..bp.receivers import Receiver
from ..bp.senders import Objective, ScalarSender
from ..bp.signals import StateDependentMaskSignalPolicy
from ..datastructures import (
    Arc,
    Demand,
    FinitePrior,
    Individual,
    InfrastructureGraph,
    MetricName,
    Scenario,
    World,
)
from .games.osmrsp import OSMRSPGame

ACCIDENT_STATE = "accident"
NO_ACCIDENT_STATE = "no_accident"
ACCIDENT_SIGNAL = "a"
NO_ACCIDENT_SIGNAL = "n"

WuAminSignal: TypeAlias = Literal["a", "n"]
WuAminRegime: TypeAlias = Literal["L1", "L2", "L3"]

EMPTY_MASK = frozenset()
HAZARD_MASK = frozenset({MetricName.HAZARD})


@dataclass(frozen=True)
class WuAminParameters:
    """Parameters from the Wu/Amin numerical example."""

    alpha_1_a: float = 3.0
    alpha_1_n: float = 1.0
    alpha_2: float = 2.0
    p: float = 0.3
    b_1: float = 15.0
    b_2: float = 20.0
    pop_lambda: float = 0.5
    D: float = 10.0
    tau: float = 2.5

    def __post_init__(self) -> None:
        if not 0.0 < self.p < 1.0:
            raise ValueError("p must lie in (0, 1).")
        if not 0.0 <= self.pop_lambda <= 1.0:
            raise ValueError("pop_lambda must lie in [0, 1].")
        if self.D <= 0.0:
            raise ValueError("D must be positive.")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive.")
        if not self.b_1 < self.b_2:
            raise ValueError("Wu/Amin requires b_1 < b_2.")
        if not self.alpha_1_a > self.alpha_2 > self.alpha_1_n:
            raise ValueError("Wu/Amin requires alpha_1_a > alpha_2 > alpha_1_n.")

    @property
    def cost_diff(self) -> float:
        return self.alpha_2 * self.D + self.b_2 - self.b_1

    @property
    def alpha_1_top_theta(self) -> float:
        return self.p * self.alpha_1_a + (1.0 - self.p) * self.alpha_1_n

    @property
    def lambda_bottom(self) -> float:
        numerator = (self.D - self.tau) * (
            self.alpha_1_top_theta + self.alpha_2
        ) - self.cost_diff
        denominator = self.D * self.p * (self.alpha_1_a + self.alpha_2)
        return numerator / denominator

    @property
    def lambda_top(self) -> float:
        return (
            1.0
            - self.cost_diff / ((self.alpha_1_a + self.alpha_2) * self.D)
            - self.tau / self.D
        )

    @property
    def regime(self) -> WuAminRegime:
        if 0.0 <= self.pop_lambda < self.lambda_bottom:
            return "L1"
        if self.lambda_bottom <= self.pop_lambda < self.lambda_top:
            return "L2"
        if self.lambda_top <= self.pop_lambda <= 1.0:
            return "L3"
        raise ValueError("pop_lambda is outside the Wu/Amin regime partition.")

    def theta(self, omega: WuAminSignal) -> float:
        return self.p if omega == ACCIDENT_SIGNAL else 1.0 - self.p

    def pi_star(self, signal: WuAminSignal, omega: WuAminSignal) -> float:
        if self.regime == "L1":
            return 1.0 if signal == omega else 0.0

        if signal == NO_ACCIDENT_SIGNAL and omega == NO_ACCIDENT_SIGNAL:
            return 1.0
        if signal == ACCIDENT_SIGNAL and omega == NO_ACCIDENT_SIGNAL:
            return 0.0

        if self.regime == "L2":
            pi_a_given_a = (
                (self.D - self.tau) * (self.alpha_1_top_theta + self.alpha_2)
                - self.cost_diff
            ) / (
                self.pop_lambda
                * self.D
                * self.p
                * (self.alpha_1_a + self.alpha_2)
            )
        else:
            pi_a_given_a = (
                (self.D - self.tau) * (self.alpha_1_top_theta + self.alpha_2)
                - self.cost_diff
            ) / (
                (
                    (self.D - self.tau) * (self.alpha_1_a + self.alpha_2)
                    - self.cost_diff
                )
                * self.p
            )

        if signal == ACCIDENT_SIGNAL and omega == ACCIDENT_SIGNAL:
            return pi_a_given_a
        if signal == NO_ACCIDENT_SIGNAL and omega == ACCIDENT_SIGNAL:
            return 1.0 - pi_a_given_a
        raise ValueError(f"Invalid signal/state pair: {signal!r}, {omega!r}.")

    def reference_policy(self) -> dict[str, dict[frozenset[MetricName], float]]:
        return {
            ACCIDENT_STATE: {
                HAZARD_MASK: self.pi_star(ACCIDENT_SIGNAL, ACCIDENT_SIGNAL),
                EMPTY_MASK: self.pi_star(NO_ACCIDENT_SIGNAL, ACCIDENT_SIGNAL),
            },
            NO_ACCIDENT_STATE: {
                HAZARD_MASK: self.pi_star(ACCIDENT_SIGNAL, NO_ACCIDENT_SIGNAL),
                EMPTY_MASK: self.pi_star(NO_ACCIDENT_SIGNAL, NO_ACCIDENT_SIGNAL),
            },
        }

    @property
    def reference_spillover(self) -> float:
        pi_a_given_a = self.pi_star(ACCIDENT_SIGNAL, ACCIDENT_SIGNAL)
        pi_a_given_n = self.pi_star(ACCIDENT_SIGNAL, NO_ACCIDENT_SIGNAL)
        return self.expected_spillover(
            pi_a_given_a=pi_a_given_a,
            pi_a_given_n=pi_a_given_n,
        )

    def expected_spillover(
        self,
        *,
        pi_a_given_a: float,
        pi_a_given_n: float,
    ) -> float:
        return sum(
            self.signal_probability(
                signal,
                pi_a_given_a=pi_a_given_a,
                pi_a_given_n=pi_a_given_n,
            )
            * max(
                self.wardrop_route_2_flow(
                    signal,
                    pi_a_given_a=pi_a_given_a,
                    pi_a_given_n=pi_a_given_n,
                )
                - self.tau,
                0.0,
            )
            for signal in (ACCIDENT_SIGNAL, NO_ACCIDENT_SIGNAL)
        )

    def signal_probability(
        self,
        signal: WuAminSignal,
        *,
        pi_a_given_a: float,
        pi_a_given_n: float,
    ) -> float:
        if signal == ACCIDENT_SIGNAL:
            return self.p * pi_a_given_a + (1.0 - self.p) * pi_a_given_n
        return self.p * (1.0 - pi_a_given_a) + (1.0 - self.p) * (
            1.0 - pi_a_given_n
        )

    def posterior_accident_probability(
        self,
        signal: WuAminSignal,
        *,
        pi_a_given_a: float,
        pi_a_given_n: float,
    ) -> float:
        signal_mass = self.signal_probability(
            signal,
            pi_a_given_a=pi_a_given_a,
            pi_a_given_n=pi_a_given_n,
        )
        if signal_mass <= 0.0:
            return self.p
        if signal == ACCIDENT_SIGNAL:
            return self.p * pi_a_given_a / signal_mass
        return self.p * (1.0 - pi_a_given_a) / signal_mass

    def alpha_1_top_beta(
        self,
        signal: WuAminSignal,
        *,
        pi_a_given_a: float,
        pi_a_given_n: float,
    ) -> float:
        beta = self.posterior_accident_probability(
            signal,
            pi_a_given_a=pi_a_given_a,
            pi_a_given_n=pi_a_given_n,
        )
        return self.alpha_1_n + beta * (self.alpha_1_a - self.alpha_1_n)

    def g_pi(
        self,
        *,
        pi_a_given_a: float,
        pi_a_given_n: float,
    ) -> float:
        alpha_n = self.alpha_1_top_beta(
            NO_ACCIDENT_SIGNAL,
            pi_a_given_a=pi_a_given_a,
            pi_a_given_n=pi_a_given_n,
        )
        alpha_a = self.alpha_1_top_beta(
            ACCIDENT_SIGNAL,
            pi_a_given_a=pi_a_given_a,
            pi_a_given_n=pi_a_given_n,
        )
        return self.cost_diff / ((alpha_n + self.alpha_2) * self.D) - (
            self.cost_diff / ((alpha_a + self.alpha_2) * self.D)
        )

    def wardrop_route_2_flow(
        self,
        signal: WuAminSignal,
        *,
        pi_a_given_a: float,
        pi_a_given_n: float,
    ) -> float:
        if self.g_pi(pi_a_given_a=pi_a_given_a, pi_a_given_n=pi_a_given_n) >= (
            self.pop_lambda
        ):
            alpha_a = self.alpha_1_top_beta(
                ACCIDENT_SIGNAL,
                pi_a_given_a=pi_a_given_a,
                pi_a_given_n=pi_a_given_n,
            )
            signal_a_probability = self.signal_probability(
                ACCIDENT_SIGNAL,
                pi_a_given_a=pi_a_given_a,
                pi_a_given_n=pi_a_given_n,
            )
            f_2_n = self.D - (
                self.cost_diff
                + self.pop_lambda
                * self.D
                * signal_a_probability
                * (alpha_a + self.alpha_2)
            ) / (self.alpha_1_top_theta + self.alpha_2)
            if signal == NO_ACCIDENT_SIGNAL:
                return f_2_n
            return f_2_n + self.pop_lambda * self.D

        alpha_signal = self.alpha_1_top_beta(
            signal,
            pi_a_given_a=pi_a_given_a,
            pi_a_given_n=pi_a_given_n,
        )
        return self.D - self.cost_diff / (alpha_signal + self.alpha_2)


@dataclass
class GameZero(OSMRSPGame):
    """
    OSMR state-dependent policy benchmarked against Wu/Amin.

    The inherited solver optimizes mask probabilities. This override evaluates
    those probabilities with the paper's Bayesian Wardrop spillover objective.
    """

    parameters: WuAminParameters
    feasibility_penalty_weight: float = 100.0

    def evaluate_policy(
        self,
        probabilities: Mapping[str, Mapping[frozenset[MetricName], float]] | None = None,
    ) -> dict[str, Any]:
        if probabilities is None:
            probabilities = self.signaling_scheme()

        pi_a_given_a = self._pi_a_given_state(probabilities, ACCIDENT_STATE)
        pi_a_given_n = self._pi_a_given_state(probabilities, NO_ACCIDENT_STATE)
        expected_spillover = self.parameters.expected_spillover(
            pi_a_given_a=pi_a_given_a,
            pi_a_given_n=pi_a_given_n,
        )
        ordering_violation = max(pi_a_given_n - pi_a_given_a, 0.0)
        feasibility_penalty = self.feasibility_penalty_weight * ordering_violation**2
        objective_value = expected_spillover + feasibility_penalty
        breakdown_rows = tuple(
            self._signal_breakdown_row(
                signal,
                pi_a_given_a=pi_a_given_a,
                pi_a_given_n=pi_a_given_n,
            )
            for signal in (ACCIDENT_SIGNAL, NO_ACCIDENT_SIGNAL)
        )

        return {
            "expected_sender_utility": objective_value,
            "expected_sender_metric": expected_spillover,
            "expected_spillover": expected_spillover,
            "feasibility_penalty": feasibility_penalty,
            "ordering_violation": ordering_violation,
            "pi_a_given_a": pi_a_given_a,
            "pi_a_given_n": pi_a_given_n,
            "reference_pi_a_given_a": self.parameters.pi_star(
                ACCIDENT_SIGNAL,
                ACCIDENT_SIGNAL,
            ),
            "reference_pi_a_given_n": self.parameters.pi_star(
                ACCIDENT_SIGNAL,
                NO_ACCIDENT_SIGNAL,
            ),
            "reference_spillover": self.parameters.reference_spillover,
            "reference_regime": self.parameters.regime,
            "breakdown_rows": breakdown_rows,
        }

    def solve(
        self,
        max_iter: int = 400,
        step_size: float = 0.08,
        finite_diff_epsilon: float = 1e-5,
        convergence_tol: float = 1e-10,
        convergence_patience: int = 40,
    ) -> dict[str, Any]:
        return self._solve_with_finite_difference_adam(
            max_iter=max_iter,
            step_size=step_size,
            finite_diff_epsilon=finite_diff_epsilon,
            convergence_tol=convergence_tol,
            convergence_patience=convergence_patience,
            progress=False,
        )

    def _pi_a_given_state(
        self,
        probabilities: Mapping[str, Mapping[frozenset[MetricName], float]],
        state_name: str,
    ) -> float:
        return float(probabilities[state_name][HAZARD_MASK])

    def _signal_breakdown_row(
        self,
        signal: WuAminSignal,
        *,
        pi_a_given_a: float,
        pi_a_given_n: float,
    ) -> dict[str, float | str]:
        signal_probability = self.parameters.signal_probability(
            signal,
            pi_a_given_a=pi_a_given_a,
            pi_a_given_n=pi_a_given_n,
        )
        route_2_flow = self.parameters.wardrop_route_2_flow(
            signal,
            pi_a_given_a=pi_a_given_a,
            pi_a_given_n=pi_a_given_n,
        )
        spillover = max(route_2_flow - self.parameters.tau, 0.0)
        return {
            "signal": signal,
            "signal_probability": signal_probability,
            "posterior_accident_probability": (
                self.parameters.posterior_accident_probability(
                    signal,
                    pi_a_given_a=pi_a_given_a,
                    pi_a_given_n=pi_a_given_n,
                )
            ),
            "route_2_flow": route_2_flow,
            "spillover": spillover,
            "weighted_contribution": signal_probability * spillover,
        }


def build_wu_amin_prior(
    world: World,
    parameters: WuAminParameters,
) -> FinitePrior:
    arc = _single_arc(world)
    common_values = {
        "travel_time": {arc: 0.0},
        "discomfort": {arc: 0.0},
        "cost": {arc: 0.0},
        "emissions": {arc: 0.0},
        "policing": {node: 0.0 for node in world.V},
    }
    accident = Scenario(
        name=ACCIDENT_STATE,
        hazard={arc: 1.0},
        **common_values,
    )
    no_accident = Scenario(
        name=NO_ACCIDENT_STATE,
        hazard={arc: 0.0},
        **common_values,
    )
    return FinitePrior(
        name="wu_amin_accident_prior",
        support={
            ACCIDENT_STATE: accident,
            NO_ACCIDENT_STATE: no_accident,
        },
        probabilities={
            ACCIDENT_STATE: parameters.p,
            NO_ACCIDENT_STATE: 1.0 - parameters.p,
        },
    )


def build_game_zero(
    *,
    parameters: WuAminParameters | None = None,
    pop_lambda: float | None = None,
    seed: int = 1,
    initial_pi_a_given_a: float = 0.5,
    initial_pi_a_given_n: float = 0.5,
) -> GameZero:
    if parameters is None:
        parameters = WuAminParameters(
            pop_lambda=0.5 if pop_lambda is None else pop_lambda
        )
    elif pop_lambda is not None:
        parameters = replace(parameters, pop_lambda=pop_lambda)

    world = _build_placeholder_world()
    prior = build_wu_amin_prior(world, parameters)
    preference = Preference(elements={MetricName.HAZARD}, relations=set())
    signal_policy = StateDependentMaskSignalPolicy(
        seed=seed,
        state_names=frozenset(prior.support),
        considered_metrics=frozenset({MetricName.HAZARD}),
        state_probabilities=_initial_state_probabilities(
            pi_a_given_a=initial_pi_a_given_a,
            pi_a_given_n=initial_pi_a_given_n,
        ),
    )
    sender = ScalarSender(
        prior=prior,
        world=world,
        preference=preference,
        objective=Objective.MINIMIZE,
        signal_policy=signal_policy,
    )
    receiver = Receiver(
        individual=next(iter(world.individuals)),
        rtype="wu_amin_placeholder",
        preference=preference,
        prior=prior,
        world=world,
        sender=sender,
        n_scenarios=len(prior.support),
    )
    return GameZero(
        sender=sender,
        receivers=[receiver],
        world=world,
        public_prior=prior,
        seed=seed,
        parameters=parameters,
    )


def solve_wu_amin_grid(
    pop_lambdas: Sequence[float] = (0.10, 0.20, 0.30, 0.50),
    *,
    seed: int = 1,
    max_iter: int = 400,
    step_size: float = 0.08,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for pop_lambda in pop_lambdas:
        result = solve_wu_amin_case(
            pop_lambda=pop_lambda,
            seed=seed,
            max_iter=max_iter,
            step_size=step_size,
        )
        rows.append(result)
    return tuple(rows)


def solve_wu_amin_case(
    *,
    pop_lambda: float,
    seed: int = 1,
    max_iter: int = 400,
    step_size: float = 0.08,
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    starts = (
        (0.50, 0.50),
        (0.90, 0.10),
        (0.70, 0.05),
        (0.30, 0.05),
    )
    for start_idx, (initial_pi_a_given_a, initial_pi_a_given_n) in enumerate(starts):
        game = build_game_zero(
            pop_lambda=pop_lambda,
            seed=seed + start_idx,
            initial_pi_a_given_a=initial_pi_a_given_a,
            initial_pi_a_given_n=initial_pi_a_given_n,
        )
        result = game.solve(
            max_iter=max_iter,
            step_size=step_size,
        )
        result = {
            **result,
            "pop_lambda": pop_lambda,
            "start": (initial_pi_a_given_a, initial_pi_a_given_n),
        }
        if best_result is None or result["expected_sender_utility"] < best_result[
            "expected_sender_utility"
        ]:
            best_result = result

    if best_result is None:
        raise RuntimeError("No Wu/Amin optimization run completed.")
    return best_result


def _initial_state_probabilities(
    *,
    pi_a_given_a: float,
    pi_a_given_n: float,
) -> dict[str, dict[frozenset[MetricName], float]]:
    _validate_probability("pi_a_given_a", pi_a_given_a)
    _validate_probability("pi_a_given_n", pi_a_given_n)
    return {
        ACCIDENT_STATE: {
            HAZARD_MASK: pi_a_given_a,
            EMPTY_MASK: 1.0 - pi_a_given_a,
        },
        NO_ACCIDENT_STATE: {
            HAZARD_MASK: pi_a_given_n,
            EMPTY_MASK: 1.0 - pi_a_given_n,
        },
    }


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")


def _build_placeholder_world() -> World:
    arc: Arc = ("O", "D")
    network = InfrastructureGraph(
        V={"O", "D"},
        A={arc},
        I={arc},
        nominal_travel_time={arc: 0.0},
        nominal_discomfort={arc: 0.0},
        nominal_hazards={arc: 0.0},
        nominal_cost={arc: 0.0},
        nominal_policing={"O": 0.0, "D": 0.0},
    )
    individual = Individual(id="wu_amin_receiver", demand=Demand("O", "D"))
    return World(network=network, individuals=frozenset({individual}))


def _single_arc(world: World) -> Arc:
    arcs = tuple(world.A)
    if len(arcs) != 1:
        raise ValueError("The Wu/Amin placeholder world must have exactly one arc.")
    return arcs[0]


if __name__ == "__main__":
    for row in solve_wu_amin_grid():
        print(
            "lambda={pop_lambda:.2f} regime={reference_regime} "
            "pi(a|a)={pi_a_given_a:.4f} ref={reference_pi_a_given_a:.4f} "
            "pi(a|n)={pi_a_given_n:.4f} spillover={expected_spillover:.4f} "
            "ref_spillover={reference_spillover:.4f}".format(**row)
        )
