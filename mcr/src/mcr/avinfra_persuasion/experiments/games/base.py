from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
from typing import Any, ClassVar

import numpy as np
from tqdm import tqdm

from ...bp.game import ConvergenceGame
from ...bp.receivers import Receiver
from ...bp.senders import Objective, Sender
from ...bp.signals import (
    MaskSignalPolicy,
    Signal,
    StateDependentMaskSignalPolicy,
    TypedStateDependentMaskSignalPolicy,
)
from ...datastructures import FinitePrior, MetricName, Prior, Scenario, World
from ..helpers import sorted_metrics


@dataclass
class BaseFiniteMaskGame(ConvergenceGame):
    sender: Sender
    receivers: list[Receiver]
    world: World
    public_prior: Prior
    seed: int

    _metric_order: tuple[MetricName, ...] = field(init=False, repr=False)
    _all_masks: tuple[frozenset[MetricName], ...] = field(init=False, repr=False)
    _logits: np.ndarray = field(init=False, repr=False)

    _validation_name: ClassVar[str | None] = None
    _min_receivers: ClassVar[int] = 1
    _exact_receivers: ClassVar[int | None] = None
    _receiver_count_error: ClassVar[str | None] = None
    _finite_prior_error: ClassVar[str | None] = None
    _signal_policy_error: ClassVar[str | None] = None
    _required_signal_policy_type: ClassVar[type[Any] | tuple[type[Any], ...] | None] = (
        None
    )

    def __post_init__(self) -> None:
        self._validate_receiver_count()
        self._validate_finite_prior()
        self._validate_signal_policy()
        self._validate_shared_model()
        self._initialize_mask_support()

    @property
    def finite_prior(self) -> FinitePrior:
        if not isinstance(self.public_prior, FinitePrior):
            raise NotImplementedError(self._finite_prior_message())
        return self.public_prior

    def _game_name(self) -> str:
        return self._validation_name or type(self).__name__

    def _finite_prior_message(self) -> str:
        if self._finite_prior_error is not None:
            return self._finite_prior_error
        return f"{self._game_name()} currently requires a FinitePrior."

    def _validate_receiver_count(self) -> None:
        if (
            self._exact_receivers is not None
            and len(self.receivers) != self._exact_receivers
        ):
            if self._receiver_count_error is not None:
                raise ValueError(self._receiver_count_error)
            raise ValueError(
                f"{self._game_name()} requires exactly "
                f"{self._exact_receivers} receiver(s)."
            )
        if self._exact_receivers is None and len(self.receivers) < self._min_receivers:
            if self._receiver_count_error is not None:
                raise ValueError(self._receiver_count_error)
            raise ValueError(
                f"{self._game_name()} requires at least "
                f"{self._min_receivers} receiver(s)."
            )

    def _validate_finite_prior(self) -> None:
        if not isinstance(self.public_prior, FinitePrior):
            raise NotImplementedError(self._finite_prior_message())

    def _validate_signal_policy(self) -> None:
        if self._required_signal_policy_type is None:
            return
        if not isinstance(self.sender.signal_policy, self._required_signal_policy_type):
            if self._signal_policy_error is not None:
                raise NotImplementedError(self._signal_policy_error)
            raise NotImplementedError(
                f"{self._game_name()} requires a supported mask signal policy."
            )

    def _validate_shared_model(self) -> None:
        if self.sender.prior != self.public_prior:
            raise ValueError("Sender prior must match the public prior.")
        if any(receiver.prior != self.public_prior for receiver in self.receivers):
            raise ValueError("Receiver priors must match the public prior.")
        if self.sender.world != self.world:
            raise ValueError("Sender world must match the game world.")
        if any(receiver.world != self.world for receiver in self.receivers):
            raise ValueError("Receiver worlds must match the game world.")

    def _initialize_mask_support(self) -> None:
        self._metric_order = sorted_metrics(
            self.sender.signal_policy.considered_metrics
        )
        self._all_masks = self._build_all_masks(self._metric_order)

    @staticmethod
    def _build_all_masks(
        metric_order: tuple[MetricName, ...],
    ) -> tuple[frozenset[MetricName], ...]:
        return tuple(
            frozenset(
                metric
                for idx, metric in enumerate(metric_order)
                if bitmask & (1 << idx)
            )
            for bitmask in range(1 << len(metric_order))
        )

    def _sender_metric(self) -> MetricName:
        if len(self.sender.preference.elements) != 1:
            raise ValueError("Sender preference must be degenerate.")
        return next(iter(self.sender.preference.elements))

    def _receiver_after_signal(
        self,
        receiver: Receiver,
        signal: Signal,
    ) -> Receiver:
        receiver.reset_for_evaluation()
        receiver.update_internal_belief(signal)
        return receiver

    def _path_choices_after_signal(
        self,
        signal: Signal,
    ) -> tuple[list[Receiver], dict[Any, Any]]:
        updated_receivers: list[Receiver] = []
        path_choices = {}

        for receiver in self.receivers:
            updated_receiver = self._receiver_after_signal(receiver, signal)
            updated_receivers.append(updated_receiver)
            path_choices[updated_receiver.individual] = (
                updated_receiver.get_path_choice()
            )

        return updated_receivers, path_choices

    def _receiver_metrics_after_realization(
        self,
        updated_receivers: list[Receiver],
        realized_scenario: Scenario,
    ) -> dict[str, Mapping[MetricName, float]]:
        return {
            receiver.id: receiver.compute_realized_metrics(realized_scenario)
            for receiver in updated_receivers
        }

    def _evaluate_multi_receiver_signal(
        self,
        signal: Signal,
        believed_scenario: Scenario,
    ) -> dict[str, Any]:
        updated_receivers, path_choices = self._path_choices_after_signal(signal)
        realized_scenario = self.world.get_realized_metrics(
            path_choices=path_choices,
            name=f"realized_{believed_scenario.name}",
            base_scenario=believed_scenario,
        )
        receiver_metrics = self._receiver_metrics_after_realization(
            updated_receivers=updated_receivers,
            realized_scenario=realized_scenario,
        )
        sender_metric = self._sender_metric()
        sender_metric_value = sum(
            metrics[sender_metric] for metrics in receiver_metrics.values()
        )

        return {
            "realized_scenario": realized_scenario,
            "path_choices": path_choices,
            "receiver_metrics": receiver_metrics,
            "sender_metric_value": sender_metric_value,
            "path_counts": Counter(choice.path for choice in path_choices.values()),
        }

    def _evaluate_signal(
        self,
        signal: Signal,
        believed_scenario: Scenario,
    ) -> dict[str, Any]:
        return self._evaluate_multi_receiver_signal(
            signal=signal,
            believed_scenario=believed_scenario,
        )


@dataclass
class BernoulliMaskGameBase(BaseFiniteMaskGame):
    _required_signal_policy_type: ClassVar[type[Any]] = MaskSignalPolicy

    def __post_init__(self) -> None:
        super().__post_init__()
        policy = self.sender.signal_policy
        if not isinstance(policy, MaskSignalPolicy):
            raise NotImplementedError(self._signal_policy_error)

        probabilities = np.array(
            [policy.probability(metric) for metric in self._metric_order],
            dtype=float,
        )
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        self._logits = np.log(clipped / (1.0 - clipped))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def signaling_scheme(
        self,
        logits: np.ndarray | None = None,
    ) -> dict[MetricName, float]:
        logits = self._logits if logits is None else np.asarray(logits, dtype=float)
        probabilities = self._sigmoid(logits)
        return {
            metric: float(probability)
            for metric, probability in zip(self._metric_order, probabilities)
        }

    def _mask_probability(
        self,
        mask: frozenset[MetricName],
        probabilities: Mapping[MetricName, float],
    ) -> float:
        mass = 1.0
        for metric in self._metric_order:
            probability = probabilities[metric]
            mass *= probability if metric in mask else (1.0 - probability)
        return mass

    def _update_sender_policy(
        self,
        probabilities: Mapping[MetricName, float],
    ) -> None:
        policy = self.sender.signal_policy
        if not isinstance(policy, MaskSignalPolicy):
            raise NotImplementedError(self._signal_policy_error)
        policy.update_probabilities(probabilities)


@dataclass
class StateDependentMaskGameBase(BaseFiniteMaskGame):
    _state_order: tuple[str, ...] = field(init=False, repr=False)

    _required_signal_policy_type: ClassVar[type[Any]] = StateDependentMaskSignalPolicy

    def __post_init__(self) -> None:
        super().__post_init__()
        policy = self.sender.signal_policy

        self._state_order = tuple(sorted(self.finite_prior.support))
        if policy.state_names != frozenset(self._state_order):
            raise ValueError(
                "State-dependent signal policy states must match the finite prior support."
            )

        probabilities = np.array(
            [
                [policy.mask_probability(state_name, mask) for mask in self._all_masks]
                for state_name in self._state_order
            ],
            dtype=float,
        )
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        self._logits = np.log(clipped)

    @staticmethod
    def _softmax_rows(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def signaling_scheme(
        self,
        logits: np.ndarray | None = None,
    ) -> dict[str, dict[frozenset[MetricName], float]]:
        if logits is None:
            logits_array = self._logits
        else:
            logits_array = np.asarray(logits, dtype=float)
            if logits_array.ndim == 1:
                logits_array = logits_array.reshape(
                    len(self._state_order),
                    len(self._all_masks),
                )
        probabilities = self._softmax_rows(logits_array)
        return {
            state_name: {
                mask: float(probability)
                for mask, probability in zip(self._all_masks, state_probabilities)
            }
            for state_name, state_probabilities in zip(self._state_order, probabilities)
        }

    def _mask_probability(
        self,
        state_name: str,
        mask: frozenset[MetricName],
        probabilities: Mapping[str, Mapping[frozenset[MetricName], float]],
    ) -> float:
        return probabilities[state_name][mask]

    def _state_probabilities_snapshot(
        self,
    ) -> dict[str, dict[frozenset[MetricName], float]]:
        policy = self.sender.signal_policy
        if not isinstance(policy, StateDependentMaskSignalPolicy):
            raise NotImplementedError(self._signal_policy_error)
        return {
            state_name: dict(distribution)
            for state_name, distribution in policy.state_probabilities.items()
        }

    @contextmanager
    def _temporary_state_distributions(
        self,
        probabilities: Mapping[str, Mapping[frozenset[MetricName], float]],
    ) -> Iterator[None]:
        policy = self.sender.signal_policy
        if not isinstance(policy, StateDependentMaskSignalPolicy):
            raise NotImplementedError(self._signal_policy_error)
        previous_probabilities = self._state_probabilities_snapshot()
        policy.update_state_distributions(probabilities)
        try:
            yield
        finally:
            policy.update_state_distributions(previous_probabilities)

    def _deterministic_scheme(
        self,
        mask_indices: tuple[int, ...],
    ) -> dict[str, dict[frozenset[MetricName], float]]:
        return {
            state_name: {
                mask: 1.0 if mask_idx == chosen_mask_idx else 0.0
                for mask_idx, mask in enumerate(self._all_masks)
            }
            for state_name, chosen_mask_idx in zip(self._state_order, mask_indices)
        }

    def _deterministic_scheme_repetitions(self) -> int:
        return len(self._state_order)

    def _update_sender_policy(
        self,
        probabilities: Mapping[str, Mapping[frozenset[MetricName], float]],
    ) -> None:
        policy = self.sender.signal_policy
        if not isinstance(policy, StateDependentMaskSignalPolicy):
            raise NotImplementedError(self._signal_policy_error)
        policy.update_state_distributions(probabilities)


@dataclass
class TypedStateDependentMaskGameBase(BaseFiniteMaskGame):
    _state_order: tuple[str, ...] = field(init=False, repr=False)
    _type_order: tuple[str, ...] = field(init=False, repr=False)

    _required_signal_policy_type: ClassVar[type[Any]] = (
        TypedStateDependentMaskSignalPolicy
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        policy = self.sender.signal_policy
        if not isinstance(policy, TypedStateDependentMaskSignalPolicy):
            raise NotImplementedError(self._signal_policy_error)

        self._state_order = tuple(sorted(self.finite_prior.support))
        if policy.state_names != frozenset(self._state_order):
            raise ValueError(
                "State-dependent signal policy states must match the finite prior support."
            )

        self._type_order = tuple(
            sorted({receiver.rtype for receiver in self.receivers})
        )
        if policy.type_names != frozenset(self._type_order):
            raise ValueError(
                "Typed state-dependent signal policy receiver types must match "
                "the game receiver types."
            )

        probabilities = np.array(
            [
                [
                    [
                        policy.mask_probability(state_name, type_name, mask)
                        for mask in self._all_masks
                    ]
                    for type_name in self._type_order
                ]
                for state_name in self._state_order
            ],
            dtype=float,
        )
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        self._logits = np.log(clipped)

    @staticmethod
    def _softmax_rows(x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def signaling_scheme(
        self,
        logits: np.ndarray | None = None,
    ) -> dict[str, dict[str, dict[frozenset[MetricName], float]]]:
        if logits is None:
            logits_array = self._logits
        else:
            logits_array = np.asarray(logits, dtype=float)
            if logits_array.ndim == 1:
                logits_array = logits_array.reshape(
                    len(self._state_order),
                    len(self._type_order),
                    len(self._all_masks),
                )
        probabilities = self._softmax_rows(
            logits_array.reshape(-1, len(self._all_masks))
        ).reshape(
            len(self._state_order),
            len(self._type_order),
            len(self._all_masks),
        )
        return {
            state_name: {
                type_name: {
                    mask: float(probability)
                    for mask, probability in zip(self._all_masks, type_probabilities)
                }
                for type_name, type_probabilities in zip(
                    self._type_order,
                    state_probabilities,
                )
            }
            for state_name, state_probabilities in zip(self._state_order, probabilities)
        }

    def _mask_probability(
        self,
        state_name: str,
        receiver_type: str,
        mask: frozenset[MetricName],
        probabilities: Mapping[
            str, Mapping[str, Mapping[frozenset[MetricName], float]]
        ],
    ) -> float:
        return probabilities[state_name][receiver_type][mask]

    def _state_type_probabilities_snapshot(
        self,
    ) -> dict[str, dict[str, dict[frozenset[MetricName], float]]]:
        policy = self.sender.signal_policy
        if not isinstance(policy, TypedStateDependentMaskSignalPolicy):
            raise NotImplementedError(self._signal_policy_error)
        return {
            state_name: {
                type_name: dict(distribution)
                for type_name, distribution in type_distributions.items()
            }
            for state_name, type_distributions in policy.state_type_probabilities.items()
        }

    @contextmanager
    def _temporary_state_distributions(
        self,
        probabilities: Mapping[
            str, Mapping[str, Mapping[frozenset[MetricName], float]]
        ],
    ) -> Iterator[None]:
        policy = self.sender.signal_policy
        if not isinstance(policy, TypedStateDependentMaskSignalPolicy):
            raise NotImplementedError(self._signal_policy_error)
        previous_probabilities = self._state_type_probabilities_snapshot()
        policy.update_state_type_distributions(probabilities)
        try:
            yield
        finally:
            policy.update_state_type_distributions(previous_probabilities)

    def _deterministic_scheme(
        self,
        mask_indices: tuple[int, ...],
    ) -> dict[str, dict[str, dict[frozenset[MetricName], float]]]:
        state_type_pairs = tuple(
            (state_name, type_name)
            for state_name in self._state_order
            for type_name in self._type_order
        )
        probabilities: dict[str, dict[str, dict[frozenset[MetricName], float]]] = {
            state_name: {} for state_name in self._state_order
        }
        for (state_name, type_name), chosen_mask_idx in zip(
            state_type_pairs,
            mask_indices,
        ):
            probabilities[state_name][type_name] = {
                mask: 1.0 if mask_idx == chosen_mask_idx else 0.0
                for mask_idx, mask in enumerate(self._all_masks)
            }
        return probabilities

    def _deterministic_scheme_repetitions(self) -> int:
        return len(self._state_order) * len(self._type_order)

    def _path_choices_after_type_signals(
        self,
        signals_by_type: Mapping[str, Signal],
    ) -> tuple[list[Receiver], dict[Any, Any]]:
        updated_receivers: list[Receiver] = []
        path_choices = {}

        for receiver in self.receivers:
            updated_receiver = self._receiver_after_signal(
                receiver,
                signals_by_type[receiver.rtype],
            )
            updated_receivers.append(updated_receiver)
            path_choices[updated_receiver.individual] = (
                updated_receiver.get_path_choice()
            )

        return updated_receivers, path_choices

    def _evaluate_typed_signals(
        self,
        signals_by_type: Mapping[str, Signal],
        believed_scenario: Scenario,
    ) -> dict[str, Any]:
        updated_receivers, path_choices = self._path_choices_after_type_signals(
            signals_by_type
        )
        realized_scenario = self.world.get_realized_metrics(
            path_choices=path_choices,
            name=f"realized_{believed_scenario.name}",
            base_scenario=believed_scenario,
        )
        receiver_metrics = self._receiver_metrics_after_realization(
            updated_receivers=updated_receivers,
            realized_scenario=realized_scenario,
        )
        sender_metric = self._sender_metric()
        sender_metric_value = sum(
            metrics[sender_metric] for metrics in receiver_metrics.values()
        )

        return {
            "realized_scenario": realized_scenario,
            "path_choices": path_choices,
            "receiver_metrics": receiver_metrics,
            "sender_metric_value": sender_metric_value,
            "path_counts": Counter(choice.path for choice in path_choices.values()),
        }

    def _update_sender_policy(
        self,
        probabilities: Mapping[
            str, Mapping[str, Mapping[frozenset[MetricName], float]]
        ],
    ) -> None:
        policy = self.sender.signal_policy
        if not isinstance(policy, TypedStateDependentMaskSignalPolicy):
            raise NotImplementedError(self._signal_policy_error)
        policy.update_state_type_distributions(probabilities)


class FiniteDifferenceAdamMixin:
    def _objective_from_flat_logits(self, flat_logits: np.ndarray) -> float:
        probabilities = self.signaling_scheme(logits=flat_logits)
        return float(self.evaluate_policy(probabilities)["expected_sender_utility"])

    def _finite_difference_gradient(
        self,
        flat_logits: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        gradient = np.zeros_like(flat_logits)
        for idx in range(flat_logits.size):
            direction = np.zeros_like(flat_logits)
            direction[idx] = epsilon
            plus = self._objective_from_flat_logits(flat_logits + direction)
            minus = self._objective_from_flat_logits(flat_logits - direction)
            gradient[idx] = (plus - minus) / (2.0 * epsilon)
        return gradient

    def _flat_logits_for_solver(self) -> np.ndarray:
        return self._logits.reshape(-1).copy()

    def _store_flat_logits(self, flat_logits: np.ndarray) -> None:
        self._logits = flat_logits.reshape(self._logits.shape)

    def _solve_with_finite_difference_adam(
        self,
        *,
        max_iter: int,
        step_size: float,
        finite_diff_epsilon: float,
        convergence_tol: float,
        convergence_patience: int,
        progress: bool = False,
    ) -> dict[str, Any]:
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if finite_diff_epsilon <= 0:
            raise ValueError("finite_diff_epsilon must be positive.")
        if convergence_patience <= 0:
            raise ValueError("convergence_patience must be positive.")

        flat_logits = self._flat_logits_for_solver()
        m = np.zeros_like(flat_logits)
        v = np.zeros_like(flat_logits)
        beta1 = 0.9
        beta2 = 0.999
        adam_eps = 1e-4

        utility_history: list[float] = []
        grad_norm_history: list[float] = []
        policy_history: list[Any] = []
        stagnant_steps = 0
        converged = False

        steps = range(1, max_iter + 1)
        if progress:
            steps = tqdm(steps)

        for step in steps:
            policy_history.append(self.signaling_scheme(logits=flat_logits))
            utility = self._objective_from_flat_logits(flat_logits)
            gradient = self._finite_difference_gradient(
                flat_logits=flat_logits,
                epsilon=finite_diff_epsilon,
            )
            if self.sender.objective == Objective.MINIMIZE:
                gradient = -gradient
            grad_norm = float(np.linalg.norm(gradient))

            m = beta1 * m + (1.0 - beta1) * gradient
            v = beta2 * v + (1.0 - beta2) * (gradient * gradient)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            flat_logits = flat_logits + step_size * m_hat / (np.sqrt(v_hat) + adam_eps)

            utility_history.append(float(utility))
            grad_norm_history.append(grad_norm)

            if len(utility_history) >= 2:
                utility_delta = abs(utility_history[-1] - utility_history[-2])
                if utility_delta < convergence_tol:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0

            if stagnant_steps >= convergence_patience:
                converged = True
                break

        self._store_flat_logits(flat_logits)
        final_probabilities = self.signaling_scheme()
        self._update_sender_policy(final_probabilities)
        evaluation = self.evaluate_policy(final_probabilities)
        return {
            "iterations": len(utility_history),
            "converged": converged,
            "utility_history": utility_history,
            "grad_norm_history": grad_norm_history,
            "policy_history": policy_history + [final_probabilities],
            "final_probabilities": final_probabilities,
            **evaluation,
        }


class EnumerationMixin:
    def _solve_by_enumeration(self) -> dict[str, Any]:
        utility_history: list[float] = []
        policy_history: list[dict[str, dict[frozenset[MetricName], float]]] = []
        best_policy: dict[str, dict[frozenset[MetricName], float]] | None = None
        best_evaluation: dict[str, Any] | None = None
        best_utility = (
            float("-inf")
            if self.sender.objective == Objective.MAXIMIZE
            else float("inf")
        )

        for mask_indices in product(
            range(len(self._all_masks)),
            repeat=self._deterministic_scheme_repetitions(),
        ):
            policy = self._deterministic_scheme(mask_indices)
            evaluation = self.evaluate_policy(policy)
            utility = float(evaluation["expected_sender_utility"])
            utility_history.append(utility)
            policy_history.append(policy)
            better_utility = (
                utility > best_utility
                if self.sender.objective == Objective.MAXIMIZE
                else utility < best_utility
            )
            if better_utility:
                best_utility = utility
                best_policy = policy
                best_evaluation = evaluation

        if best_policy is None or best_evaluation is None:
            raise RuntimeError(
                "Deterministic policy enumeration produced no candidate."
            )

        final_probabilities = best_policy
        self._update_sender_policy(final_probabilities)
        return {
            "iterations": len(utility_history),
            "converged": True,
            "search_mode": "deterministic_enumeration",
            "utility_history": utility_history,
            "grad_norm_history": [],
            "policy_history": policy_history + [final_probabilities],
            "final_probabilities": final_probabilities,
            **best_evaluation,
        }
