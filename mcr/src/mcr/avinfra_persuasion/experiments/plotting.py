from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..datastructures import MetricName
from ..bp.senders import Objective
from .helpers import format_mask


def plot_policy_learning(
    metric_x: MetricName | str,
    metric_y: MetricName | str,
    result: Mapping[str, object],
    ax: Axes | None = None,
) -> Axes:
    """
    Plot the learned disclosure-policy trajectory in two selected dimensions.

    The ``result`` argument is expected to be the dictionary returned by
    ``BaseGame.solve()``, with a ``policy_history`` entry containing one
    probability dictionary per iteration.
    """
    metric_x = MetricName.coerce(metric_x)
    metric_y = MetricName.coerce(metric_y)
    if metric_x == metric_y:
        raise ValueError("metric_x and metric_y must be different.")

    points = _policy_points(metric_x, metric_y, result)

    ax = ax or plt.subplots(figsize=(6, 6))[1]
    _draw_policy_trajectory(ax=ax, points=points)

    _format_policy_axes(ax, metric_x, metric_y, title="Policy learning")
    ax.legend(loc="best", frameon=False)
    return ax


def plot_policy_gradient_field(
    metric_x: MetricName | str,
    metric_y: MetricName | str,
    game: object,
    result: Mapping[str, object] | None = None,
    ax: Axes | None = None,
    *,
    grid_size: int = 21,
    finite_diff_epsilon: float = 1e-4,
    boundary_epsilon: float = 1e-3,
    normalize: bool = True,
    show_colorbar: bool = True,
) -> Axes:
    """
    Plot the local policy-gradient field over the Bernoulli policy square.

    The field is evaluated on probability coordinates, but the direction uses
    the same finite-difference logit gradient and sender objective convention
    as the solver. If ``result`` is supplied, its realized policy trajectory is
    overlaid.
    """
    metric_x = MetricName.coerce(metric_x)
    metric_y = MetricName.coerce(metric_y)
    if metric_x == metric_y:
        raise ValueError("metric_x and metric_y must be different.")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if not 0.0 < boundary_epsilon < 0.5:
        raise ValueError("boundary_epsilon must lie in (0, 0.5).")

    metric_order = tuple(getattr(game, "_metric_order", ()))
    if metric_x not in metric_order or metric_y not in metric_order:
        raise ValueError("Both metrics must be present in the game's metric order.")
    if not hasattr(game, "_finite_difference_gradient"):
        raise ValueError("game must provide _finite_difference_gradient().")
    if not hasattr(game, "signaling_scheme"):
        raise ValueError("game must provide signaling_scheme().")

    base_policy = game.signaling_scheme()
    base_probabilities = {
        metric: float(_policy_probability(base_policy, metric))
        for metric in metric_order
    }
    x_index = metric_order.index(metric_x)
    y_index = metric_order.index(metric_y)

    grid = np.linspace(boundary_epsilon, 1.0 - boundary_epsilon, grid_size)
    x_values, y_values = np.meshgrid(grid, grid)
    x_force = np.zeros_like(x_values, dtype=float)
    y_force = np.zeros_like(y_values, dtype=float)

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            probabilities = dict(base_probabilities)
            probabilities[metric_x] = float(x_values[row_idx, col_idx])
            probabilities[metric_y] = float(y_values[row_idx, col_idx])
            logits = _logits_from_probabilities(metric_order, probabilities)
            gradient = game._finite_difference_gradient(
                flat_logits=logits,
                epsilon=finite_diff_epsilon,
            )
            if game.sender.objective == Objective.MINIMIZE:
                gradient = -gradient

            x_probability = probabilities[metric_x]
            y_probability = probabilities[metric_y]
            x_force[row_idx, col_idx] = (
                x_probability * (1.0 - x_probability) * gradient[x_index]
            )
            y_force[row_idx, col_idx] = (
                y_probability * (1.0 - y_probability) * gradient[y_index]
            )

    speed = np.hypot(x_force, y_force)
    arrow_length = 0.85 / max(grid_size - 1, 1)
    if normalize:
        x_plot = np.divide(
            x_force,
            speed,
            out=np.zeros_like(x_force),
            where=speed > 0.0,
        ) * arrow_length
        y_plot = np.divide(
            y_force,
            speed,
            out=np.zeros_like(y_force),
            where=speed > 0.0,
        ) * arrow_length
    else:
        max_speed = float(np.max(speed))
        scale = arrow_length / max_speed if max_speed > 0.0 else 0.0
        x_plot = x_force * scale
        y_plot = y_force * scale

    ax = ax or plt.subplots(figsize=(6.6, 6))[1]
    quiver = ax.quiver(
        x_values,
        y_values,
        x_plot,
        y_plot,
        speed,
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="viridis",
        width=0.0035,
        alpha=0.82,
        zorder=1,
    )
    if show_colorbar:
        colorbar = ax.figure.colorbar(quiver, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Policy-space update norm")

    if result is not None:
        points = _policy_points(metric_x, metric_y, result)
        _draw_policy_trajectory(ax=ax, points=points)
        ax.legend(loc="best", frameon=False)

    _format_policy_axes(ax, metric_x, metric_y, title="Policy gradient field")
    return ax


def plot_state_policy_gradient_field(
    metric_x: MetricName | str,
    metric_y: MetricName | str,
    game: object,
    state_name: str,
    result: Mapping[str, object] | None = None,
    ax: Axes | None = None,
    *,
    grid_size: int = 9,
    finite_diff_epsilon: float = 1e-4,
    boundary_epsilon: float = 1e-3,
    policy_step: float = 0.2,
    normalize: bool = True,
    show_colorbar: bool = True,
) -> Axes:
    """
    Plot a projected policy-gradient field for one state-conditional mask policy.

    State-dependent mask policies are multinomial distributions over masks. For
    the two-metric policies used in ``experiment_2_1``, this projects each
    state distribution to marginal disclosure coordinates:
    ``P(reveal metric_x | state)`` and ``P(reveal metric_y | state)``.
    """
    metric_x = MetricName.coerce(metric_x)
    metric_y = MetricName.coerce(metric_y)
    if metric_x == metric_y:
        raise ValueError("metric_x and metric_y must be different.")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if policy_step <= 0.0:
        raise ValueError("policy_step must be positive.")
    if not 0.0 < boundary_epsilon < 0.5:
        raise ValueError("boundary_epsilon must lie in (0, 0.5).")

    state_order = tuple(getattr(game, "_state_order", ()))
    all_masks = tuple(getattr(game, "_all_masks", ()))
    metric_order = tuple(getattr(game, "_metric_order", ()))
    if not state_order or not all_masks or not metric_order:
        raise ValueError("game must expose state, mask, and metric order metadata.")
    if state_name not in state_order:
        raise ValueError(f"Unknown state {state_name!r}.")
    if frozenset(metric_order) != frozenset({metric_x, metric_y}):
        raise ValueError(
            "Projected state policy gradient fields currently require exactly "
            "the two selected metrics."
        )
    if not hasattr(game, "_finite_difference_gradient"):
        raise ValueError("game must provide _finite_difference_gradient().")
    if not hasattr(game, "signaling_scheme"):
        raise ValueError("game must provide signaling_scheme().")

    state_index = state_order.index(state_name)
    masks_per_state = len(all_masks)
    row_start = state_index * masks_per_state
    row_stop = row_start + masks_per_state
    base_policy = game.signaling_scheme()

    grid = np.linspace(boundary_epsilon, 1.0 - boundary_epsilon, grid_size)
    x_values, y_values = np.meshgrid(grid, grid)
    x_force = np.zeros_like(x_values, dtype=float)
    y_force = np.zeros_like(y_values, dtype=float)

    for row_idx in range(grid_size):
        for col_idx in range(grid_size):
            current_point = np.array(
                [
                    float(x_values[row_idx, col_idx]),
                    float(y_values[row_idx, col_idx]),
                ],
                dtype=float,
            )
            probabilities = _state_policy_with_independent_state_distribution(
                base_policy=base_policy,
                state_name=state_name,
                all_masks=all_masks,
                metric_x=metric_x,
                metric_y=metric_y,
                probability_x=current_point[0],
                probability_y=current_point[1],
            )
            flat_logits = _state_policy_logits(
                state_order=state_order,
                all_masks=all_masks,
                probabilities=probabilities,
            )
            gradient = game._finite_difference_gradient(
                flat_logits=flat_logits,
                epsilon=finite_diff_epsilon,
            )
            if game.sender.objective == Objective.MINIMIZE:
                gradient = -gradient

            next_logits = flat_logits.copy()
            next_logits[row_start:row_stop] += policy_step * gradient[row_start:row_stop]
            next_policy = game.signaling_scheme(logits=next_logits)
            next_point = _state_policy_marginal_point(
                next_policy[state_name],
                metric_x,
                metric_y,
            )
            force = next_point - current_point
            x_force[row_idx, col_idx] = force[0]
            y_force[row_idx, col_idx] = force[1]

    speed = np.hypot(x_force, y_force)
    arrow_length = 0.85 / max(grid_size - 1, 1)
    if normalize:
        x_plot = np.divide(
            x_force,
            speed,
            out=np.zeros_like(x_force),
            where=speed > 0.0,
        ) * arrow_length
        y_plot = np.divide(
            y_force,
            speed,
            out=np.zeros_like(y_force),
            where=speed > 0.0,
        ) * arrow_length
    else:
        max_speed = float(np.max(speed))
        scale = arrow_length / max_speed if max_speed > 0.0 else 0.0
        x_plot = x_force * scale
        y_plot = y_force * scale

    ax = ax or plt.subplots(figsize=(6.6, 6))[1]
    quiver = ax.quiver(
        x_values,
        y_values,
        x_plot,
        y_plot,
        speed,
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="viridis",
        width=0.0035,
        alpha=0.82,
        zorder=1,
    )
    if show_colorbar:
        colorbar = ax.figure.colorbar(quiver, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Projected update norm")

    if result is not None:
        points = _state_policy_points(metric_x, metric_y, state_name, result)
        _draw_policy_trajectory(ax=ax, points=points)
        ax.legend(loc="best", frameon=False)

    _format_policy_axes(
        ax,
        metric_x,
        metric_y,
        title=f"Policy gradient field: {state_name}",
    )
    return ax


def plot_state_policy_gradient_fields(
    metric_x: MetricName | str,
    metric_y: MetricName | str,
    game: object,
    result: Mapping[str, object] | None = None,
    axes: Sequence[Axes] | np.ndarray | None = None,
    *,
    grid_size: int = 9,
    finite_diff_epsilon: float = 1e-4,
    boundary_epsilon: float = 1e-3,
    policy_step: float = 0.2,
    normalize: bool = True,
    show_colorbar: bool = False,
) -> np.ndarray:
    """Plot one projected gradient field and trajectory for each state."""
    state_order = tuple(getattr(game, "_state_order", ()))
    if not state_order:
        raise ValueError("game must expose non-empty _state_order metadata.")

    if axes is None:
        ncols = min(2, len(state_order))
        nrows = int(np.ceil(len(state_order) / ncols))
        _, axes_array = plt.subplots(
            nrows,
            ncols,
            figsize=(6.2 * ncols, 5.7 * nrows),
            squeeze=False,
        )
    else:
        axes_array = np.asarray(axes, dtype=object)
        if axes_array.size < len(state_order):
            raise ValueError("axes must contain at least one axis per state.")

    flat_axes = axes_array.reshape(-1)
    for ax, state_name in zip(flat_axes, state_order):
        plot_state_policy_gradient_field(
            metric_x,
            metric_y,
            game,
            state_name,
            result=result,
            ax=ax,
            grid_size=grid_size,
            finite_diff_epsilon=finite_diff_epsilon,
            boundary_epsilon=boundary_epsilon,
            policy_step=policy_step,
            normalize=normalize,
            show_colorbar=show_colorbar,
        )

    for ax in flat_axes[len(state_order):]:
        ax.set_visible(False)

    return axes_array


def _policy_points(
    metric_x: MetricName,
    metric_y: MetricName,
    result: Mapping[str, object],
) -> np.ndarray:
    policy_history = result.get("policy_history")
    if not isinstance(policy_history, Sequence) or not policy_history:
        raise ValueError(
            "result must contain a non-empty 'policy_history' sequence."
        )

    return np.asarray(
        [
            [
                float(_policy_probability(policy, metric_x)),
                float(_policy_probability(policy, metric_y)),
            ]
            for policy in policy_history
        ],
        dtype=float,
    )


def _draw_policy_trajectory(
    *,
    ax: Axes,
    points: np.ndarray,
) -> None:
    colors = np.linspace(0.2, 0.95, len(points))

    if len(points) > 1:
        deltas = points[1:] - points[:-1]
        ax.quiver(
            points[:-1, 0],
            points[:-1, 1],
            deltas[:, 0],
            deltas[:, 1],
            colors[:-1],
            angles="xy",
            scale_units="xy",
            scale=1,
            cmap="viridis",
            width=0.004,
            alpha=0.85,
            zorder=2,
        )

    ax.plot(
        points[:, 0],
        points[:, 1],
        color="#7f8c8d",
        linewidth=1.1,
        alpha=0.6,
        zorder=1,
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        cmap="viridis",
        s=28,
        edgecolors="#202020",
        linewidths=0.5,
        zorder=3,
    )
    ax.scatter(
        [points[0, 0]],
        [points[0, 1]],
        s=70,
        color="#f39c12",
        edgecolors="#202020",
        linewidths=0.7,
        zorder=4,
        label="start",
    )
    ax.scatter(
        [points[-1, 0]],
        [points[-1, 1]],
        s=80,
        color="#c0392b",
        edgecolors="#202020",
        linewidths=0.8,
        zorder=5,
        label="final",
    )

    for idx, (x_value, y_value) in enumerate(points):
        if idx in {0, len(points) - 1}:
            ax.annotate(
                str(idx),
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )


def _format_policy_axes(
    ax: Axes,
    metric_x: MetricName,
    metric_y: MetricName,
    *,
    title: str,
) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"P(reveal {metric_x.value})")
    ax.set_ylabel(f"P(reveal {metric_y.value})")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.35)


def _logits_from_probabilities(
    metric_order: Sequence[MetricName],
    probabilities: Mapping[MetricName, float],
) -> np.ndarray:
    probability_values = np.asarray(
        [probabilities[metric] for metric in metric_order],
        dtype=float,
    )
    clipped = np.clip(probability_values, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def _policy_probability(
    policy: object,
    metric: MetricName,
) -> float:
    if not isinstance(policy, Mapping):
        raise ValueError("Each policy_history entry must be a mapping.")

    if metric in policy:
        return float(policy[metric])

    metric_name = metric.value
    if metric_name in policy:
        return float(policy[metric_name])

    raise ValueError(f"Policy history does not contain {metric.value!r}.")


def _state_policy_points(
    metric_x: MetricName,
    metric_y: MetricName,
    state_name: str,
    result: Mapping[str, object],
) -> np.ndarray:
    policy_history = result.get("policy_history")
    if not isinstance(policy_history, Sequence) or not policy_history:
        raise ValueError(
            "result must contain a non-empty 'policy_history' sequence."
        )

    points = []
    for policy in policy_history:
        if not isinstance(policy, Mapping) or state_name not in policy:
            raise ValueError(
                "Each state policy history entry must contain the requested state."
            )
        points.append(
            _state_policy_marginal_point(policy[state_name], metric_x, metric_y)
        )
    return np.asarray(points, dtype=float)


def _state_policy_marginal_point(
    distribution: object,
    metric_x: MetricName,
    metric_y: MetricName,
) -> np.ndarray:
    if not isinstance(distribution, Mapping):
        raise ValueError("State distribution must be a mapping.")
    return np.asarray(
        [
            _state_policy_marginal_probability(distribution, metric_x),
            _state_policy_marginal_probability(distribution, metric_y),
        ],
        dtype=float,
    )


def _state_policy_marginal_probability(
    distribution: Mapping[object, object],
    metric: MetricName,
) -> float:
    probability = 0.0
    for mask, mask_probability in distribution.items():
        if metric in frozenset(MetricName.coerce(item) for item in mask):
            probability += float(mask_probability)
    return probability


def _state_policy_with_independent_state_distribution(
    *,
    base_policy: Mapping[str, Mapping[frozenset[MetricName], float]],
    state_name: str,
    all_masks: Sequence[frozenset[MetricName]],
    metric_x: MetricName,
    metric_y: MetricName,
    probability_x: float,
    probability_y: float,
) -> dict[str, dict[frozenset[MetricName], float]]:
    probabilities = {
        state: dict(distribution)
        for state, distribution in base_policy.items()
    }
    probabilities[state_name] = {
        mask: _independent_mask_probability(
            mask=mask,
            metric_x=metric_x,
            metric_y=metric_y,
            probability_x=probability_x,
            probability_y=probability_y,
        )
        for mask in all_masks
    }
    return probabilities


def _independent_mask_probability(
    *,
    mask: frozenset[MetricName],
    metric_x: MetricName,
    metric_y: MetricName,
    probability_x: float,
    probability_y: float,
) -> float:
    probability = 1.0
    for metric, reveal_probability in (
        (metric_x, probability_x),
        (metric_y, probability_y),
    ):
        probability *= (
            reveal_probability
            if metric in mask
            else 1.0 - reveal_probability
        )
    return probability


def _state_policy_logits(
    *,
    state_order: Sequence[str],
    all_masks: Sequence[frozenset[MetricName]],
    probabilities: Mapping[str, Mapping[frozenset[MetricName], float]],
) -> np.ndarray:
    probability_values = np.asarray(
        [
            [probabilities[state_name][mask] for mask in all_masks]
            for state_name in state_order
        ],
        dtype=float,
    )
    clipped = np.clip(probability_values, 1e-9, 1.0 - 1e-9)
    return np.log(clipped).reshape(-1)


def plot_state_mask_policy(
    result: Mapping[str, object],
    ax: Axes | None = None,
) -> Axes:
    """
    Plot a compact heatmap of the learned state-conditional mask distribution.

    The ``result`` argument is expected to be the dictionary returned by
    ``BaseGame.solve()``, with a ``final_probabilities`` entry mapping each
    state name to a mask-probability table.
    """
    final_probabilities = result.get("final_probabilities")
    if not isinstance(final_probabilities, Mapping) or not final_probabilities:
        raise ValueError(
            "result must contain a non-empty 'final_probabilities' mapping."
        )

    state_names = tuple(sorted(str(state_name) for state_name in final_probabilities))
    first_distribution = final_probabilities[state_names[0]]
    if not isinstance(first_distribution, Mapping) or not first_distribution:
        raise ValueError(
            "Each state distribution in 'final_probabilities' must be a "
            "non-empty mapping."
        )

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
    values = np.asarray(
        [
            [
                float(
                    _state_mask_probability(
                        final_probabilities[state_name],
                        mask,
                    )
                )
                for mask in masks
            ]
            for state_name in state_names
        ],
        dtype=float,
    )

    width = max(6.0, 1.3 * len(masks))
    height = max(3.0, 0.8 * len(state_names) + 1.5)
    ax = ax or plt.subplots(figsize=(width, height))[1]
    image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(masks)))
    ax.set_xticklabels([format_mask(mask) for mask in masks], rotation=25, ha="right")
    ax.set_yticks(range(len(state_names)))
    ax.set_yticklabels(state_names)
    ax.set_xlabel("Mask")
    ax.set_ylabel("State")
    ax.set_title("State-dependent mask policy")

    for row_idx, state_name in enumerate(state_names):
        for col_idx, mask in enumerate(masks):
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
    colorbar.set_label("P(mask | state)")
    return ax


def _state_mask_probability(
    distribution: object,
    mask: frozenset[MetricName],
) -> float:
    if not isinstance(distribution, Mapping):
        raise ValueError("Each state distribution must be a mapping.")
    if mask in distribution:
        return float(distribution[mask])
    raise ValueError(f"State distribution does not contain mask {format_mask(mask)!r}.")
