# Bayesian Persuasion

Core library for **Bayesian Persuasion for real-world infrastructure information design**.

This is a clean rewrite of the prior `mcr.avinfra_persuasion` code (kept for
reference, untracked, under `../archive/`). The benpy-based optimization layer
is intentionally left behind; the new stack targets `gurobipy` / `taichi`.

## Layout

- `bp.world` — solver-agnostic world model: `InfrastructureGraph`, `World`,
  `Individual` / `Demand`, `Scenario`, and the `Belief` / `Prior` hierarchy.

## Usage

From the repo root:

```bash
uv sync
```

Then, from any notebook or script using the workspace `.venv`:

```python
from bp.world import World, InfrastructureGraph, Scenario
```

`bp` is installed as an editable workspace member, so edits under `src/bp/`
are picked up without reinstalling.
