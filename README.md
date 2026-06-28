# Bayesian Persuasion for Autonomous Vehicle Infrastructure

With this repo, we are trying to answer the question:

```quote
When an independent 
```

## Installation

### Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- A C++20-capable compiler (Xcode Command Line Tools on macOS)
- GLPK (required by the BenPy solver — see below)

```bash
brew install glpk
```

### Install

```bash
git clone <repo-url> ./bp4avi
cd mixed-customer-response

# Set GLPK paths so BenPy builds correctly (macOS)
GLPK_PREFIX="$(brew --prefix glpk)"
export PATH="$GLPK_PREFIX/bin:$PATH"
export CFLAGS="-I$GLPK_PREFIX/include"
export CPPFLAGS="-I$GLPK_PREFIX/include"
export LDFLAGS="-L$GLPK_PREFIX/lib -Wl,-rpath,$GLPK_PREFIX/lib"

uv sync
```

This will:
1. Create a virtual environment under `.venv/`
2. Install all Python dependencies (including BenPy from its git source)
3. Build the `mcr` package, which compiles the Cython/C++ MDA extension (`_mosp_ext`)

### Running tests

```bash
uv run pytest
```

---

### Troubleshooting BenPy on macOS

If `uv sync` fails while building BenPy, make sure the GLPK environment variables above are set and that `brew install glpk` was run first.
