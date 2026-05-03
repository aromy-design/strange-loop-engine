# Experiments

This directory contains the scientific experiments for the Metabolic Loop System project.

## How to run

All experiments are deterministic given a seed and runnable from the project root:

```bash
# E1 — Lesion study (~17 min for 8 seeds × 5000 steps × 10 conditions)
python -m experiments.lesion_study --seeds 8 --steps 5000

# Quick smoke (~6 min, 3 seeds × 2000 steps)
python -m experiments.lesion_study --quick

# Analyze E1 results
python -m experiments.analyze_lesion
python -m experiments.plot_lesion

# E2 — Sleep effect
python -m experiments.sleep_study --seeds 15 --steps 8000

# E3 — Sparsity sweep
python -m experiments.sparsity_sweep --seeds 8 --steps 4000

# E5 — Trajectory (long-run with time series)
python -m experiments.trajectory_study --seeds 5 --steps 15000

# Generate architecture diagram
python -m experiments.generate_diagram
```

## Directory layout

```
experiments/
├── runner.py              — headless agent runner
├── lesion_study.py        — E1 orchestrator
├── sleep_study.py         — E2 orchestrator
├── sparsity_sweep.py      — E3 orchestrator
├── trajectory_study.py    — E5 orchestrator (time series logging)
├── analyze_lesion.py      — t-tests + Cohen's d for E1
├── plot_lesion.py         — bar charts for E1
├── generate_diagram.py    — architecture PNG
├── configs/               — YAML configs (future)
├── results/
│   ├── lesion/
│   │   ├── summary.csv
│   │   ├── stats_report.txt
│   │   └── stats_table.csv
│   ├── sleep/
│   ├── sparsity/
│   └── trajectory/
└── plots/
    ├── architecture.png
    ├── lesion_eats.png
    ├── lesion_danger.png
    ├── lesion_mirror.png
    └── ...
```

## Reproducibility

- All RNG seeded explicitly (NumPy + Python random)
- World seed = agent seed by default (can override)
- Output CSVs include all hyperparameters
- Total CPU budget per E1 full run: ~10 min (no GPU needed)

## Statistical conventions

- Welch's t-test (unequal variances) for between-condition comparisons
- Cohen's d effect size
- Bonferroni correction when comparing many lesion conditions to FULL
- Significance levels: `*` p<0.05, `**` p<0.01, `***` p<0.001
- Sample size: pilot 3-8 seeds, full study 30 seeds

## Conditions available for ablation (lesion flags)

Pass any subset to `MetabolicAgent(lesion=[...])`:

- `spatial_map` — disable TD value map
- `behavioral_modes` — force EXPLORE, no mode switching
- `sleep` — no sleep cycle, no replay
- `insect_brain` — zero out insect brain motor contribution
- `adjacent_food` — no reflex eating when food adjacent
- `mushroom` — silence mushroom body Kenyon cells
- `central_complex` — silence CX
- `lateral_horn` — silence LH
- `vision` — silence visual lobe
- `antennal` — silence olfactory lobe
- `associative` — disable signal-outcome conditioning
- `pheromones` — disable trail field

## Hardware

- Tested on consumer Windows 11, Python 3.14, NumPy 1.x, scipy 1.x
- ~5-7 ms per simulation step with full agent (5350 neurons total)
- No GPU required for current scale
