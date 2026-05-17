# strange-loop-engine

Code, data, and pre-registered analysis for the paper:

> **The trained mushroom body is dispensable for aversive transfer in an embodied insect-brain agent with the real Drosophila connectome**
> Emanuele Cerni, 2026.
> Pre-registration: OSF, DOI [10.17605/OSF.IO/397GJ](https://doi.org/10.17605/OSF.IO/397GJ) (CC0 1.0).

## What this is

A sparse-spiking insect-brain agent (`InsectBrainV2`, 81,000 LIF neurons across
seven named circuits) embodied in a small foraging environment, trained under
cross-projection STDP plasticity. The brain runs in real-time on a CPU
(~5.9 ms per simulated tick) and can optionally be instantiated with the
actual Janelia FlyEM hemibrain v1.2.1 connectome substituted into its
inter-circuit projections. Three lesion paradigms (E7v2 pre-registered,
E10/E12 connectome replication, E13 selective pathway lesions) probe whether
the trained mushroom body is the locus of the avoidance memory the agent
learns. Short answer: it isn't.

## Quick start

```
pip install -r requirements.txt
python -m experiments.e7v2_full_mb_silencing --seeds 30 --workers 2
python -m experiments.analyze_e7v2
python -m experiments.generate_paper_figures
```

## Repository layout

```
backend/                       # spiking brain + world + agent + plasticity
  insect_brain_v2.py           # 7-circuit InsectBrainV2 architecture
  agent.py                     # MetabolicAgent (brain + shortcuts + homeostat)
  world.py                     # 16x16 GridWorld (food, danger, shelter)
  spiking_field.py             # SparseSpikingCircuit (LIF, scipy.sparse)
  flyem_connectome.py          # FlyEM hemibrain loader and ROI->circuit mapping
  spatial_map.py               # TD(λ) spatial value map (shortcut layer)
  behavioral_modes.py          # FORAGE/FLEE/REST/EXPLORE/SLEEP controller

experiments/                   # all simulation runs and analysis scripts
  e7v2_full_mb_silencing.py    # pre-registered shortcuts-active null (n=30)
  e10_scaffolded.py            # shortcuts-removed dissociation (n=8)
  e12_flyem_danger.py          # FlyEM-substituted dissociation (n=8)
  e13_selective_lesion.py      # selective MB pathway lesions (n=8)
  analyze_e7v2.py              # pre-registered analysis (Welch + bootstrap d + JZS BF_01)
  analyze_e13.py               # E13 analysis with same triple-criterion
  generate_paper_figures.py    # all figures from CSVs
  results/                     # CSV outputs of each experiment
    e7v2/results.csv
    e10_scaffolded/results.csv
    e12_flyem/results.csv
    e13_selective/results.csv

figures/                       # 300-dpi paper figures
mb-silencing-prereg.md         # OSF pre-registration document
PAPER_DRAFT.md                 # full paper draft
LICENSE                        # MIT
requirements.txt
```

## How to reproduce the paper

The four CSVs in `experiments/results/` are committed and contain the exact
data reported in the paper. To reproduce them from scratch:

```bash
# E7v2 pre-registered null (60 runs, ~5-6h on 2 workers)
python -m experiments.e7v2_full_mb_silencing --seeds 30 --workers 2
python -m experiments.analyze_e7v2

# E10 dissociation (random wiring, 32 runs)
python -m experiments.e10_scaffolded --seeds 8 --workers 2

# E12 FlyEM-substituted dissociation (16 runs, requires FlyEM data — see below)
FLYEM_DATA_DIR=$(pwd)/data/flyem/exported-traced-adjacencies-v1.2 \
  python -m experiments.e12_flyem_danger --seeds 8 --workers 2

# E13 selective pathway lesions (48 runs, ~5-6h on 2 workers)
python -m experiments.e13_selective_lesion --seeds 8 --workers 2
python -m experiments.analyze_e13

# Regenerate all figures
python -m experiments.generate_paper_figures
```

All simulation runs are deterministic given the integer seed, so reproduction
should yield bit-identical CSVs on the same Python/NumPy/SciPy versions.

## FlyEM data (separate download)

The Janelia FlyEM hemibrain v1.2.1 dataset (~150 MB) is **not** in this
repository. To run E12 (FlyEM-substituted), download it from
[https://www.janelia.org/project-team/flyem/hemibrain](https://www.janelia.org/project-team/flyem/hemibrain),
specifically the `exported-traced-adjacencies-v1.2.tar.gz` archive, and
extract into `data/flyem/exported-traced-adjacencies-v1.2/`. Then set the
environment variable `FLYEM_DATA_DIR` to that path before running E12.

The expected files in that directory are `traced-neurons.csv`,
`traced-roi-connections.csv`, and `traced-total-connections.csv`.

## Pre-registration

The central appetitive null hypothesis test (E7v2) was pre-registered on the
Open Science Framework before any of the 60 simulation runs were started.
The registration is permanently archived under DOI
[10.17605/OSF.IO/397GJ](https://doi.org/10.17605/OSF.IO/397GJ) with the
pre-registration document included at `mb-silencing-prereg.md`. The locked
confirmation criteria for the appetitive null are: Bonferroni-corrected
p > 0.05 AND |Cohen's d| < 0.5 AND JZS BF_01 > 3 (r = 0.707). The full E7v2
results.csv was not opened until all 60 runs had completed.

E12 used a code-comment hypothesis (d ≤ -3 for the danger transfer with the
FlyEM substitution) locked into the script before data collection but not
deposited on OSF; we report it as confirmatory only for that single
criterion. E13 (selective pathway lesions) is exploratory.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you build on this work, please cite the pre-registration DOI and the paper
(citation block to be filled at acceptance):

```
Cerni, E. (2026). The trained mushroom body is dispensable for aversive
transfer in an embodied insect-brain agent with the real Drosophila
connectome. [TODO: venue and DOI at acceptance].
Pre-registration: OSF DOI 10.17605/OSF.IO/397GJ.
```
