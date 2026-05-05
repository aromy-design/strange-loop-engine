# Pre-registration: full mushroom body silencing in an embodied insect-brain model

Author: Emanuele Cerni (independent)
Date: 2026-05-03
Project: OSF Gq8k9

## Background

In three earlier experiments (E8, E10, E12) I found that the mushroom body (MB) in our embodied insect-brain model transfers aversive learning (danger-zone avoidance) but not appetitive learning (foraging) once the behavioral shortcut layers are removed at test time. Effect sizes on the aversive side were large (Cohen's d between -3.5 and -6.6) and replicated when I substituted the random sparse projections in the InsectBrainV2 architecture with the Drosophila FlyEM hemibrain connectome (Janelia v1.2.1).

The earlier lesion test on the appetitive side (E7) used n=8 and zeroed only the direct MB→motor projection, leaving the indirect MB→LH→motor route intact. With n=8 the design has roughly 30% power for d=0.5, so the null I observed (p=0.31, d=0.53) is not informative.

This pre-registration covers a follow-up with proper power and full MB output silencing.

## Design

Two conditions, 30 seeds each. Both train 30,000 steps with all shortcuts active and the MB intact. After training, the conditions diverge for the test phase:

- FULL: no change. Run 8,000 test steps.
- LESION: set W_mush_to_motor.data and W_mush_to_lh.data to zero. Run 8,000 test steps.

Shortcut layers (spatial map, adjacent-food reflex, behavioral modes) stay active during the test phase in both conditions. This is intentional. I'm asking whether the trained MB contributes anything to behavior even when the shortcuts are doing most of the work — not whether it can stand alone.

Two outcome variables per run: eat rate over 8k test steps, and number of danger-zone entries over 8k test steps.

## Hypotheses

I expect a null on the appetitive side and a positive on the aversive side.

1. Appetitive (eat rate). Prediction: |mean(FULL.eat) − mean(LESION.eat)| / pooled_sd < 0.5, p > 0.05. With n=30 this is ~80% power for d=0.5. A clean null would mean the trained MB does not drive foraging behavior. I will also report a Bayes Factor BF_01 with a JZS prior (r=0.707). If BF_01 > 3 I claim evidence for the null.

2. Aversive (danger entries). Prediction: mean(LESION.danger) > mean(FULL.danger) by at least 1.5 pooled SDs, p < 0.05. This serves both as a positive control on the silencing manipulation and as additional evidence for the dissociation already observed in E8/E10/E12.

## Analysis plan

Welch's t-test for each outcome. Cohen's d with bootstrap 95% CI (10,000 resamples). Bonferroni correction across the two outcome tests (α = 0.025 per test). For the appetitive null, additionally a Bayes Factor.

No interim analysis. All 60 runs go to completion before the data are inspected.

## What is not pre-registered

Anything beyond the two outcomes above is exploratory: any analysis of inter-circuit firing rates during the test phase, weight evolution within condition, or seed-level correlations. These will be flagged as exploratory in the paper.

E13 (selective MB pathway lesion, also currently running) is a separate exploratory analysis and is not covered by this pre-registration. The earlier E7 (n=8) and the corresponding parts of E10 are exploratory by virtue of timing.

## Stopping rule

Run all 60 seeds (30 per condition). No optional stopping based on interim looks.

## Code, data, environment

Code: experiments/e7v2_full_mb_silencing.py (will be in the public repository at first paper submission).

Hardware: Windows 11 Pro, Python 3.14.3, NumPy 2.4.4, scipy 1.17.1, torch 2.11.0+cpu, no GPU.

Seeds: integers 1 through 30.

Estimated runtime: ~3–4 hours on the target machine (2 worker processes).
