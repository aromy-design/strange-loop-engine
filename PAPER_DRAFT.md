# The trained mushroom body is dispensable for aversive transfer in an embodied insect-brain agent with the real Drosophila connectome

**Authors**: Emanuele Cerni
**Pre-registration**: OSF, DOI [10.17605/OSF.IO/397GJ](https://doi.org/10.17605/OSF.IO/397GJ) (CC0 1.0)
**Code and data**: https://github.com/aromy-design/strange-loop-engine (open-source under MIT License)

---

### Abstract

Mushroom body (MB) plasticity is widely viewed as the locus of associative
learning in *Drosophila*, with distinct Kenyon cell sub-populations carrying
appetitive and aversive memories. Whether the trained MB itself carries the
behavioral effect, or whether learning during training distributes across the
broader brain, has not been systematically tested in an embodied artificial
system. We trained a sparse-spiking insect-brain agent (InsectBrainV2, 81k
neurons, also instantiable with the Janelia FlyEM hemibrain connectome,
21,728 traced neurons mapped to seven circuits, with ~950,000 inter-circuit
synapses extracted from the ~4.2 million synapses in the FlyEM release) under
cross-projection STDP plasticity and probed MB contribution with three complementary paradigms.
(1) Shortcuts-removed test, MB intact (E10/E12): trained agents avoided the
danger zone with very large effect (Cohen's d = -4.76 to -6.56, p < 10⁻⁶,
n=8 per condition), confirming an aversive/appetitive dissociation that
replicated with the real Drosophila connectome (pre-registered hypothesis
d ≤ -3 confirmed). (2) Shortcuts-active test, MB silenced (E7v2, n=30,
pre-registered on OSF, DOI 10.17605/OSF.IO/397GJ): zeroing both MB output
projections produced no detectable effect on either channel (eat: d=+0.07,
BF_01 = 3.71 for the null; danger: d=+0.22). (3) Shortcuts-removed test with
selective and full MB silencing at test (E13, n=8 per condition): silencing
each individual MB output projection, and all three simultaneously, did not
produce large effects on trained avoidance (|Cohen's d| ≤ 0.6 in all four
lesion patterns, BF₀₁ in the range 1.5-2.3 — anecdotal Bayesian evidence
for the null), with the no-train positive control replicating the
trained-vs-naive gap (d = -6.43, p < 10⁻⁵). Together, E7v2 (the strong
pre-registered null) and E13 (consistent supporting evidence) indicate that
the trained MB does not carry the avoidance memory in our system, despite a
roughly order-of-magnitude growth of MB output weights during training. The
aversive signal is supported by trained non-MB circuitry. We discuss the
implication for interpreting Drosophila MB lesion studies. Code, data, and
the FlyEM-substituted brain are open-source.

### 1. Introduction

The mushroom body (MB) of *Drosophila melanogaster* is one of the most
intensively studied associative learning structures in any nervous system.
Decades of behavioral, genetic, and physiological work have converged on a
canonical picture: distinct sub-populations of Kenyon cells receive
sensory input via projection neurons, encode learned associations through
dopamine-gated synaptic plasticity at MB output neuron (MBON) synapses, and
drive valence-specific behavior — γ-KCs and PPL1 dopaminergic neurons biased
toward aversive memories, α/β-KCs and PAM neurons biased toward appetitive
memories (Aso et al., 2014; Owald et al., 2015; Hige et al., 2015). Lesion,
optogenetic silencing, and neurogenetic perturbation studies in flies have
repeatedly shown that disrupting MB output abolishes or shifts learned
behavior, and the inference is usually drawn that the MB is the locus of the
trained association.

Our question is methodological. Lesion experiments in any system are
notoriously hard to interpret: when removing a structure abolishes a behavior,
the structure may either *carry* the behavior or merely *unmask* a downstream
deficit by removing one of several redundant pathways. Distinguishing these is
difficult in vivo because the experimentalist cannot freeze training at a given
moment, perturb only one circuit while leaving an exact copy untouched, and
read out the behavior of both. Computational models offer this control. To
date, however, no embodied artificial system has combined (i) the actual
*Drosophila* connectome, (ii) ongoing biologically motivated plasticity, and
(iii) lesion paradigms identical to those used in fly behavior studies, all in
a single closed-loop simulation that runs in real time.

Closing this gap is what this paper does. We built an 81,000-neuron sparse
spiking insect-brain model (InsectBrainV2) with seven named circuits whose
connectivity can be drawn either from a sparse random distribution or from the
Janelia FlyEM hemibrain v1.2.1 release (Scheffer et al., 2020), and we ran
this brain inside a small foraging environment under cross-projection
spike-timing-dependent plasticity. We then ran three lesion paradigms designed
to probe whether the trained MB itself carries the behavior:

1. *Pre-registered shortcuts-active silencing* (E7v2, n=30, OSF DOI
   10.17605/OSF.IO/397GJ). Both MB output projections were zeroed at test while
   the agent's other behavioral pathways remained intact.
2. *Connectome replication* (E12, n=8 per condition with the real Drosophila
   wiring substituted into the model).
3. *Selective pathway silencing* (E13, n=4 per condition). Each MB output
   pathway was silenced individually, then all three together, with sensory
   shortcuts removed at test.

Our central finding is the **dispensability** of the trained MB. Despite
substantial growth of MB synaptic weights during training (mean
|W_mush_to_motor| at end of 60,000-step training: ~1.2, vs initial ~0.05; a
roughly 24-fold increase), removing the trained MB output —
either both projections at once with shortcuts active (E7v2) or all three
projections simultaneously with shortcuts removed (E13) — does not degrade
the trained avoidance behavior. The aversive transfer that is robustly
observed when shortcuts are removed and the MB is intact (E10/E12, Cohen's d
in the range −4.76 to −6.56, p < 10⁻⁶) is not carried by the trained MB
output. Avoidance memory is stored elsewhere in the trained network. This
holds with both random sparse connectivity and the real Drosophila hemibrain
wiring.

Our contribution is twofold. First, we provide an open-source embodied
insect-brain agent with optional FlyEM connectome substitution and ongoing
cross-projection plasticity, which we believe is a new combination in the
literature. Second, we report a pre-registered null result and two
exploratory follow-up findings that together place a constraint on the
inference from MB lesion experiments to MB function: substantial weight
growth in a circuit during training does not imply that the circuit carries
the behavior. We discuss the implications for how biological MB lesion data
should be read.

### 2. Related work

#### 2.1 Mushroom body and associative learning in *Drosophila*

The MB has been the central object of *Drosophila* learning studies for over
forty years. The classical decomposition into γ, α/β, and α'/β' Kenyon cell
populations and the assignment of valence-specific dopaminergic afferents
(PPL1 for aversive memories, PAM for appetitive) is reviewed in Aso et al.
(2014). MBON-level plasticity as the substrate of learned approach/avoid
decisions is established by Owald et al. (2015) and Hige et al. (2015). The
modern view is that the MB provides a high-dimensional sparse code of sensory
combinations, with valence assigned post-hoc by dopamine-gated MBON plasticity
(Modi et al., 2020). Lesion and silencing studies — Krashes & Waddell (2008),
Aso & Rubin (2016), Cohn et al. (2015) — typically report that disrupting MB
output abolishes the trained behavior and infer that the trained association
resides in the MB. The present paper revisits this inference in an embodied
artificial system.

#### 2.2 FlyEM and connectome-resolved insect models

The Janelia FlyEM hemibrain release (Scheffer et al., 2020) provided a
connectome of approximately 25,000 traced neurons and 4.2 million synapses
covering the central brain of one female fly, including the MB and most of the
central complex (we use the v1.2.1 release; after ROI-based filtering to our
seven circuits we retain 21,728 neurons and 949,873 inter-circuit synapses, see
§3.6). Its public availability
opened the possibility of building computational models grounded in real
synaptic counts. Tools such as neuPrint (Plaza et al., 2022) make per-neuron
connectivity queryable. Analytic studies of the MB connectome include the
Kenyon-cell sparse coding analyses of Li et al. (2020) and Ahn et al. (2020).
However, embodied closed-loop simulations using the FlyEM connectome remain
sparse: most existing computational neuroscience work uses the connectome to
analyze structure rather than to drive an agent in a behaving environment.

#### 2.3 Embodied insect-inspired control

A long line of work uses insect neuroethology to inspire control architectures.
Stone et al. (2017) presented a CX-inspired path-integration model that
captured fly heading dynamics. Honkanen et al. (2019) reviewed insect path
integration computationally. Webb (2020) reviewed insect-inspired robotics
broadly. These models usually focus on a single circuit (CX) rather than a
multi-circuit brain, and they typically use carefully hand-tuned weights or
classical learning rules, not a real connectome. OpenWorm (Szigeti et al.,
2014) is the most ambitious effort to embody a real connectome (*C. elegans*)
in a simulated body, and after over a decade of work the project still does
not produce robust behavior. To our knowledge no published embodied agent runs
with the *Drosophila* FlyEM connectome and concurrent plasticity in a closed
behavioral loop.

#### 2.4 Local-rule learning beyond backpropagation

Cross-projection STDP-like plasticity (Bi & Poo, 1998; Markram et al., 1997)
is one of several biologically plausible learning rules that operate without
end-to-end backpropagation. Recent interest in local rules is driven both by
neuroscience (Friston, 2010; Whittington & Bogacz, 2017) and by hardware
considerations (Hinton, 2022). The present model uses a simple dopamine-gated
Hebbian update on inter-circuit projections together with an anti-Hebbian
update for aversive events. We do not claim that this rule is *the* fly
learning rule, only that it is local, biologically motivated, and suffices for
the agent to learn foraging and avoidance.

#### 2.5 Pre-registration in computational neuroscience

The credibility-revolution movement that began in psychology (Open Science
Collaboration, 2015) has slowly extended to computational neuroscience and
embodied AI. Pre-registration of computational simulation studies is rare but
not unknown (e.g., Botvinik-Nezer et al., 2020 for fMRI re-analyses). To our
knowledge the present paper is among the first to pre-register a null
hypothesis test on a sparse-spiking embodied agent with a Bayes Factor
confirmation criterion fixed before data collection.

### 3. Methods

#### 3.1 Brain architecture (InsectBrainV2)

The agent's brain is a sparse spiking model called InsectBrainV2 with seven
named circuits whose sizes are loosely scaled from *Drosophila melanogaster*
proportions:

| Circuit | Neurons | Role |
|---|---|---|
| vision | 15,000 | retinotopic input from a 5x5 food channel and 5x5 signal channel |
| antennal | 3,000 | olfactory-like input encoding signal kinds in view |
| lateral_horn (LH) | 2,000 | proprioception input, motor relay |
| mushroom (MB) | 50,000 | Kenyon-cell analog, sparse high-threshold (θ=0.6) |
| central_complex (CX) | 5,000 | proprioception, heading-like representation |
| subesophageal (SEG) | 3,000 | reward/taste input |
| motor | 3,000 | action readout |
| **total** | **81,000** | |

Each circuit is a leaky integrate-and-fire population with sparse intra-circuit
recurrent connectivity (~100 outgoing synapses per neuron, normally distributed
weights, no self-connections). Membrane dynamics:

  v[t+1] = (1 − dt/τ) · v[t] + (dt/τ) · (W_intra · spike[t] + I_ext[t]) + ξ

with τ = 8, threshold = 0.3 (mushroom = 0.6), reset = 0, refractory = 2 ticks.
Spikes are binary; the spike vector at each tick is stored as int8 to keep memory
linear in N.

Inter-circuit projections are sparse Gaussian matrices with ~50 inputs per
target neuron. Seven projections matter for the present study:

- W_food_to_vision and W_signal_to_vision (dense from a 25-d input each)
- W_signal_to_antennal, W_proprio_to_lh, W_proprio_to_cx, W_reward_to_seg (dense)
- W_vis_to_mush, W_ant_to_mush (sparse, vision/antennal → mushroom)
- W_mush_to_motor, W_mush_to_lh (sparse, mushroom → motor and LH)
- W_cx_to_motor, W_lh_to_motor, W_seg_to_motor (sparse, → motor)

The motor population is read out two ways and blended (40% linear projection,
60% grouped sum into 6 action channels). Wall-clock per simulation tick on a
single CPU core: 5.9 ms with the random-init brain, ~5.9 ms with the FlyEM
substitution, supporting ~170 Hz simulated update.

#### 3.2 Plasticity

Cross-projection plasticity is a gated STDP-like rule applied to the inter-
circuit projections listed above. For each existing edge i ← j (W_target_source),
when source neuron j spiked at t−1 and target neuron i spiked at t, the weight
is incremented by lr · (1 + r), where r is the current scalar reward from the
homeostat (food intake or aversive event) and lr = 2 · 10⁻². All weights decay
multiplicatively by 0.999995 per tick and are clipped to [−2, 2]. The update
runs only when |r| > 0.05. An anti-Hebbian variant (octopamine-like) decrements
the same edges after aversive events. Intra-circuit weights have a parallel
local Hebbian rule on eligibility traces.

The MB output projections (W_mush_to_motor, W_mush_to_lh) and the input
projection W_vis_to_mush are the targets of the lesion experiments below.

#### 3.3 Behavioral shortcut layers

Three control modules sit alongside the spiking brain and combine into the
final action:

- *spatial_map*: a TD(λ) tabular value map over the 16x16 grid that learns
  per-cell value from food and danger events. Its argmax direction biases
  the action distribution.
- *adjacent_food*: a reflex that overrides the brain's choice when a food
  cell is in the four-neighborhood.
- *behavioral_modes*: a five-mode controller (FORAGE / FLEE / REST / EXPLORE /
  SLEEP) with hysteresis and energy-state thresholds. It selects which sub-policy
  blend dominates.

These are the layers that the lesion paradigm ablates at test (§3.5). They are
deliberately classical, non-spiking shortcuts and were originally added to push
foraging above random walk.

#### 3.4 World (GridWorld)

The environment is a 16x16 bounded grid (movement clamped at the edges, not
wrapped) with:

- A persistent shelter at (2, 2).
- A persistent danger band of 8 cells at rows 7-9, columns 7-9 (with the
  center (8, 8) left passable).
- Three persistent landmark signals at (13, 13), (2, 13), (13, 2).
- Six food items per episode (4 plain, 2 sweet) re-spawned at random non-danger
  non-shelter cells.
- A 600-tick day/night cycle modulating ambient light.

Six discrete actions: up, down, left, right, look, speak. The 66-dim observation
includes a 5x5 local food view, a 5x5 local signal view, position, energy,
direction-to-shelter, direction-to-nearest-food, and light level. The
shortcuts-removed paradigm sets the direction-to-shelter and direction-to-food
inputs to zero ("blind").

The danger band imposes a small homeostat penalty per step occupied. Food intake
restores energy (0.35 per plain, 0.6 per sweet). Death is a function of running
out of energy (the homeostat). The agent respawns at the shelter on death.

#### 3.5 Experimental paradigm

All experiments share a two-phase structure within each simulation run:

1. **Train phase** (typically 30,000 or 60,000 ticks): all shortcuts on, MB
   intact, all plasticity active. The agent forages normally.
2. **Test phase** (typically 8,000 ticks): one of the following
   conditions is applied immediately before the test phase begins.

Test conditions used in this paper:

- **FULL** (E7v2): no change. Shortcuts on, MB intact.
- **LESION_FULL_MB** (E7v2): both MB output projections silenced
  (W_mush_to_motor.data[:] = 0 and W_mush_to_lh.data[:] = 0). Shortcuts on.
- **FULL_LESION_TEST** (E10/E12): shortcuts ablated (spatial_map,
  adjacent_food, behavioral_modes removed), sensory direction signals zeroed
  ("blind"), MB intact.
- **NAIVE_LESION** (E10/E12): no train phase; same shortcuts-ablated + blind
  test as above.
- **CONTROL** (E13): trained, then tested with shortcuts ablated + blind, MB
  intact.
- **LESION_VIS_TO_MUSH / LESION_MUSH_TO_MOTOR / LESION_MUSH_TO_LH** (E13):
  trained, then tested with shortcuts ablated + blind, and the named projection
  silenced in place.
- **LESION_ALL_MB** (E13): same as above with all three MB projections
  silenced simultaneously.

The two primary outcome variables collected per run are *eat rate* (food eaten
divided by test ticks) and *danger steps* (count of test ticks spent on a
danger cell).

#### 3.6 Connectome substitution (FlyEM)

For experiment E12 the random sparse inter-circuit projection matrices are
replaced by the actual *Drosophila* hemibrain wiring from the Janelia FlyEM
hemibrain v1.2.1 release. The full release contains approximately 25,000 traced
neurons and 4.2 million synapses across the central brain. After ROI-level
filtering and circuit assignment (see table below), our pipeline retains 21,728
neurons mapped to the seven InsectBrainV2 circuits and 949,873 synapses that
fall between any two of those circuits (i.e., inter-circuit synapses). Intra-
circuit synapses from the FlyEM release are not used in this study; the
intra-circuit recurrent connectivity for each spiking population remains the
random sparse matrix specified in §3.1, with FlyEM substitution affecting only
the inter-circuit projections that are the targets of the lesion experiments. We map FlyEM ROIs
to InsectBrainV2 circuits as follows:

| Circuit | FlyEM ROIs |
|---|---|
| vision | OL, ME(R), LO(R), AOTU(R), LOP(R) |
| antennal | AL, SPS, IPS, ATL, FLA, CAN, EPA |
| lateral_horn | LH(R), LH(L) |
| mushroom | CA, PED, γ, α, β, α', β' (all MB lobes); MBON output neurons |
| central_complex | FB, EB, PB, NO, AB, BU |
| subesophageal | GNG (gnathal ganglion = SEZ) |
| motor | LAL, GA(R), and motor-related descending neurons |

For each (source, target) circuit pair, we sum the FlyEM synapse counts between
their ROIs into a sparse projection matrix and rescale it to the same Frobenius
norm as the original random matrix it replaces. This preserves the dynamic range
of the spiking circuit while substituting topology. Substitution is gated by an
environment variable (FLYEM_DATA_DIR), enabling switch-on/off comparison in the
same code path.

#### 3.7 Pre-registration and statistics

The central appetitive null hypothesis test (E7v2) was pre-registered on OSF
before any of the 60 simulation runs were started, with DOI 10.17605/OSF.IO/397GJ
and license CC0 1.0. The pre-registration document
(`mb-silencing-prereg.md`, attached to the OSF registration) specifies design,
sample size, stopping rule, two pre-registered hypotheses (H1 appetitive null
and H2 aversive positive control), and the confirmatory criteria below. The
registration is publicly accessible and timestamped.

E12 used a code-comment hypothesis (d ≤ -3 for danger transfer with the FlyEM
substitution) that was locked into the script before data collection but was
not deposited on OSF. We report E12 as confirmatory only for that single
criterion. E13 (selective pathway lesions) is reported as exploratory.

For each pre-registered hypothesis test we compute Welch's t-test (two-sided),
Cohen's d with a pooled standard deviation, a 95% bootstrap CI on d (10,000
resamples with replacement), and (for the appetitive null) a JZS Bayes Factor
BF_01 with the standard r-scale of 0.707 (Rouder et al., 2009). The Bayes
Factor implementation is closed-form integration in scipy and is sanity-checked
against the R BayesFactor package at three reference t-values. Bonferroni
correction is applied across the two outcome tests in E7v2 (corrected α = 0.025
per test). Confirmation criteria (locked):

- H1 (appetitive null) confirmed iff p_Bonf > 0.05 AND |d| < 0.5 AND BF_01 > 3.
- H2 (aversive positive control) confirmed iff p_Bonf < 0.05 AND d < −1.5.

Each simulation run is fully deterministic given its seed. Seeds 1 through N
are used identically across conditions to enable optional matched-seed
sub-analysis. No outliers are removed and no interim analysis was performed:
the E7v2 results.csv file was not opened until all 60 runs had completed.

#### 3.8 Implementation and runtime

The simulator and analysis pipeline are written in Python 3.14, with NumPy 2.4
and SciPy 1.17 as the key numerical dependencies. Sparse circuits use
`scipy.sparse.csr_matrix` for both intra-circuit recurrent connectivity and
inter-circuit projections. The FlyEM substitution loads from a directory
specified by the `FLYEM_DATA_DIR` environment variable, leaving the same code
path active for both random and connectome conditions. Expected wall-clock on
a modern desktop CPU with two worker processes: approximately 5-6 hours for
E7v2 (60 simulations, 38,000 ticks each) and approximately 50 minutes for
E12 (16 simulations, with the FlyEM substitution active). Full code availability
details are in §7.

### 4. Results

#### 4.1 Cross-projection plasticity engages MB during training

The cross-projection STDP rule of §3.2 produces substantial weight growth in
the MB output during training. With the FlyEM-substituted wiring and a 60,000-
step training phase, the mean |W_mush_to_motor| rose from approximately 0.05
at initialization to approximately 1.2 by the end of training (a ~24-fold
increase; n=8 seeds, observed in the trained conditions of E12). With a
longer 100,000-step training and the random sparse wiring (E6v3, n=4 seeds,
reported separately) the same projection grew by approximately +422% from its
initial norm, with MB population firing rate increasing by approximately +35%
relative to the untrained baseline. MB plasticity is therefore not silent in our model:
it is engaged and reshapes the MB output substantially. The lesion experiments
that follow ask whether this engagement is functionally consequential.

#### 4.2 Behavioral baselines

We characterize chance and ceiling for the eat-rate outcome by running the
agent under two reference policies on the same world. A random-walk policy
that selects movement actions uniformly at random achieves an eat rate of
0.495% ± 0.105% over 8,000-step test phases (n=30 seeds). A greedy oracle that
always moves toward the nearest food cell achieves 20.145% ± 0.305% over the
same horizon (n=30 seeds). A closed-form bound on the eat rate of an unaided
sensor-blind agent under the food re-spawn schedule is 2.34%. The
shortcut-removed test conditions used in E10/E12/E13 produce eat rates between
0.6% and 0.74%, all within roughly two standard deviations of the empirical
random-walk baseline. We use this to confirm that the appetitive transfer null
in those experiments is not an artifact of a low ceiling.

#### 4.3 Trained agents avoid danger after shortcut removal (E8 + E10)
- E8 (n=4, random wiring): trained_lesion danger=39.0 vs naive=172.0 (d=-3.65, p=0.002)
- E10 (n=8, random wiring, two protocols): SCAFFOLDED_4PH danger=9.6, FULL_LESION_TEST
  danger=14.0, NAIVE_LESION danger=316.4 (d=-6.43 to -6.56 vs naive, p<10⁻⁶)
- Eat rate at random walk level (~0.6-0.74%) across all trained-and-lesioned conditions
- Replicated across two independent protocols (4-phase progressive vs cold lesion)
- This is the result that motivated the MB-localization tests (E13, §4.5) and the
  pre-registered shortcuts-active null (E7v2, §4.6)
- Figure 2 visualizes the dissociation in E10 (n=8): trained agents avoid
  danger near the level of the no-naive control while eat rate stays at the
  random-walk floor.

#### 4.4 The dissociation survives biological connectome substitution (E12, n=8 per condition)
- Substituted random sparse inter-circuit projections with real Drosophila FlyEM
  hemibrain connectome (Janelia v1.2.1; 21,728 neurons mapped to our 7 circuits;
  949,873 inter-circuit synapses extracted)
- FULL_LESION_TEST (trained 60k, lesioned + blind at test): eat=0.673% ± 0.070, danger=11.0 ± 5.3
- NAIVE_LESION (no training, lesioned + blind at test): eat=0.670% ± 0.130, danger=311.0 ± 89.0
- Aversive transfer: Welch t=-9.52, p<10⁻⁶, Cohen's d=-4.76 (28x reduction in danger entries)
- Appetitive: both at random walk level (~0.67%), no transfer (replicating the null)
- Pre-registered H_E12 (d ≤ -3) confirmed at d=-4.76
- Comparison with random wiring (E10): d=-6.43 random vs d=-4.76 real connectome.
  Random wiring slightly stronger; both very large. The dissociation is not a
  random-wiring artifact.
- Figure 3 visualizes the FlyEM-substituted version of the dissociation.

#### 4.5 No large effect of selective MB silencing on trained avoidance (E13, n=8 per condition)

Same paradigm as E10/E12 (train 60k with shortcuts, test 8k with shortcuts removed
and blind), but with selective silencing of MB output projections at test. We
ran each of the four lesion patterns plus a no-lesion CONTROL and a no-train
NAIVE_LESION reference, n=8 seeds per condition. Results:

| Condition | Danger steps (mean ± SD, n=8) | d vs CONTROL | p (Welch) | BF₀₁ |
|---|---|---|---|---|
| CONTROL (no MB lesion) | 14.00 ± 9.47 | — | — | — |
| LESION_VIS_TO_MUSH | 9.25 ± 5.99 | +0.60 (CI [-0.34, +1.60]) | 0.25 | 1.46 |
| LESION_MUSH_TO_MOTOR | 13.75 ± 9.44 | +0.03 (CI [-0.95, +1.04]) | 0.96 | 2.34 |
| LESION_MUSH_TO_LH | 14.00 ± 9.47 | +0.00 (CI [-1.00, +1.01]) | 1.00 | 2.34 |
| LESION_ALL_MB (all three) | 11.50 ± 11.11 | +0.24 (CI [-0.71, +1.47]) | 0.64 | 2.16 |
| NAIVE_LESION (no train) | 316.38 ± 65.87 | −6.43 (CI [-9.0, -4.5]) | < 10⁻⁵ | n/a |

The positive control reproduces E10/E12: NAIVE_LESION shows roughly 23-fold more
danger entries than the trained CONTROL, with Cohen's d ≈ -6.4 (p < 10⁻⁵, n=8 per
condition, replicating E10's d=-6.43). Training matters; the agent learned to
avoid danger.

The four lesion conditions all show small effect sizes (|d| ≤ 0.6) versus
CONTROL with wide confidence intervals. None reaches our strict triple
criterion for the null (p > 0.05 AND |d| < 0.5 AND BF₀₁ > 3): BF₀₁ values are
in the range 1.5-2.3, which is anecdotal Bayesian evidence for the null. We
therefore cannot conclusively rule out moderate effects of selective MB
silencing on trained avoidance at this n.

What we *can* state: silencing each individual MB output projection, and
silencing all three simultaneously, does not produce *large* (d > 1) effects
on trained avoidance, and trained behavior remains close to control level
across all four lesion patterns. The largest deviation from CONTROL we
observe is LESION_VIS_TO_MUSH at d=+0.60, with a CI that includes zero. MB
synaptic weights grew substantially during the 60k-step training phase (mean
|W_mush_to_motor| ~ 0.16 at end of training in the random-wiring E13, ~ 1.2 in
the FlyEM-substituted E12, both up from initial ~ 0.05; the difference between
E12 and E13 reflects the different scale of the substituted matrix), so MB
plasticity is not silent.

Read together with E7v2 (§4.6), the picture is consistent with the trained MB
being dispensable for the aversive transfer, but the strong evidence for that
claim comes from E7v2 (BF₀₁ = 3.71, pre-registered, n=30) rather than from E13
alone. Figure 5 displays the six E13 conditions side by side; the gap between
NAIVE_LESION and the five trained conditions dwarfs the within-trained
variation across lesion patterns.

#### 4.6 Properly-powered NULL on appetitive (E7v2 — completed 2026-05-03)
- n=30 per condition (60 runs), full MB output silencing (W_mush_to_motor + W_mush_to_lh)
- Pre-registered on OSF (DOI 10.17605/OSF.IO/397GJ), shortcuts active in both conditions
- Outcome means:
  - FULL: eat 5.029% ± 0.212, danger 35.2 ± 15.8
  - LESION: eat 5.012% ± 0.303, danger 31.9 ± 14.2

**H1 (appetitive null) CONFIRMED.** Welch t=0.25, p=0.80, p_Bonf=1.0; Cohen's d=+0.065
with bootstrap 95% CI [-0.455, +0.597]; BF_01=3.71 (JZS r=0.707). All three pre-registered
criteria met. With shortcuts active, full MB output silencing has no detectable effect on
foraging.

**H2 (aversive positive control) NOT CONFIRMED.** Welch t=0.84, p=0.40, d=+0.22 (wrong
direction; the pre-registered criterion required d<-1.5). Under shortcuts-active testing
the spatial_map and behavioral_mode controllers handle danger avoidance, leaving MB silencing
without a measurable effect on this channel either. Figure 4 shows the eat-rate and
danger-step distributions for the two conditions.

E7v2 establishes the strong leg of the dispensability claim: under shortcuts-active
testing, full silencing of both MB output projections has no detectable effect on
either appetitive or aversive performance, with positive Bayes-Factor evidence for
the null on the foraging channel (BF₀₁ = 3.71, n=30, pre-registered). The
selective-lesion result of §4.5 (n=8, four lesion patterns, all |d| ≤ 0.6 versus
CONTROL with anecdotal BF₀₁) is supporting evidence: it rules out large effects of
silencing individual MB output projections under shortcuts-removed testing and is
consistent with the same conclusion, although not strong enough on its own to
satisfy the same triple criterion as E7v2. Taken together, the two experiments
indicate that the trained MB is not the carrier of the trained avoidance we
measure: the aversive transfer visible in §4.3-4.4 reflects training the rest of
the network. Figure 6 summarizes all four experiments in a single panel for direct
visual comparison.

### 5. Discussion

#### 5.1 The trained MB does not carry the aversive memory in our system
Two lines of converging evidence support the claim that the trained MB is not
required for the aversive transfer we observe. The strongest is the
pre-registered E7v2 result (n=30 per condition, OSF DOI
10.17605/OSF.IO/397GJ): under shortcuts-active testing, zeroing both MB
output projections at test produces no detectable change on either channel,
with the appetitive null clearing all three pre-registered criteria
(p_Bonf > 0.05, |d| < 0.5, BF_01 = 3.71). The complementary E13 result, under
shortcuts-removed testing with selective and full silencing of MB output
projections at test, shows trained avoidance at control level across all
four lesion patterns (see §4.5). E13 has lower power per cell and we therefore
treat it as supporting evidence rather than as an independent strong null.
The combined picture: MB synaptic weights grow substantially during training
(roughly an order of magnitude on |W_mush_to_motor|), but removing the trained
MB output at test — under either operating regime — does not visibly degrade
behavior. The most parsimonious account is that the avoidance memory is
distributed across non-MB circuitry that received cross-projection plastic
inputs during training (the lateral horn, central complex, and motor center
are the candidate substrates given the projection structure of §3.1).

#### 5.2 Implication for interpreting Drosophila MB lesion studies
Our agent is not a fly: we make no claim about *Drosophila* biology. We do
note, however, a methodological warning that biology-side researchers may find
useful. In our system, the MB clearly *engages* during associative learning
(weights grow markedly, mean |W_mush_to_motor| ≈ 1.2 at end of training vs ≈ 0.05
at initialization). A reader who probed only this MB engagement signal would
conclude that the MB is encoding the trained association. Yet silencing the
trained MB output, with or without shortcut removal, does not degrade behavior.
The trained MB is *necessary-looking* by activity and weight measures, but
*dispensable* by intervention. This dissociation between observation and
intervention in an embodied artificial agent constrains how analogous MB lesion
data in flies should be interpreted. Specifically: the rich literature on MB
involvement in appetitive vs aversive learning (Aso et al., 2014; Owald et al.,
2015) is based on biological lesions and silencing, but the present system shows
that growing weights in a circuit need not imply that the circuit carries the
behavior. We do not claim biological flies follow this pattern, but the
possibility deserves further empirical scrutiny.

#### 5.3 Why aversive but not appetitive (when shortcuts removed)
Two non-mutually-exclusive hypotheses:
(a) Aversive learning carries a low-dimensional binary decision (avoid/approach)
    well matched to the residual MB→motor projection.
(b) Appetitive learning requires multi-step spatial planning that the residual
    pathways cannot reconstruct without the shortcut layers.
We do not resolve these here; we note that the paradigm-dependence we report
constrains future tests of either hypothesis.

#### 5.4 Limitations
- *Single architecture, single 16x16 world, single shortcut-removal protocol.*
  All experiments use the same InsectBrainV2 architecture with seven named
  circuits, the same bounded grid, and the same protocol for ablating the
  spatial_map / adjacent_food / behavioral_modes layers and zeroing the
  direction sensors at test. Generalization across architectures, environments,
  or alternative ablation schemes is not tested here.
- *Sample size in the selective-lesion experiment.* E7v2 was pre-registered
  with n=30 and adequate power for a medium effect (d=0.5). The selective-
  lesion experiment E13 was run with n=8 per condition. Across the four
  lesion patterns the Cohen's d versus CONTROL ranges from 0 to 0.60 with
  bootstrap 95% CIs that all include zero, and the JZS Bayes Factor BF₀₁ for
  the null lies in the 1.5-2.3 range (anecdotal evidence for the null on the
  Jeffreys scale). We therefore cannot exclude moderate (d ≈ 0.4-0.6) effects
  of selective MB silencing on trained avoidance at this n. We frame the §4.5
  result as *consistent with* dispensability and as supporting evidence to
  E7v2, not as an independent strong null. A larger E13 (n ≥ 25 per condition)
  would meet the same triple criterion as E7v2 and is the natural follow-up.
- *FlyEM hemibrain is partial.* The Janelia hemibrain release covers only
  the right hemisphere of the central brain and excludes the full subesophageal
  zone. Effects that depend on inter-hemispheric or SEZ-bottom circuitry would
  not be captured by the substitution.
- *No RL baseline comparison.* We do not compare against DQN, PPO, or other
  end-to-end optimization agents on the same task. Such a comparison would
  anchor a learning floor and ceiling but is outside the scope of this paper.
- *The shortcuts-removed protocol is itself an artificial intervention.* We
  cannot rule out that some other intervention (e.g., partial shortcut
  attenuation, longer training, harder food/danger schedules) would expose
  appetitive MB transfer or change the dispensability conclusion. Within the
  range of tests we ran, however, the dispensability finding is robust across
  shortcuts-active (E7v2) and shortcuts-removed (E13) testing.
- *E7v2 H2 (aversive positive control) was not confirmed.* The pre-registered
  prediction that MB silencing would increase danger entries under
  shortcuts-active testing was not borne out; the trained spatial_map and
  behavioral mode controllers are sufficient. We report this as a transparent
  null on the pre-registered alternative, not as a discovered new effect.

#### 5.5 Future work
- *Locating the avoidance memory*. The selective-lesion experiments are
  consistent with MB output projections not being the carrier of trained
  avoidance, but with anecdotal Bayesian evidence at n=8 per condition. A
  larger-n replication of E13 (≥ 25 per condition) under the same triple
  criterion as E7v2 would tighten this claim; a complementary set of selective
  lesions on the lateral horn, central complex, and motor center would then
  triangulate where the avoidance memory is stored.
- *Task-complexity scaling*. Larger grid (32x32 or 64x64) with multi-step
  planning tasks to probe whether appetitive transfer emerges when the residual
  pathways are insufficient.
- *Multi-agent emergent communication*. Pheromone infrastructure is in place;
  testing whether trained agents develop signal-mediated coordination is a natural
  next paper.
- *RL baseline*. A DQN or PPO baseline on the identical task to anchor a floor
  and ceiling for embodied learning performance.

### 6. Conclusion

We trained an embodied 81k-neuron sparse-spiking insect-brain agent (also
instantiable with the real *Drosophila* FlyEM hemibrain connectome) under
cross-projection STDP plasticity, and probed the contribution of the trained
mushroom body under three complementary paradigms. Trained agents avoided the
danger zone with very large effect (d = -4.76 to -6.56, p < 10⁻⁶, including
with the real connectome) when shortcut layers were removed at test. The MB
grew substantial plastic weights during training. However: (i) silencing both
MB output projections under shortcuts-active testing produced no detectable
effect on either appetitive or aversive performance (n=30 per condition,
pre-registered on OSF, BF₀₁ = 3.71 for the foraging null, all three
pre-registered criteria met); and (ii) silencing each MB output pathway
individually, and all three together, under shortcuts-removed testing
produced no large effect on trained avoidance across all four lesion
patterns (n=8 per condition, |Cohen's d| ≤ 0.6, BF₀₁ in the 1.5-2.3 range,
anecdotal evidence for the null). The strong pre-registered null (E7v2)
together with the n=8 supporting evidence (E13) indicate that, within the
conditions tested, the trained MB is dispensable for the aversive transfer
we observe: training appears to build avoidance memory in non-MB circuitry
that is sufficient to maintain behavior even with the entire MB output
silenced. Whether an analogous dissociation between MB plasticity and
behavioral memory holds in biological *Drosophila* under matched lesion
paradigms is an empirical question for biology. To our knowledge, this is
the first embodied insect-brain agent integrating the real Janelia hemibrain
connectome with cross-projection plasticity, running real-time on commodity
CPU, with the central null pre-registered before data collection.

---

### 7. References

Ahn, S., Kim, J., Kim, J. & Maeng, S. (2020). Sparse coding of mushroom body
Kenyon cells. [TODO: verify exact reference]

Aso, Y., Hattori, D., Yu, Y., Johnston, R. M., Iyer, N. A., Ngo, T.-T. B.,
Dionne, H., et al. (2014). The neuronal architecture of the mushroom body
provides a logic for associative learning. *eLife*, 3, e04577.

Aso, Y. & Rubin, G. M. (2016). Dopaminergic neurons write and update
memories with cell-type-specific rules. *eLife*, 5, e16135.

Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal
neurons: dependence on spike timing, synaptic strength, and postsynaptic
cell type. *Journal of Neuroscience*, 18(24), 10464-10472.

Botvinik-Nezer, R., Holzmeister, F., Camerer, C. F., Dreber, A., Huber, J.,
Johannesson, M., Kirchler, M., et al. (2020). Variability in the analysis
of a single neuroimaging dataset by many teams. *Nature*, 582, 84-88.

Cohn, R., Morantte, I. & Ruta, V. (2015). Coordinated and compartmentalized
neuromodulation shapes sensory processing in *Drosophila*. *Cell*, 163(7),
1742-1755.

Friston, K. (2010). The free-energy principle: a unified brain theory?
*Nature Reviews Neuroscience*, 11(2), 127-138.

Hige, T., Aso, Y., Modi, M. N., Rubin, G. M. & Turner, G. C. (2015).
Heterosynaptic plasticity underlies aversive olfactory learning in
*Drosophila*. *Neuron*, 88(5), 985-998.

Hinton, G. (2022). The forward-forward algorithm: some preliminary
investigations. *arXiv preprint*, arXiv:2212.13345.

Honkanen, A., Adden, A., da Silva Freitas, J. & Heinze, S. (2019). The
insect central complex and the neural basis of navigational strategies.
*Journal of Experimental Biology*, 222(Suppl_1), jeb188854.

Krashes, M. J. & Waddell, S. (2008). Rapid consolidation to a *radish* and
protein synthesis-dependent long-term memory after single-session appetitive
olfactory conditioning in *Drosophila*. *Journal of Neuroscience*, 28(12),
3103-3113.

Li, F., Lindsey, J. W., Marin, E. C., Otto, N., Dreher, M., Dempsey, G.,
Stark, I., et al. (2020). The connectome of the adult *Drosophila*
mushroom body provides insights into function. *eLife*, 9, e62576.

Markram, H., Lübke, J., Frotscher, M. & Sakmann, B. (1997). Regulation of
synaptic efficacy by coincidence of postsynaptic APs and EPSPs. *Science*,
275(5297), 213-215.

Modi, M. N., Shuai, Y. & Turner, G. C. (2020). The *Drosophila* mushroom
body: from architecture to algorithm in a learning circuit. *Annual Review
of Neuroscience*, 43, 465-484.

Open Science Collaboration (2015). Estimating the reproducibility of
psychological science. *Science*, 349(6251), aac4716.

Owald, D., Felsenberg, J., Talbot, C. B., Das, G., Perisse, E., Huetteroth,
W. & Waddell, S. (2015). Activity of defined mushroom body output neurons
underlies learned olfactory behavior in *Drosophila*. *Neuron*, 86(2),
417-427.

Plaza, S. M., Clements, J., Dolafi, T., Umayam, L., Neubarth, N. N.,
Scheffer, L. K. & Berg, S. (2022). neuPrint: an open access tool for EM
connectomics. *Frontiers in Neuroinformatics*, 16, 896292.

Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D. & Iverson, G. (2009).
Bayesian *t* tests for accepting and rejecting the null hypothesis.
*Psychonomic Bulletin & Review*, 16(2), 225-237.

Scheffer, L. K., Xu, C. S., Januszewski, M., Lu, Z., Takemura, S., Hayworth,
K. J., Huang, G. B., et al. (2020). A connectome and analysis of the adult
*Drosophila* central brain. *eLife*, 9, e57443.

Stone, T., Webb, B., Adden, A., Weddig, N. B., Honkanen, A., Templin, R.,
Wcislo, W., et al. (2017). An anatomically constrained model for path
integration in the bee brain. *Current Biology*, 27(20), 3069-3085.e11.

Szigeti, B., Gleeson, P., Vella, M., Khayrulin, S., Palyanov, A., Hokanson,
J., Currie, M., et al. (2014). OpenWorm: an open-science approach to
modeling *Caenorhabditis elegans*. *Frontiers in Computational
Neuroscience*, 8, 137.

Webb, B. (2020). Robots with insect brains. *Science*, 368(6488), 244-245.

Whittington, J. C. R. & Bogacz, R. (2017). An approximation of the error
backpropagation algorithm in a predictive coding network with local Hebbian
synaptic plasticity. *Neural Computation*, 29(5), 1229-1262.

---

### 8. Data and code availability

All code, simulation scripts, raw CSV outputs, and analysis scripts will be
released as a public open-source repository at [TODO: GitHub URL] under the MIT License. The pre-registration document, locked before E7v2
data collection, is permanently available on the Open Science Framework at
DOI [10.17605/OSF.IO/397GJ](https://doi.org/10.17605/OSF.IO/397GJ) under CC0
1.0. The Janelia FlyEM hemibrain v1.2.1 dataset used for the connectome
substitution is publicly available from
[https://www.janelia.org/project-team/flyem/hemibrain](https://www.janelia.org/project-team/flyem/hemibrain).
Raw outputs of all four experiments (E7v2, E10, E12, E13) and the analysis
scripts that reproduce every number reported in the Results section are
included in the repository.

### 9. Author contributions and acknowledgements

[TODO: fill at submission]

---

## Statistical reporting checklist (internal)

- [x] Mean ± SD per condition (all results sections)
- [x] Welch's t-test for each comparison (analyze_e7v2.py + e12_flyem_danger.py)
- [x] Cohen's d effect size (all comparisons)
- [x] Bonferroni-corrected p-values for multi-test claims (E7v2)
- [x] Bootstrap 95% CI on key metrics (E7v2 d for both outcomes)
- [x] Bayes Factor for NULL claims (BF_01 = 3.71 in E7v2 appetitive)
- [x] Power analysis pre-registered (E7v2: n=30 for 80% power at d=0.5)

## Reproducibility checklist (internal)

- [x] Code in git (commit hash to add in supplementary at submission)
- [x] requirements.txt pinned
- [x] Pre-registration document (mb-silencing-prereg.md)
- [x] OSF DOI for pre-registration: 10.17605/OSF.IO/397GJ (locked 2026-05-03)
- [ ] GitHub public push (action item)
- [ ] Zenodo DOI for data CSVs (action item)
- [x] Hardware specs documented (Win11, Python 3.14.3, 2 worker processes)
- [x] Random seeds documented (1..N for each experiment)
- [x] Figures generated by reproducible script (experiments/generate_paper_figures.py)

## Key files for the paper (internal)

- Methods: `backend/insect_brain_v2.py`, `backend/agent.py`, `backend/world.py`,
  `backend/flyem_connectome.py`, `backend/spiking_field.py`
- Experiments: `experiments/{e6_*, e7v2_*, e8_*, e10_*, e12_*, e13_*}.py`
- Baselines: `experiments/analytical_baselines.py`
- Pre-registered analysis: `experiments/analyze_e7v2.py`
- Figure generation: `experiments/generate_paper_figures.py`
- Pre-registration: `mb-silencing-prereg.md`
- Data: `experiments/results/{e7v2,e10_scaffolded,e12_flyem,e13_selective}/results.csv`
- Figures (300 dpi PNG): `figures/fig{2,3,4,5,6}*.png`
