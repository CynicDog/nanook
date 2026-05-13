# Red-Team Cross-Validation Review — SDC Methods & Metrics

Author: independent code review (Opus 4.7, 1M-context run, 2026-05-12).
Scope: every SDC method and metric currently under `src/nanook/`.
References used:
- Canonical pseudocode in `../pseudonymization-proposal/pseudo_code/`.
- Primary papers in `../pseudonymization-proposal/papers/`.
- Hundepool et al., *Handbook on Statistical Disclosure Control* (CASC, 2010).

## Calibration rule

A simplification of the canonical pseudocode is **not** a finding when the
implementation's docstring honestly describes what it does. Findings are
reserved for:

1. The docstring contradicting the pseudocode.
2. The method name implying canonical behaviour that the code doesn't deliver.
3. The simplified output breaking downstream contracts (sign flips, biased
   weights, hidden nondeterminism, etc.).

Severity ladder: **MAJOR → MINOR → NIT**. The reviewer-agent pass flagged
several "BLOCKER" items that did not survive close reading; they are listed
under [Rejected claims](#rejected-claims) for audit.

---

## Summary

| ID  | Severity | Area                         | One-liner |
|-----|----------|------------------------------|-----------|
| M1  | RESOLVED | `multiplicative_noise`       | Log-normal multiplier + Höhne moment rescaling now implemented; `intensity` param renamed `sigma_log`. |
| M2  | RESOLVED | `microaggregation` (MDAV)    | Second-cluster seed switched to `arg max d(x_i, x[far_idx])`; step 2/3 of the pseudocode now visually distinct in code. |
| M3  | RESOLVED | `massc`                      | Full 4-step pipeline (categorical MDAV → mode-tuple collapse → within-cluster permute → subsample → rake) replaces the previous within-QI swap. |
| M4  | RESOLVED | `Pipeline.seed`              | `apply` spawns one `SeedSequence` substream per step from `self.seed`; explicit per-step seeds still win. |
| m1  | MINOR    | `noise_addition`             | Uncorrelated case only. |
| m2  | MINOR    | `top_bottom_coding`          | `clip` mode only. |
| m3  | MINOR    | `local_suppression`          | Greedy heuristic only (no ILP). |
| m4  | MINOR    | `sampling`                   | Bernoulli approximation, not SRS-WoR. |
| m5  | MINOR    | `data_swapping`              | Unrestricted random pairs, not similarity-based. |
| m6  | MINOR    | `rank_swapping`              | Sequential iteration + uniform-from-window. |
| m7  | MINOR    | `rounding`                   | `random_within_bin` (jitter) ≠ pseudocode `random_unbiased` (probabilistic). |
| n1  | NIT      | `pram`                       | `dict(value_counts.iter_rows())` is version-fragile. |
| n2  | NIT      | `kl_divergence`              | Log base = nats; no API to change. |

Metrics — `k_anonymity`, `l_diversity` (distinct / entropy / recursive),
`t_closeness` (ordinal CDF + nominal LP), `lambda_measure`, `il1s`,
`kl_divergence` (Laplace smoothing) — all match their pseudocode. Edge cases
(empty frames, constant columns, all-null sensitive) verified.

---

## MAJOR findings

### M1: Multiplicative noise — RESOLVED

**Files**

- Implementation: `src/nanook/core/perturbative/multiplicative_noise.py:45-61`
- Pseudocode: `../pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/multiplicative_noise.md` (steps 1–3)
- Paper: Höhne (2004), *Varianten zur multiplikativen Anonymisierung*, referenced as the canonical source.

**What the code does**

```
factors = 1.0 + rng.normal(0.0, intensity, df.height)   # line 55
... pl.col(col).cast(pl.Float64) * pl.Series("_nk_factor", factors) ...
```

A Gaussian-centred-at-1 multiplier is drawn per row and applied directly.
Zeros are preserved. No standardisation, no rescaling.

**What the pseudocode requires**

1. Draw `Z_{ik}` from a **positive** distribution (log-normal centred at 1,
   e.g. `LogNormal(0, σ_log²)` so `E[Z] = exp(σ_log²/2)`).
2. Compute original moments `μ_X,k, σ_X,k` and raw perturbed moments
   `μ_{X^a},k, σ_{X^a},k`.
3. Apply moment-preserving rescaling
   `X^{aR}_{ik} := (σ_X,k / σ_{X^a},k)(X^a_{ik} - μ_{X^a},k) + μ_X,k`.

**Why it matters**

- With `intensity ≥ 0.25` or so, `1 + N(0, intensity)` produces non-trivial
  negative-factor draws. Sign flips on protected values violate the
  multiplicative-noise contract (the whole reason this method exists is to
  protect strictly-positive variables like income and turnover without
  generating impossible negatives — *exactly* the property the implementation
  loses).
- Without step 3, `E[Z] ≠ E[X]` and `Var[Z] ≠ Var[X]`. The advertised "first
  two moments restored" guarantee from Höhne 2004 does not hold.

**Suggested resolution**

Either implement the canonical method (log-normal multiplier + rescale) or
rename and re-document the existing one (e.g. `gaussian_factor_perturbation`)
to remove the implication that this is Höhne's multiplicative noise. The
inline reference at the top of `multiplicative_noise.py:5` should match
whichever choice is made.

**Resolution**

Canonical Höhne implemented at `multiplicative_noise.py:64-86`. The `intensity`
param is now `sigma_log`; the `Pipeline.multiplicative_noise` fluent helper is
updated accordingly. Zeros still short-circuit to identity (documented). Tests:
`test_multiplicative_noise_preserves_first_two_moments` and
`test_multiplicative_noise_never_flips_sign_on_positive_input`.

---

### M2: Microaggregation MDAV — RESOLVED

**Files**

- Implementation: `src/nanook/core/perturbative/microaggregation.py:100-125` (`_mdav_cluster`)
- Pseudocode: `../pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/microaggregation.md` (step 2, lines 27–28)
- Paper: Domingo-Ferrer & Mateo-Sanz (2002), *Practical Data-Oriented
  Microaggregation*, IEEE TKDE 14(1):189–201
  (`papers/perturbative/Domingo-Ferrer-Mateo-Sanz-2002-practical-data-oriented-microaggregation.pdf`).

**What the code does**

```
while remaining.size >= 2 * k:
    centroid = x[remaining].mean(axis=0)           # line 101
    far_idx  = remaining[argmax(||x[remaining] - centroid||)]
    # First cluster: k nearest to far_idx (correct)
    ...
    if remaining.size >= 2 * k:
        d_centroid_2 = ||x[remaining] - centroid|| # line 115 — REUSES old centroid
        far2 = remaining[argmax(d_centroid_2)]
        # Second cluster: k nearest to far2
```

Both seeds use distance to the **original** centroid `c̄ = mean(U)` computed
before the first cluster was removed.

**What the pseudocode requires**

Step 2 (lines 23–29) of the MDAV pseudocode:

> While `|U| ≥ 3 · k`:
> - Compute centroid `c̄ ← mean({x_i : i ∈ U})`.
> - Find `r ← arg max_{i ∈ U} d(x_i, c̄)`.
> - Form `G_r` from `r` and its `k-1` nearest neighbours.
> - Find `s ← arg max_{i ∈ U \ G_r} d(x_i, x_r)`. **← farthest from `x_r`, not from centroid.**
> - Form `G_s` from `s` and its `k-1` nearest neighbours.
> - Append `G_r, G_s` to `P`; update `U ← U \ (G_r ∪ G_s)`.

**Why it matters**

The implementation still produces a valid k-anonymous partition because every
group has `k` members and the residual group at the end picks up everything
left over. But the partition is **not** MDAV: the second cluster is anchored
to whichever residual point happens to be far from the *pre-removal* centroid,
not far from the first seed `r`. SSE / SST (the native utility metric the
docs/12 chapter cites for microaggregation) is sensitive to this choice and
the resulting partition is in general suboptimal compared to the published
MDAV variant that all the benchmark tables in §4 of Domingo-Ferrer &
Mateo-Sanz 2002 use.

**Suggested resolution**

Replace line 115 with `d_far = np.linalg.norm(x[remaining] - x[far_idx], axis=1)`
and pick `far2 = remaining[int(np.argmax(d_far))]`. The rest of the block
(nearest-k pick on line 118) is already correct.

**Resolution**

Fixed at `microaggregation.py:93-138` — the second-cluster seed now uses
`arg max d(x_i, x[far_idx])`, and the loop is split into step-2 (`|U| ≥ 3k`)
and step-3 (`|U| ≥ 2k`) phases that match the pseudocode line-for-line.
Test: `test_microaggregation_second_seed_is_farthest_from_first` (drives
`_mdav_cluster` directly with a hand-crafted 2-D matrix where canonical and
pre-fix disagree on a swing point's cluster assignment).

---

### M3: MASSC — RESOLVED

**Files**

- Implementation: `src/nanook/core/perturbative/massc.py:46-88`
- Pseudocode: `../pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/massc.md` (steps 1–4)
- Paper: Singh, Yu & Dunteman (2003), *MASSC: a new data mask for limiting
  statistical information loss and disclosure*.

**What the code does**

```
qis = list(ctx.quasi_identifiers)
rows_by_group: dict[tuple, list[int]] = {}
for i, row in enumerate(df.select(qis).iter_rows(named=True)):
    key = tuple(row[c] for c in qis)
    rows_by_group.setdefault(key, []).append(i)

for indices in rows_by_group.values():
    ...
    picked = rng.choice(indices, size=n_swap, replace=False).tolist()
    rng.shuffle(picked)
    swaps.extend((picked[i], picked[i + 1]) for i in range(0, n_swap, 2))
# apply() swaps self.column values pairwise
```

Records are bucketed by exact QI tuple, and pairs are swapped on
`self.column` — the **sensitive** column declared on the step.

**What the pseudocode requires**

1. **Micro-Agglomeration.** Cluster records into k-anonymous groups via a
   categorical-distance clustering (typically MDAV with Hamming or
   Torra-2004 distance). Assign each cluster a representative key tuple
   `q_g`; every cluster member gets `q_g` for its key columns. The dataset
   is now k-anonymous on `K`.
2. **Substitution.** For each cluster and each record, draw a cluster-mate
   uniformly at random and replace **`x_{i,K}` with `x_{j,K}`** — i.e. swap
   the key/QI tuple, breaking the deterministic original→masked link while
   keeping the within-cluster distribution unchanged.
3. **Subsampling.** Sample `n_sub = ⌊f_sub · n⌋` records without replacement
   from the substituted dataset.
4. **Calibration.** Compute post-stratification weights `w_i` (raking or
   GREG) so the weighted marginals of the calibration variables match the
   population totals.

**Why it matters**

The current implementation:

- Skips step 1 entirely. No k-anonymity is enforced; if a QI tuple appears
  only once in the input, that record is dropped from swap candidacy
  (`m < 2` branch at line 68) and remains uniquely identifiable.
- Operates on `self.column`, which by convention is the **sensitive**
  column. The pseudocode substitutes the **key tuple**. These have
  opposite disclosure/utility profiles.
- Skips steps 3 and 4; the result therefore has no design-weight column.

What's delivered is "within-QI-group categorical perturbation on the
sensitive column". The method name MASSC, the file's pseudocode reference,
and the public API (`Pipeline.massc`) all overpromise.

**Suggested resolution**

Two acceptable paths:

- Rename to `within_group_swap` (or similar), update docstring and remove
  the MASSC pseudocode reference, and document the actual disclosure /
  utility profile.
- Or implement all four steps: clustering via `Microaggregation` with a
  categorical distance, substitute on the QI tuple, subsample, calibrate
  through raking. This is a significant rewrite.

**Resolution**

Full 4-step rewrite. New helpers:

- `_internal/categorical_mdav.py` — Hamming-distance MDAV with mode-tuple
  centroid; carries the M2 second-seed fix.
- `_internal/raking.py` — iterative proportional fitting for the calibration
  step.

`massc.py` now: (1) clusters by Hamming MDAV on QI codes and collapses each
cluster's QI tuple to the per-column mode (this is what delivers k-anonymity
on the QIs); (2) permutes non-QI columns within each cluster so the
deterministic original→masked link is broken; (3) subsamples
`floor(f_sub * n)` records; (4) rakes design weights against the original
frame's population totals on the calibration variables. The public API now
takes `k`, `f_sub`, `calibration_vars` (drops `column` and `fraction`); the
`Pipeline.massc` fluent helper signature changes accordingly. Tests:
`test_massc_output_is_k_anonymous_on_qis`, `test_massc_subsample_size`,
`test_massc_weights_sum_to_population_total_per_calibration_cell`,
`test_massc_substitution_breaks_qi_linkage`.

---

### M4: Pipeline seed — RESOLVED

**Files**

- Declaration: `src/nanook/pipeline.py:77,188`
- Apply loop (does not use seed): `src/nanook/pipeline.py:220-235`
- Per-step seed usage: `multiplicative_noise.py:54`, `sampling.py:55`,
  `noise_addition.py:61`, `rank_swapping.py:66`, `data_swapping.py:55`,
  `pram.py:80`, `massc.py:57`, `rounding.py:59`, `resampling.py:53`.

**What the code does**

`Pipeline.__init__` takes an optional `seed`, stores it on `self.seed`, and
round-trips it through `to_dict`/`from_dict`. The field is **never read**
inside `apply()`. Each stochastic step instead reads its own
`step.params["seed"]`.

**Why it matters**

Users will reasonably expect `Pipeline(seed=42)` to make the whole pipeline
deterministic — it's the only `seed=` parameter exposed at the top level,
and the docstring at `pipeline.py:73-80` does not disclaim propagation. As
written, that constructor argument is dead weight: the only way to get
reproducible output is to set a `seed=` on every stochastic step
individually. Pipelines built fluently with `Pipeline(seed=42).sampling(fraction=0.5)`
silently get OS-entropy randomness.

**Suggested resolution**

In `Pipeline.apply`, derive per-step substreams from `self.seed` using
`np.random.SeedSequence(self.seed).spawn(len(self.steps))` (or
`generate_state`) and inject them where `step.params.get("seed")` is
currently `None`. Explicit per-step seeds should win — document the
precedence. Alternative: drop the field entirely and require explicit
per-step seeds. Either is fine; the current half-state is the worst of both.

**Resolution**

`Pipeline.apply` now spawns one `SeedSequence` substream per step from
`self.seed` and injects each into the corresponding `effective_params["seed"]`
when (and only when) the step has no explicit `seed=`. Explicit per-step
seeds win. Precedence is documented in `Pipeline.__init__`. Tests:
`test_pipeline_with_seed_is_deterministic_without_per_step_seeds`,
`test_pipeline_seed_changes_propagate_to_stochastic_steps`,
`test_explicit_step_seed_wins_over_pipeline_seed` (in
`tests/integration/test_pipeline_reproducibility.py`).

---

## MINOR findings

These are documented simplifications under the calibration rule — listed for
transparency only. None requires immediate action.

### m1: `noise_addition` — uncorrelated case only

`src/nanook/core/perturbative/noise_addition.py:51-63` implements the
uncorrelated noise model `Σ_ε = α · diag(σ_k²)`. The pseudocode
(`noise_addition.md`, mode list) also offers correlated (Kim 1986) and
linear-transform (Brand 2002) variants. The docstring is honest about being
the uncorrelated variant.

### m2: `top_bottom_coding` — `clip` only

`src/nanook/core/perturbative/top_bottom_coding.py:85-91` applies
`pl.col.clip(lower, upper)`. The pseudocode also lists `plus_marker`
(append a marker indicating clipping happened) and `interval` (replace
clipped values with the truncated interval string) modes. The docstring
does not advertise the latter two.

### m3: `local_suppression` — greedy only

`src/nanook/core/perturbative/non_perturbative/local_suppression.py:52-82`
implements only the greedy heuristic. The pseudocode also describes an ILP
formulation. The docstring (line 3) explicitly calls itself "Greedy
heuristic", which satisfies the calibration rule.

### m4: `sampling` — Bernoulli, not SRS-WoR

`src/nanook/core/perturbative/non_perturbative/sampling.py:51-60` performs
independent Bernoulli inclusion (`rng.random < fraction`), not SRS without
replacement. The HT weight `1/π = 1/fraction` it emits is mathematically
correct for Bernoulli, so downstream analytic estimators remain unbiased —
the deviation from the pseudocode is a *sampling-design* simplification, not
a weight bug. The docstring at lines 1–10 makes this explicit.

### m5: `data_swapping` — unrestricted random pairs

`src/nanook/core/perturbative/data_swapping.py:48-67` selects pairs
uniformly at random over the whole frame. Canonical data swapping
(Dalenius & Reiss 1978; pseudocode lines 23–30) restricts swaps to pairs
that are close on declared background variables. Marginal distribution is
preserved; joint distribution is randomised. The docstring describes the
implemented behaviour and does not invoke Dalenius–Reiss.

### m6: `rank_swapping` — sequential + uniform-from-window

`src/nanook/core/perturbative/rank_swapping.py:70-74` iterates rank
positions in fixed order and picks the swap partner uniformly from
`[lo, hi]`. The pseudocode prescribes random rank order plus
nearest-in-window. The two converge on most real distributions; this is a
convention-divergent simplification, not a correctness bug.

### m7: `rounding` — `random_within_bin` ≠ `random_unbiased`

`src/nanook/core/perturbative/rounding.py:56-61` exposes
`random_within_bin=True`: deterministic round + uniform jitter in
`[-base/2, base/2]`. The pseudocode's `random_unbiased` mode is
*probabilistic rounding*: round down with probability `1 - f`, up with
probability `f`, ensuring `E[z] = x`. Implementation's jitter has
`E[jitter] = 0`, so `E[z] = round(x/b) · b`, which is **biased** toward the
nearest multiple. Analysts who need unbiased totals (the classical use case
in official-stats cell-count rounding) currently have no mode that gives it.

---

## NIT

### n1: `pram` — `value_counts` row-format fragility

`src/nanook/core/perturbative/pram.py:65`:

```
counts = df.get_column(col).value_counts(...)
marginal_counts = dict(counts.iter_rows())
```

This works in `polars==1.13.1` (pinned in `pyproject.toml:20`) where
`value_counts` returns a 2-column frame iterable as `(value, count)`. A
column-name-keyed access (`counts["count"] / counts[col]`) is safer across
polars versions.

### n2: `kl_divergence` — base of log

`src/nanook/metrics/utility/kl_divergence.py:75` uses `math.log` (natural
log). `report.py:146` documents the choice ("per-column divergences in
nats"). The pseudocode is base-flexible; expose a `base: float = math.e`
parameter (or accept a `units: {"nats", "bits"}` argument) so analysts can
ask for bits.

---

## Rejected claims

For audit: these were flagged by the reviewer-agent's first pass and
**withdrawn** after closer reading.

- *local_suppression "cost_priority is inverted vs docstring"*. Verified:
  `score = delta / cost` at `local_suppression.py:131`; greedy picks
  `max(score)`; therefore higher `cost` lowers the score and **discourages**
  selection of that column for suppression — exactly what the docstring at
  line 38 promises.

- *sampling "HT weights wrong for Bernoulli"*. For Bernoulli inclusion with
  probability `π = fraction`, the HT weight is `w = 1/π = 1/fraction`. That
  is what `sampling.py:59` emits. The sampling-design simplification
  (Bernoulli vs SRS-WoR) is captured separately as `m4`.

- *rank_swapping "permutation construction broken"*. Re-traced at
  `rank_swapping.py:69-79`: `permutation` indexes into `sorted_values`, and
  swapping `permutation[a], permutation[b]` correctly exchanges which sorted
  value ends up at which rank. The gather + scatter is sound.

- *l-diversity "recursive (c, l) off-by-one"*. `l_diversity.py:123-129` is
  0-indexed correctly: with `freqs` sorted descending, `freqs[l-1]` is
  `r_l` and `freqs[l-1:]` is the tail `[r_l, …, r_m]`. The early-return on
  `len(freqs) < l` is also correct.

---

## Metrics — fidelity confirmed

| Metric          | Pseudocode | Verdict |
|-----------------|------------|---------|
| `k_anonymity`   | `risk_metrics/k_anonymity.md` | Matches. Group-by QIs, filter classes `< k`, count violations and uniques. |
| `l_diversity`   | `risk_metrics/l_diversity.md` | All three modes (distinct, entropy, recursive (c, l)) implemented to the formula. Entropy uses natural log per pseudocode convention. |
| `t_closeness`   | `risk_metrics/t_closeness.md` | Ordinal path computes `Σ |F_p − F_q| · Δ_v` on the sorted support. Nominal path solves the standard transportation LP via `_internal/emd.py:emd_nominal`. |
| `lambda_measure`| `information_loss_metrics/lambda_measure.md` | `λ_k = (1/n) Σ |x_ik − z_ik| / R_k`; file-level `λ = mean_k λ_k`. Zero-range columns contribute 0. |
| `il1s`          | `information_loss_metrics/il1s.md` | `IL1s_k = (1/n) Σ |x_ik − z_ik| / s_k`; constant columns excluded; sample std (no ddof override). |
| `kl_divergence` | `information_loss_metrics/kl_divergence.md` | Laplace smoothing `(q_count · n + ε) / (n + ε · m)` applied to the protected marginal; `P = 0` terms skipped. |

Helpers:

- `src/nanook/_internal/emd.py` — both paths reviewed; the ordinal sum
  uses `cumsum(p)[:-1]` and `cumsum(q)[:-1]` against `deltas`, which is
  the correct interior-boundary integration.
- `src/nanook/_internal/grouping.py` — `equivalence_class_sizes` and
  `per_record_class_size` use polars `group_by` which treats nulls as one
  group; this matches the pseudocode convention but is currently
  untested (see [T2](#t2)).

---

## Test backlog

These reproduce as findings of their own — the user explicitly asked
coverage to be in scope. Each item lists the target test file. All five
are now landed; the bonus item below is an incidental bug uncovered by T2.

### T0 (incidental, RESOLVED)

`_internal/grouping.py:per_record_class_size` used a left-join on the QI
columns, which silently drops null-keyed rows because SQL null-equality is
unknown. T2's null-handling tests exposed this; the helper was rewritten
to use `pl.len().over(qis)`, which partitions nulls together.

### T1 — RESOLVED

- [ ] Add hand-computed golden-value tests, one per metric, against examples in the handbook.
- Today: `tests/unit/test_lambda_measure.py:17-21` (`handbook_example_one_tenth`) is the only one.
- Add to:
  - `tests/unit/test_il1s.py` — small frame where `s_k` is hand-computable.
  - `tests/unit/test_kl_divergence.py` — two 3-bin categoricals.
  - `tests/unit/test_k_anonymity.py` — pin violations + sample_uniques on a 6-record hand-traced frame.
  - `tests/unit/test_l_diversity.py` — entropy against `H = -Σ p log p`.
  - `tests/unit/test_t_closeness.py` — ordinal EMD against a 3-point support summed by hand.

### T2 — RESOLVED

- [x] Null-handling tests for QIs and the sensitive column.
- `_internal/grouping.py:11-17` claims to handle nulls but no test
  asserts it. Add `tests/unit/test_grouping.py` (or extend the existing
  metric tests) with frames where:
  - a single QI column has nulls (does the equivalence class with a null
    have the size the pseudocode implies?),
  - the sensitive column has nulls (does `l_diversity.distinct` exclude
    them per the pseudocode convention?).

### T3 — RESOLVED

- [x] Method ↔ native-metric round-trip integration tests.
- Already partially exists: `tests/unit/test_perturbative.py:77-82`
  asserts microaggregation → k-anonymity. Extend:
  - `local_suppression` → `k_anonymity` (holds at `target_k`).
  - `pram` → `kl_divergence` (close to zero at `retention=1`, monotonic in
    `1 − retention`).
  - `noise_addition` → `lambda_measure` (monotonic in `intensity`).
- Add as parametrised tests in `tests/integration/test_method_metric_pairs.py`.

### T4 — RESOLVED

- [x] Property-based tests with `hypothesis` (already in dev extras,
  `pyproject.toml:32`):
  - multiset preservation for `data_swapping` and `rank_swapping`,
  - `sampling(fraction=1.0)` is identity,
  - `noise_addition` `λ` is monotonically non-decreasing in `intensity`,
  - `microaggregation` output is k-anonymous for every `k ∈ [2, 10]`.

### T5 — RESOLVED

- [x] Pipeline reproducibility integration test.
- Build a `Pipeline` from a JSON payload, apply it twice on the same
  input, assert frame equality. This test should pass *only* after [M4](#m4-pipeline-seed)
  is resolved — wire it in deliberately as a regression for that fix.

---

## Appendix — reproducing each verification

| Finding | How to reproduce |
|---------|------------------|
| M1      | `python -c "import numpy as np; r = np.random.default_rng(0); print((1 + r.normal(0, 0.5, 100_000) < 0).mean())"` — observe ~2 % negative factors at `intensity=0.5`. Then `pl.read_*` a positive-only column, apply `MultiplicativeNoise(intensity=0.5)`, observe sign flips. |
| M2      | Read `microaggregation.py:100-125` and `microaggregation.md` lines 27–28 side-by-side; trace `n = 3k + 1` records on paper and observe the second-cluster seed disagrees. |
| M3      | `grep -nE "(agglomerat|subsampl|calibrat)" src/nanook/core/perturbative/massc.py` → no hits; `grep -nE "sensitive|self\.column" src/nanook/core/perturbative/massc.py` → confirms the swap operates on `self.column`. |
| M4      | `grep -n "self\.seed" src/nanook/pipeline.py` returns 3 hits (init, `to_dict`, `from_dict`). None inside `apply()`. |

---

## Acceptance criteria for the follow-up work

A subsequent PR per MAJOR finding should produce:

1. A `tests/...` regression test that fails on `main@<commit-this-review-was-pushed-on>` and passes after the fix.
2. Updated docstrings if the fix changes observable behaviour.
3. A note in this REVIEW.md (or its successor) marking the finding resolved.
