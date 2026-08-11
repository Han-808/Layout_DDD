# Final Benchmark Metric Scoring Specification

This document is the authoritative scoring specification. It replaces all
earlier metric-scoring proposals in this repository. Implementations MUST use a
versioned scoring profile and MUST preserve the raw findings needed to
reconstruct every reported score.

The scoring layer consumes validated deterministic and VLM findings. It does
not change grouping, discovery, camera selection, evidence acquisition, Judge
semantics, metric ownership, or the public evaluation interfaces, and it MUST
NOT add VLM calls solely for scoring.

## 1. Scoring profiles

Two explicit profiles are defined. They answer different research questions
and MUST NOT be mixed in one leaderboard.

| Profile ID | L1 | L2 | L3 | Intended use |
| --- | ---: | ---: | ---: | --- |
| `intrinsic_validity_v1` | 30 | 0 | 70 | Promptless intrinsic scene validity |
| `prompt_conditioned_quality_v1` | 20 | 20 | 60 | Prompt/reference-conditioned generation quality |

Missing L2 MUST NOT trigger silent weight renormalization. A run without an L2
task selects `intrinsic_validity_v1` explicitly:

\[
S_{\mathrm{final}}=100\left(0.30S_{L1}+0.70S_{L3}\right).
\]

The prompt-conditioned profile uses:

\[
S_{\mathrm{final}}=100\left(0.20S_{L1}+0.20S_{L2}+0.60S_{L3}\right).
\]

All layer and metric scores are in `[0, 1]`; the final scalar is in `[0, 100]`.

## 2. Internal metric weights

### 2.1 L1 physical validity

Collision, support, and out-of-bounds are equally weighted inside L1:

\[
S_{L1}=\frac{S_{\mathrm{collision}}+S_{\mathrm{support}}+
S_{\mathrm{OOB}}}{3}.
\]

Under `intrinsic_validity_v1`, each L1 metric therefore owns 10 final points.

### 2.2 L3 implicit visual validity

The fixed L3 ratio is:

| L3 metric | Internal L3 weight | Final weight in `intrinsic_validity_v1` |
| --- | ---: | ---: |
| Scale consistency | 12% | 8.4 |
| Style consistency | 12% | 8.4 |
| Object-pairing consistency | 12% | 8.4 |
| Functional consistency | 44% | 30.8 |
| Semantic-placement consistency | 20% | 14.0 |

\[
\begin{aligned}
S_{L3}={}&0.12S_{\mathrm{scale}}
+0.12S_{\mathrm{style}}
+0.12S_{\mathrm{pairing}}\\
&+0.44S_{\mathrm{functional}}
+0.20S_{\mathrm{placement}}.
\end{aligned}
\]

The integer ratio is the stored configuration. Expanded absolute weights are
derived values and MUST NOT be maintained as independent configuration.

## 3. Canonical scene denominator

Every metric uses the same deterministic scene-object denominator:

\[
N_{\mathrm{scene}}=
\left|\text{canonical evaluation object IDs}\right|.
\]

The deterministic adapter freezes the ordered object-ID list before metric
evaluation and writes it to the run manifest. Each semantic asset instance is
counted once, including fixtures, wall-mounted objects, hanging objects, and
articulated objects. Exclude:

- room-shell geometry;
- cameras, lights, and rendering helpers;
- benchmark annotations and render proxies;
- empty transform nodes;
- duplicate mesh parts belonging to one semantic asset instance;
- duplicate RGB, overlay, contour, or other evidence representations.

There is no denominator based on applicable objects, directed objects,
discovered relations, groups, probes, evidence images, or Judge calls. Those
structures are used only for routing and audit.

## 4. Immutable burden ledger

L3 MUST NOT use binary metric scoring. L1 and L3 both emit event records that
can be represented as object-equivalent burden.

For a confirmed-invalid event `e` with reliable normalized magnitude
`M(e) in [0, 1]`:

\[
P(e)=0.4+0.6M(e).
\]

Rules:

- a valid event has burden `0`;
- a minimum confirmed-invalid event has burden `0.4`;
- a maximum-severity event has burden `1.0`;
- when no reliable magnitude exists, use `0.4` and record magnitude as
  unavailable;
- when only a validated proxy exists, use it and label the magnitude as proxy;
- VLM confidence MUST NOT affect burden;
- semantic object category or perceived importance MUST NOT change object
  weight.

For metric `m`, allocate each event to its scoring target objects and compute:

\[
B_m=\sum_o\min\left(1,\sum_{e\rightarrow o}P_o(e)\right).
\]

`1.0` burden means one fully invalid object-equivalent. One object contributes
at most `1.0` to one metric, even if several findings affect it.

### 4.1 Attribution and relation burden

- A unary defect is charged to its scoring target object.
- A relation defect is charged to the responsible endpoint when responsibility
  is identifiable.
- If responsibility cannot be isolated, split one total event burden across
  the minimum repair set. Do not charge one full burden to every endpoint.
- Blockers, context objects, and relation partners are causal evidence, not
  automatic additional penalties.

### 4.2 Deduplication and metric ownership

- Global, cross-group, group-local, retry, and repair observations of the same
  underlying defect merge into one scored event.
- Stable defect identity and ownership references MUST be retained in the
  burden ledger.
- A Functional ownership event suppresses only the exact same causal failure
  from Placement scoring. An independent Placement failure on the same object
  remains scoreable.
- Physical violations belong to L1; wrong scene or ensemble membership belongs
  to Object Pairing; impaired ordinary use belongs to Functional; remaining
  implausible position or orientation belongs to Placement.

## 5. Metric deduction

Metric-specific `n_m` values are retained. A shared universal coefficient is
not used.

| Metric | `n_m` |
| --- | ---: |
| Collision | 2 |
| Support | 3 |
| OOB | 3 |
| Scale | 3 |
| Style | 2 |
| Object pairing | 2 |
| Functional | 2 |
| Placement | 2 |

These coefficients are explicit saturation parameters and also affect local
score sensitivity. Every audit report MUST therefore expose both the nominal
weight `W_m` and the effective local factor `W_m n_m`.

The prevalence deduction is:

\[
D_{\mathrm{prevalence},m}
=\min\left(1,n_m\frac{B_m}{N_{\mathrm{scene}}}\right).
\]

To prevent a severe defect from disappearing in a dense scene, use the
metric-independent worst-event floor:

\[
D_{\mathrm{floor},m}=0.25P_{\max,m},
\]

where `P_max,m` is the largest confirmed event burden before responsibility is
split across relation endpoints.

The final metric deduction and score are:

\[
D_m=\max\left(D_{\mathrm{prevalence},m},D_{\mathrm{floor},m}\right),
\qquad S_m=1-D_m.
\]

The two deductions are combined with `max`, not addition, so the same event is
not charged twice. A metric can lose all of its own weight but cannot consume
another metric's allocation.

Under `intrinsic_validity_v1`, one maximum-severity event has the following
minimum final-score effect from the floor:

| Metric | Minimum final-point deduction |
| --- | ---: |
| Collision | 2.5 |
| Support | 2.5 |
| OOB | 2.5 |
| Scale | 2.1 |
| Style | 2.1 |
| Object pairing | 2.1 |
| Functional | 7.7 |
| Placement | 3.5 |

Prevalence can produce a larger deduction in sparse scenes or when several
independent object-equivalents are invalid.

## 6. Metric-specific burden semantics

### 6.1 Collision

Let `D` be penetration depth along the minimum-overlap axis and `T_a`, `T_b`
the two projected object thicknesses along that axis:

\[
T_{\min}=\min(T_a,T_b),\qquad T_{\mathrm{thin}}=0.04\ \mathrm{m}.
\]

\[
R_c=
\begin{cases}
D/T_{\min},&T_{\min}\geq0.04\ \mathrm{m},\\
D/\sqrt{T_aT_b},&T_{\min}<0.04\ \mathrm{m}.
\end{cases}
\]

\[
M_c=\operatorname{clip}\left(\frac{R_c}{0.20},0,1\right).
\]

Reliable containment or deep intersection may set `M_c=1`. A collision pair
is one relation defect; split its burden across endpoints unless causal
ownership identifies one responsible object.

### 6.2 Support

Let `G` be robust positive clearance, `T_contact` the existing contact
tolerance, and `H` object height:

\[
G_{\mathrm{excess}}=\max(0,G-T_{\mathrm{contact}}),
\qquad
G_{\mathrm{sat}}=\max(0.04\ \mathrm{m},0.15H).
\]

\[
M_s=\operatorname{clip}\left(
\frac{G_{\mathrm{excess}}}{G_{\mathrm{sat}}},0,1
\right).
\]

A reliably missing support path may set `M_s=1`. Deduplicate by unsupported
causal root; descendants remain audit-visible but are not charged again for
the same failure.

### 6.3 Out of bounds

For boundary face `f`, let `p_f` be effective penetration after floor-contact
tolerance and `E_normal(f)` the object extent along the face normal:

\[
r_f=\frac{p_f}{E_{\mathrm{normal}(f)}},
\qquad
R_{\mathrm{OOB}}=\max_f r_f.
\]

\[
M_{\mathrm{OOB}}=
\operatorname{clip}\left(\frac{R_{\mathrm{OOB}}}{0.20},0,1\right).
\]

One object contributes at most one OOB object-equivalent even when it crosses
multiple boundary faces.

### 6.4 Scale consistency

Controlled categories are `oversized`, `undersized`, and
`relative_scale_mismatch`.

| Severity | Burden |
| --- | ---: |
| `noticeable` | 0.4 |
| `gross` | 1.0 |

When a trusted size interval `[s_min,s_max]` exists:

\[
q=\max\left(1,\frac{s}{s_{\max}},\frac{s_{\min}}{s}\right),
\qquad
M_{\mathrm{scale}}=
\operatorname{clip}\left(\frac{\log q}{\log2},0,1\right).
\]

A native asset bounding box alone is not a trusted real-world size prior.

### 6.5 Style consistency

Controlled categories are `style_outlier` and `style_cluster_conflict`.

| Severity | Burden |
| --- | ---: |
| `noticeable` | 0.4 |
| `gross` | 1.0 |

Charge the minimum object set whose replacement restores coherence. Ordinary
material, color, age, or functional-category variation alone is not a style
defect.

### 6.6 Object-pairing consistency

Controlled categories are `out_of_context_object` and
`incompatible_object_set`. Object Pairing has no mild invalid tier:

\[
P_{\mathrm{pairing}}=
\begin{cases}
0,&\text{valid},\\
1,&\text{invalid}.
\end{cases}
\]

For an incompatible set of individually plausible objects, split one total
burden across the minimum repair set. The denominator remains `N_scene`, not
the number of possible pairs or discovered relations.

### 6.7 Functional consistency

Functional consistency asks whether ordinary intended use remains possible in
the authored arrangement. The typed checks currently include:

- `directional_correspondence`;
- `relative_use_geometry`;
- `architecture_orientation`;
- `clearance`;
- `local_affordance_confirmation`.

| Severity | Burden |
| --- | ---: |
| `impaired` | 0.4 |
| `blocked` | 1.0 |

`impaired` means core use remains possible through ordinary use motions but is
materially degraded. `blocked` means core use requires translating,
reorienting, removing, or otherwise rearranging a placed scene object. Opening
an articulated part or pulling out a chair as part of ordinary use is not
layout reconfiguration.

Charge the object whose function is impaired or the responsible relation
endpoint. Blockers and architectural boundaries remain causal evidence unless
they are independently responsible scoring targets. Forced invalid with no
reliable severity uses `impaired=0.4`.

### 6.8 Semantic-placement consistency

Placement concerns positional plausibility that remains after physical,
membership, and functional ownership are resolved. Its typed checks are
limited to:

- `support_and_height`;
- `scene_zone`;
- `contextual_anchor`.

`contextual_anchor` is non-operational. Watching, operating, opening,
approaching, reaching, or other action-required relations belong to
Functional.

| Severity | Burden |
| --- | ---: |
| `atypical` | 0.4 |
| `implausible` | 1.0 |

Only the placement check's subject object is a scoring target. Context IDs are
used for evidence and explanation, not automatic penalties. A check marked
`excluded_function_owned` contributes zero Placement burden and MUST reference
the exact Functional ownership event. Forced invalid with no reliable
severity uses `atypical=0.4`.

## 7. Terminal-state rules

`need_more_evidence` and ambiguity are internal states, not completed metric
results. At evidence-budget exhaustion, a scientifically completed Judge path
must make a forced binary choice and record at least:

```json
{
  "forced_binary": true,
  "evidence_ambiguous": true,
  "stop_reason": "budget_exhausted"
}
```

Infrastructure, protocol, schema, rendering, or endpoint failure is not a
scientific verdict. It fails the affected run or metric and MUST NOT be
converted into valid, invalid, or zero burden.

If an auxiliary camera selector fails after a decodable packet has already
passed EvidenceGate, the existing packet remains scientifically usable. The
Controller may make one final forced-binary Judge call on that retained packet
and MUST record the selector failure as a degraded terminal choice. If no
validated packet exists, or if the final Judge transport/schema call fails,
the scope is an infrastructure failure rather than a scientific result.

Every required global, cross-group relation, group-local, and typed-check
scope must therefore terminate as `evaluated`, `evaluated_degraded`, or
`infrastructure_failure`. A final scientific `unresolved` state is a control
contract violation, not a third verdict.

## 8. Required run outputs

Every scored run MUST persist enough immutable information for exact post-hoc
reconstruction without invoking a VLM:

- scoring profile ID and resolved profile configuration;
- frozen `N_scene` and ordered canonical object IDs;
- raw validated findings and evidence references;
- event category, severity, magnitude, magnitude source, and burden;
- affected, causal, context, and scoring-target IDs;
- ownership, relation-split, and deduplication references;
- per-object capped burdens and total `B_m`;
- `P_max,m`, prevalence deduction, floor deduction, and final metric score;
- nominal metric weight, `n_m`, `W_m n_m`, saturation burden, and maximum
  metric contribution;
- individual metric scores and the L1/L2/L3 vector;
- canonical final scalar;
- forced-binary rate, evidence-ambiguity rate, unresolved state, and
  infrastructure failures.

The post-hoc scorer is a pure projection over these records. It MUST NOT alter
raw verdicts, re-run discovery, request evidence, or call the Judge.

## 9. Validation requirements

Tests MUST verify:

1. profile weights and internal ratios are normalized and versioned;
2. no-L2 runs select `intrinsic_validity_v1` without silent renormalization;
3. L3 is reconstructed from burden rather than binary metric verdicts;
4. global/local/relation duplicates do not create multiple penalties;
5. one object contributes at most one object-equivalent per metric;
6. relation splitting preserves one total event burden;
7. exact Function-owned failures are excluded from Placement while independent
   Placement failures remain scoreable;
8. effective `W_m n_m`, saturation, and floor calculations match the reported
   score;
9. semantic object-importance weighting is absent;
10. a saved run can reproduce every metric, layer, and final score without a
    VLM call;
11. infrastructure failure never becomes a scientific valid/invalid result;
12. object-count sensitivity is reported for representative scene sizes and
    for fixed-defect scenes with additional unrelated valid objects.

The canonical scalar is intended for ranking, but benchmark reports MUST also
publish the complete metric vector, layer vector, burden totals, severity
distribution, and operational reliability statistics.
