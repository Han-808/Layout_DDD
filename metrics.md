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
| `intrinsic_validity_v2` | 30 | 0 | 70 | Promptless intrinsic scene validity |
| `prompt_conditioned_quality_v2` | 20 | 20 | 60 | Prompt/reference-conditioned generation quality |

Missing L2 MUST NOT trigger silent weight renormalization. A run without an L2
task selects `intrinsic_validity_v2` explicitly:

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

Under `intrinsic_validity_v2`, each L1 metric therefore owns 10 final points.

### 2.2 L3 implicit visual validity

The fixed L3 ratio is:

| L3 metric | Internal L3 weight | Final weight in `intrinsic_validity_v2` |
| --- | ---: | ---: |
| Scale consistency | 4% | 2.8 |
| Style consistency | 7% | 4.9 |
| Object-pairing consistency | 9% | 6.3 |
| Functional consistency | 52% | 36.4 |
| Semantic-placement consistency | 28% | 19.6 |

\[
\begin{aligned}
S_{L3}={}&0.04S_{\mathrm{scale}}
+0.07S_{\mathrm{style}}
+0.09S_{\mathrm{pairing}}\\
&+0.52S_{\mathrm{functional}}
+0.28S_{\mathrm{placement}}.
\end{aligned}
\]

The integer ratio is the stored configuration. Expanded absolute weights are
derived values and MUST NOT be maintained as independent configuration.
The v1 profile IDs remain readable for historical replay and retain the former
12/12/12/44/20 L3 ratio; new runs default to v2.

### 2.3 Metric-scoped public task context

L3 prompt context uses the versioned `l3_metric_prompt_context_v1` interface.
The evaluator freezes a bank of short public values and records an explicit
field selection for every metric. By default, Style and Object Pairing receive
only `room_type`; Scale, Function, and Placement receive no generation prompt.
The complete generation instruction is excluded unless a versioned experiment
explicitly selects `original_prompt` for a metric. Room type is disambiguating
context rather than a required answer: it must not impose stereotypical
contents or a stereotypical style. Authorized deviations and an explicit
visual-style specification remain separate structured inputs.

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

For collision, support, OOB, scale, style, and object pairing, allocate each
event to its scoring targets and compute:

\[
B_m=\sum_o\min\left(1,\sum_{e\rightarrow o}P_o(e)\right).
\]

Functional and Placement expose several typed checks for the same object. They
therefore use the strongest confirmed burden for each object instead of
stacking check count:

\[
B_m=\sum_o\max_{e\rightarrow o}P_o(e),
\qquad m\in\{\mathrm{Functional},\mathrm{Placement}\}.
\]

The raw event sum remains in the audit ledger, while the effective burden uses
the formula above. Thus two independent `0.4` checks on one object contribute
`0.4`, and a later `1.0` finding replaces rather than adds to an earlier
`0.4`. This preserves check-level evidence without rewarding a workflow for
inventing more obligations. `1.0` burden means one fully invalid
object-equivalent.

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
- Function and Placement claims are typed and judged independently. A Function
  verdict, shared object identity, similar wording, or role overlap never
  suppresses a Placement check by itself.
- After the static Placement claim has independently been found invalid, its
  duplicate burden may be excluded only when the Placement row cites one
  supplied final Function `event_id` as `function_event_ref` and explicitly
  confirms `same_physical_event=true`. This exact-event exclusion changes no
  Function verdict and is recorded in both ledgers. Independent Placement
  failures on the same object remain scoreable.
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
score sensitivity. Every audit report MUST therefore expose the nominal
weight `W_m`, the base local factor `W_m n_m`, and the multiplier-adjusted
local factor.

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

First compute the base metric deduction:

\[
D_{\mathrm{base},m}=\max\left(
D_{\mathrm{prevalence},m},D_{\mathrm{floor},m}\right).
\]

The versioned `deduction_multiplier` parameter defaults to `2.0`. It applies
to Collision, Support, OOB, Scale, Style, and Object Pairing. Functional and
Placement retain an applied multiplier of `1.0`:

\[
\alpha_m=
\begin{cases}
\texttt{deduction\_multiplier},&m\in\{Collision,Support,OOB,Scale,Style,Pairing\},\\
1,&m\in\{Functional,Placement\}.
\end{cases}
\]

\[
D_m=\min\left(1,\alpha_m D_{\mathrm{base},m}\right),
\qquad S_m=1-D_m.
\]

The public runner parameter is `--deduction-multiplier`; setting it to `1.0`
reproduces the unscaled deduction projection without changing evidence,
verdicts, severity, ownership, or metric weights.

The two deductions are combined with `max`, not addition, so the same event is
not charged twice. A metric can lose all of its own weight but cannot consume
another metric's allocation.

Under `intrinsic_validity_v2`, one maximum-severity event has the following
minimum final-score effect from the floor:

| Metric | Minimum final-point deduction |
| --- | ---: |
| Collision | 5.0 |
| Support | 5.0 |
| OOB | 5.0 |
| Scale | 1.4 |
| Style | 2.45 |
| Object pairing | 3.15 |
| Functional | 9.1 |
| Placement | 4.9 |

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
used for evidence and explanation, not automatic penalties. Placement accepts
`valid`, `invalid`, internal `unresolved`, or `excluded_function_owned`. The
last conclusion is legal only for the exact stable-event contract above; it
requires no Placement defect and cannot be inferred from object overlap alone.
Forced invalid with no reliable severity uses `atypical=0.4`.

Placement Discovery remains a sparse routing prior. Its
`considered_object_ids` means inspected, not proven normal. The global and
group-local Judges retain a generic missed-check sweep and may register a
typed Judge-originated check from current evidence or request evidence through
the existing bounded loop. Neutral Function prerequisite context may draw
attention to clearance requirements or admitted binary relations, but carries
no verdict, severity, geometry decision, suppression instruction, or automatic
Placement check.

The current canonical profile enables the experimental
`residual_global_review` only after the typed global/group/target Placement
passes. It receives one angled global view, one top view, and one identity
view, and may emit only a novel `scene_zone` or non-operational
`contextual_anchor` finding. It may not repeat a subject already charged by a
typed Placement finding. Its compact semantic context contains the broad
`scene_type`, the complete `object_id`/category/group roster, completed typed
checks, and stable Function ownership references. `scene_type` is only an
activity-program prior: aesthetic theme, the full generation prompt,
stereotypical required contents, and prior metric verdicts are excluded.
Object presence or shared group membership never proves a relation. The
adapter verifies that the scene type and complete object roster were delivered
intact and records a delivery audit. Raw boundary, architecture, center, size,
and rotation fields may remain as supporting spatial facts, but free-text object
descriptions are removed and visual layout remains the primary evidence. The
same Judge episode must first return exactly one non-scoring global-position
observation for every group, using only its `group_id`/`object_ids`, the three
global views, and raw spatial facts. Grouping labels, reasons, and formation
edges are excluded. After every group has been inspected, the Judge synthesizes
one scene-level verdict; group observations are audit records, not votes or
independent penalty events. A malformed or missing observation row lowers that
observation coverage and marks ambiguity without turning the whole metric into
an infrastructure failure.

The residual review is a separate Placement
component rather than an additional full-weight penalty path:

\[
S_{\mathrm{placement}}
=0.80S_{\mathrm{typed}}+0.20S_{\mathrm{residual\ global}}.
\]

Each component uses the same canonical burden projection before mixing. Thus
residual scene-level synthesis can consume at most 20% of the Placement
metric. The report must expose both component scores, weights, events,
evidence, and the residual Judge episode. Disabling this policy restores the
single typed Placement score without changing the public evaluator interface.

## 7. Terminal-state and partial-coverage rules

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

Validation remains strict and fail-closed at each contract boundary, but a
recoverable malformed auxiliary result MUST have the smallest possible blast
radius. Each logical VLM call receives at most one same-evidence schema-repair
retry. After that retry:

- discovery keeps valid object/relation rows and defaults or drops only the
  malformed atomic rows;
- a recoverable planner/discovery failure disables only its specialized probes
  while retained global/group evidence and Judge episodes continue;
- a recoverable Judge-row failure keeps legal rows and defaults only the
  affected episode to `valid` with `evidence_ambiguous=true`;
- Placement permits only two deterministic transport repairs:
  `placement_check_type -> check_type`, and stable generation/rekeying of a
  missing or duplicate `proposal_id`. No semantic field may be invented.

Transport/API/auth/rate-limit/timeout failure, irreparable canonical input,
missing or undecodable evidence, unexpected program failure, or failure of all
grouping backends remains an infrastructure failure. These failures MUST NOT
be converted into a scientific verdict.

If an auxiliary camera selector fails after a decodable packet has already
passed EvidenceGate, the existing packet remains scientifically usable. The
Controller may make one final forced-binary Judge call on that retained packet
and MUST record the selector failure as a degraded terminal choice. If no
validated packet exists, or if the final Judge transport call fails, the scope
is an infrastructure failure rather than a scientific result. A final schema
shape failure after the single repair retry follows the item-level policy
above when legal evidence and rows remain.

Every required global, cross-group relation, group-local, and typed-check
scope must therefore terminate as `evaluated`, `evaluated_degraded`, or
`infrastructure_failure`. A final scientific `unresolved` state is a control
contract violation, not a third verdict.

Function and Placement freeze a score-grounding coverage fraction
`c = grounded_obligations / eligible_obligations` and an earned score mass over
the full eligible denominator. A metric, layer, case, or model score is
publishable only when its total `c >= 0.80`. The published percentage is
`100 * earned_score_mass / c`, equivalently the score observed on grounded
obligations. When `c < 0.80`, the public score is `null` and the scope status is
`failed_coverage_threshold`; the observed score mass and coverage remain in the
audit record. Missing obligations are excluded from both earned points and the
normalization denominator, so they are neither zero nor full credit. Observed
burden is never extrapolated to ungrounded obligations and missing obligations
are never defaulted valid.

Reports publish the frozen eligible/grounded counts, defaulted units,
`grounded_score_weight`, and `grounded_score_fraction`. Layer and final score
outputs are normalized over grounded score weight only after passing the 80%
publication threshold and MUST always be accompanied by their coverage. UIs
MUST show the published percentage and coverage, or an explicit coverage-
threshold failure; they MUST NOT hide missing coverage or silently treat it as
pass/fail.

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
2. no-L2 runs select `intrinsic_validity_v2` without silent renormalization;
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
