Non-rectangular evaluator v1
============================

This evaluator is an additive workflow selected only with
``evaluation_mode="non_rectangular_multi_room"``. Omitting
``evaluation_mode`` (or passing ``None``) preserves the canonical rectangular
dispatch and arguments exactly. Unknown modes fail before canonical or legacy
dispatch.

Frozen scope
------------

* Scene → room → object hierarchy.
* One authoritative evaluation unit per room, in layout order.
* Global coordinates and object IDs are preserved; rooms are not recentered.
* Each room evaluation sees only that room's objects, floor polygon, and walls.
* Point clouds, doors, windows, ceiling support, and cross-room judging are out
  of scope.
* Functional precedes Semantic Placement. There is no separate scene-allocation
  metric.

Geometry and evidence acquisition
---------------------------------

``PolygonRoomGeometry`` is the shared deterministic substrate. It owns exact
polygon containment, camera-to-target line of sight, bounded ray placement
with wall clearance, representative interior anchors, wall-local frames, and
object-footprint helpers.

Existing metric-specific candidate semantics remain unchanged. A polygon gate
runs before preview rendering and rejects or repairs AABB-valid poses that are
outside a concavity, too close to a wall, wall-occluded, or unable to frame the
target. Because the benchmark has no ceiling, a bounded high-elevation repair
is allowed while camera XY remains inside the polygon. Active-camera actions
are revalidated by the same gate.

OOB local acquisition uses an explicit bounded ladder only after the nominal
edge-local bank is empty: 8 cm wall clearance with the original focus, then
4 cm with a 2 cm inward focus inset, 2 cm with a 5 cm inset and 45/60/72 degree
high-angle views, and finally zero wall clearance with at most a 10 cm inward
focus inset plus a room-internal local orthographic top view. The 10 cm value is
an evidence-acquisition bound only. It never expands the room polygon, changes
the measured crossing, exempts an OOB event, or changes Collision/Support
thresholds.

The global packet remains one top view plus one selected perspective. The top
view fits the complete polygon envelope. The perspective targets a
deterministic representative interior point and uses a polygon-feasible camera
location.

Metric behavior
---------------

Collision
~~~~~~~~~

The canonical object-object detector, event semantics, ranking, visual roles,
and Judge path are reused. Only camera feasibility and wall LOS use the polygon
gate.

OOB
~~~

The rectangular six-plane detector is not called. Horizontal OOB is the area
of the projected oriented bounding-box footprint outside the exact room
polygon. One object remains one scoring event even if it crosses multiple
edges. Each event carries edge-scoped wall IDs, intersection focus, inward
normal, tangent, and penetration measurements. Its local camera bank is
generated in each violated edge's local frame. Floor tolerance is preserved;
ceiling is excluded.

Support
~~~~~~~

The canonical downward floor/object contact detector and grounded-contact
graph are reused. Cardinal wall clearances are replaced by exact
footprint-to-wall-segment measurements. Wall attachment evidence carries the
actual wall ID and local frame. Ceiling candidates and ceiling wording are
absent in the selected mode.

L3
~~

The existing Scale, Style, Object Pairing, Functional, and Semantic Placement
workflow is reused. The room type is the only program context exposed to
Functional and Placement. Generator task-slot intent and object-plan text are
removed before detectors, grouping, camera selection, or Judge calls.

Failure behavior
----------------

* Count or program-mapping terminal gates stop before room evaluation.
* A bounded local-camera search that produces no view is normal evidence
  exhaustion, not an evaluator infrastructure failure.
* After local exhaustion, P0b uses retained room-global evidence plus exact
  object/polygon/wall geometry for a mandatory binary Judge call. The same
  path supports a geometry-only Judge call when zero images survive.
* The nonrect forced-binary P0b call receives one bounded same-evidence schema
  repair. If the repaired response is still invalid, it remains an explicit
  Judge response-contract failure and is never converted into a score.
* If no visual or Judge conclusion is available but the validated geometry is
  sufficient, Collision/OOB/Support use an audited metric-specific deterministic
  binary rule. Scene-quality scopes use the existing conservative default-valid
  terminal rule and retain explicit ambiguity/degradation provenance.
* L1 deterministic continuity is applied per event. A hard failure on one
  event does not suppress safe closure of neighboring unresolved events, but a
  metric score is withheld until every required event has a binary result.
* Missing evidence is never represented as an ordinary visual judgement. Every
  forced/defaulted result records its decision source, visual count, matched
  rule, geometry version, and degraded state.
* API/transport/authentication errors, malformed contracts, invalid polygon or
  object geometry, corrupt artifacts, hash drift, and ambiguous partial writes
  remain real failures and are not converted to scores.
* Sanitized per-metric diagnostics are written before room normalization. They
  retain metric ID, source status, event keys, routes, evidence counts, typed
  error class, and fallback rejection reason without prompt or response bodies.
* A scene score is available only when every required room report is complete.

Scoring
-------

The public route uses ``non_rectangular_room_weighted_v1``. L1 and L3 carry
0.30 and 0.70 respectively. Collision, OOB, and Support are equal within L1.
Within L3 the weights are Scale 0.04, Style 0.07, Object Pairing 0.09,
Functional 0.52, and Semantic Placement 0.28. Functional and Placement room
scores are aggregated by planned object instances; the other metrics use their
evaluated object denominator. Count and program-mapping factors apply only to
Functional. Reports remain non-official and non-publishable until the benchmark
release freezes its publication policy.

Verification invariants
-----------------------

* Rectangular camera, OOB, and Support outputs are byte-identical to the branch
  baseline for inputs without explicit non-rectangular metadata.
* Tests cover a concave L-room, concavity-void cameras, arbitrary-wall OOB,
  no-ceiling Support, room projection, public routing, and a no-API full mock.
* Thirty-eight completed generated scenes from four model families (184 rooms
  and 1,484 objects) are used as an integration audit for room projections,
  polygon detectors, Support evidence, and camera banks. Across those rooms,
  all 198 routed OOB events produced a bounded local candidate: 117 nominal,
  6 R1, 40 R2, and 35 R3, with no algorithm exception or empty final bank.
  The same audit found no algorithm exception or empty bank across 195
  Collision and 127 Support routed events; all 184 rooms also produced both
  polygon-global poses.
* The previously failing Opus ``scene_011760`` queen-bed event now renders an
  R3 room-internal orthographic raw/highlight packet; its filing-cabinet event
  resolves at R1. Their exact sub-millimetre crossings remain unchanged.
