Nonrect wall tolerance and materialization error classification
=============================================================

Materialization revision v3 and read-only inspector v2 fix two defects without
changing geometry, input layouts, metric implementations, or the 1e-5 m absolute
coordinate tolerance:

* Vertex multisets now require a one-to-one matching within tolerance instead of
  equality of rounded coordinate buckets. This preserves multiplicity and accepts
  small floating-point errors at bucket boundaries. Nonfinite or malformed points,
  missing vertices, and true out-of-tolerance displacements fail closed.
* Backend contract errors retain their type instead of being wrapped as retryable
  infrastructure errors. True explicit infrastructure failures still use the
  existing retry policy.

Inspector reports record the matching revision and tolerance. The new
materialization revision participates in future plan and campaign identities.
Existing v2 artifacts remain independently hash-verifiable; this does not authorize
blindly rewriting their manifests or bypassing strict campaign resume. Adopting
this version in a paused v2 campaign requires a separate audited identity upgrade.

Regression validation covers float32 wall heights, positive/negative bucket
boundaries, exact tolerance, out-of-tolerance shifts, duplicates, permutations,
ambiguous matching, nonfinite points, and nonretryable contract failures while
other rooms continue. The existing infrastructure retry test remains unchanged.
