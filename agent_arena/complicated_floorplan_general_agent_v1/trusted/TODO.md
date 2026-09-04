# Trusted-side launch TODO

The fixed suite, public Agent workspace, database capability service, outer
Seatbelt executor, and scoped model-gateway primitive are implemented.

Before any paid or official generation:

- [ ] Select the exact Agent entrants; no implementation family is preselected.
- [ ] Pin each Agent executable version and content hash.
- [ ] Record each underlying model/service identity and compute policy exactly
      when observable; otherwise disclose the field as opaque or unavailable.
- [ ] Register and verify one thin adapter per selected Agent runtime.
- [ ] Set a common model-request budget in addition to the fixed tool budget.
- [ ] Register the exact context/output limits, requested reasoning setting,
      temperature or explicit provider default, maximum model turns, and
      wall-clock limit.
- [ ] Freeze one infrastructure retry/timeout policy; ambiguous in-flight
      requests must not be duplicated, and official episodes must not use
      best-of-run selection.
- [ ] Run one isolated, no-evaluator preflight per Agent.
- [ ] Confirm no credential, host-home, repo, other episode, or arbitrary
      network access from inside the Agent process.
- [ ] Freeze the entrant profile hashes.
- [ ] Persist the harness-produced launch record, trusted host-side submission
      seal, and trusted hash-chained tool transcript; never record credentials,
      headers, endpoints, or hidden reasoning content.
- [ ] Obtain explicit launch approval.

Do not weaken the network policy or inject host credentials to bypass these
gates.
