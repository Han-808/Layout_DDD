#!/bin/zsh
# Re-roll one Open-space +10 gap campaign until its briefs land, or until a bound
# is hit.  The frozen core deliberately refuses semantic retries
# (frozen_two_stage/retry_policy.py: "a fresh-case retry campaign remains an outer
# workflow and must not be smuggled into this one-shot policy"), so the outer loop
# and all of its bounds live here.
#
# This script generates nothing itself: every round is one ordinary invocation of
# run_open_space_plus10.sh into a fresh round directory.  It never edits config --
# narrow a campaign's `ordered_brief_ids` once, as a reviewed commit, before using
# this.  That keeps the loop away from the trust-pinned registries entirely, so it
# can run concurrently with other campaigns.
#
#   export API3_API_KEY=...            # or API2_APP_CREDENTIAL, per family
#   Support/bash/local/run_gap_until_success.sh api3-opus5-plus10-gap-v1
#
# Env: MAX_ROUNDS (12), ROUND_DELAY (300), MAX_PREFLIGHT_BACKOFFS (4),
#      PREFLIGHT_BACKOFF (420), CONTINUE_PAST_GATE (unset).
#
# Exit: 0 success  1 exhausted  2 refused before spending  3 no free round dir
#       4 credit exhausted  5 permanent preflight failure  6 circuit breaker
#       7 review gate  8 rate-limit backoff exhausted  9 local abort  143 signal

set -uo pipefail
emulate -L zsh
setopt null_glob

SCRIPT_PATH="${${(%):-%x}:A}"
REPO_ROOT="${SCRIPT_PATH:h:h:h:h}"
if [[ "$REPO_ROOT" == */.claude/worktrees/* ]]; then
  PRIMARY_ROOT="${REPO_ROOT%%/.claude/worktrees/*}"
else
  PRIMARY_ROOT="$REPO_ROOT"
fi
LAUNCHER="$REPO_ROOT/Support/bash/local/run_open_space_plus10.sh"
REGISTRY="$REPO_ROOT/configs/generation/campaign_v2/campaigns_v2.json"
RUNS="$PRIMARY_ROOT/Support/artifacts/outputs/e2e_scenegen_repro/runs"

(( $# == 1 )) || { print -u2 -- "usage: ${SCRIPT_PATH:t} <campaign_id>"; exit 2 }
CAMPAIGN="$1"

# Family and the credential it needs, and the glob that spans every campaign id
# this model has ever generated under -- coverage is the union across all of them,
# so a brief banked by an earlier campaign must not be re-rolled.
case "$CAMPAIGN" in
  api3-opus5-*)    FAMILY=api3; CRED=API3_API_KEY;        SIBLINGS='api3-opus5-plus10-*' ;;
  api3-sonnet5-*)  FAMILY=api3; CRED=API3_API_KEY;        SIBLINGS='api3-sonnet5-plus10-*' ;;
  api3-fable5-*)   FAMILY=api3; CRED=API3_API_KEY;        SIBLINGS='api3-fable5-plus10-*' ;;
  api2-kimi-k3-*)  FAMILY=api2; CRED=API2_APP_CREDENTIAL; SIBLINGS='api2-kimi-k3-plus10-*' ;;
  api2-glm53-*)    FAMILY=api2; CRED=API2_APP_CREDENTIAL; SIBLINGS='api2-glm53-plus10-*' ;;
  api2-gpt56sol-*) FAMILY=api2; CRED=API2_APP_CREDENTIAL; SIBLINGS='api2-gpt56sol-plus10-*' ;;
  api2-gpt6-astra-*) FAMILY=api2; CRED=API2_APP_CREDENTIAL; SIBLINGS='api2-gpt6-astra-*' ;;
  tokenhub-*)      FAMILY=tokenhub; CRED=TOKENHUB_API_KEY; SIBLINGS='tokenhub-hy4-preview-plus10-*' ;;
  *) print -u2 -- "unknown campaign family: $CAMPAIGN"; exit 2 ;;
esac

MAX_ROUNDS="${MAX_ROUNDS:-12}"
ROUND_DELAY="${ROUND_DELAY:-300}"
MAX_PREFLIGHT_BACKOFFS="${MAX_PREFLIGHT_BACKOFFS:-4}"
PREFLIGHT_BACKOFF="${PREFLIGHT_BACKOFF:-420}"

STATE="$RUNS/_gap_until_success/$CAMPAIGN"
mkdir -p "$STATE"

# ---- refusals, before anything is spent -------------------------------------
[[ -x "$LAUNCHER" ]] || { print -u2 -- "launcher is not executable: $LAUNCHER"; exit 2 }
[[ -r "$REGISTRY" ]] || { print -u2 -- "campaign registry is unreadable: $REGISTRY"; exit 2 }
if [[ -z "${(P)CRED:-}" ]]; then
  # The launcher would fall back to a hidden interactive prompt and the loop would
  # hang on it forever, so require the export up front.
  print -u2 -- "$CRED is not exported; the loop must not inherit an interactive prompt"
  exit 2
fi
if [[ -f "$STATE/run.pid" ]] && kill -0 "$(<"$STATE/run.pid")" 2>/dev/null; then
  print -u2 -- "another loop for $CAMPAIGN is live (pid $(<"$STATE/run.pid"))"
  exit 2
fi
print $$ > "$STATE/run.pid"

CHILD_PID=""
cleanup() {
  [[ -n "$CHILD_PID" ]] && kill -TERM "$CHILD_PID" 2>/dev/null
  rm -f "$STATE/run.pid"
}
on_signal() { print -- "\ninterrupted; stopping after the current round"; cleanup; exit 143 }
trap on_signal INT TERM HUP
trap cleanup EXIT

# ---- helpers ----------------------------------------------------------------
requested_briefs() {
  jq -r --arg c "$CAMPAIGN" \
    '.campaigns[] | select(.campaign_id==$c) | .ordered_brief_ids[]' "$REGISTRY"
}

# Briefs this campaign asks for that are not yet complete anywhere, for any
# campaign of the same model.  `eligible_for_strict_one_shot_evaluation` is what
# the merge and the publication path require, so a bare `complete` is not enough.
remaining_briefs() {
  local -a want banked
  want=(${(f)"$(requested_briefs)"})
  banked=()
  local f
  for f in "$RUNS"/open_space_plus10_r*/${~SIBLINGS}/brief_*/case.result.json; do
    banked+=(${(f)"$(jq -r 'select(.status=="complete" and .eligible_for_strict_one_shot_evaluation==true) | .brief_id' "$f" 2>/dev/null)"})
  done
  local b
  for b in $want; do
    (( ${banked[(Ie)$b]} )) || print -- "$b"
  done
}

# Generating rounds already spent, read off disk rather than from a counter, so a
# restart after a credit top-up cannot silently re-authorise the whole budget.
# A preflight-aborted round leaves no campaign directory and so costs nothing.
rounds_consumed() {
  local -a dirs
  dirs=("$RUNS"/open_space_plus10_r*/"$CAMPAIGN"(N/))
  print -- ${#dirs}
}

# The launcher's guard only tests $OUTPUT_BASE/<campaign_id>, but sharing a base
# with another campaign would also share _launcher_logs and make the preflight
# records ambiguous, so claim a wholly unused round directory.
free_base() {
  local n
  for n in {1..200}; do
    [[ -e "$RUNS/open_space_plus10_r$n" ]] && continue
    print -- "$RUNS/open_space_plus10_r$n"
    return 0
  done
  return 1
}

# Every non-zero status is flattened to failure_category "transport_or_http" by
# campaign/execution.py, so only .report.http_status discriminates a permanent 401
# from a transient 429.  Read the exact record the launcher printed, filtered to
# this campaign and to mode "run" -- a glob for the newest file would pick up a
# concurrent campaign's record from a shared base.
preflight_status() {
  local round_log="$1" record
  record=$(sed -n 's/^preflight record: //p' "$round_log" | tail -1)
  [[ -n "$record" && -r "$record" ]] || { print -- "none"; return }
  jq -r --arg c "$CAMPAIGN" \
    'select(.campaign_id==$c and .mode=="run") | .report.http_status // "null"' \
    "$record" 2>/dev/null | tail -1 | read -r status
  print -- "${status:-none}"
}

# A case reached a content verdict only if the model actually answered and the
# core judged the payload.  The frozen core emits exactly six statuses; these are
# the three that mean "the model produced something we could judge".
content_verdicts() {
  local base="$1" n=0 f
  for f in "$base/$CAMPAIGN"/brief_*/case.result.json; do
    jq -e '.status=="complete" or .status=="stage_a_schema_invalid" or .status=="placement_schema_invalid"' \
      "$f" >/dev/null 2>&1 && (( n += 1 ))
  done
  print -- $n
}

# Credit exhaustion is the one failure that must never be retried: 400 is absent
# from every retryable status set, the campaign keeps going and keeps spending,
# and the fix is a human one.
credit_exhausted() {
  local base="$1"
  [[ -d "$base/$CAMPAIGN" ]] || return 1
  grep -rlq 'credit balance is too low' "$base/$CAMPAIGN" 2>/dev/null
}

# Stop and surface a brief whose only content verdicts are the relation-endpoint
# contract error and which has never produced an accepted object plan: that is the
# signature of a wall rather than a near miss.  Armed only when it is the last
# brief standing, so the gate can never abandon unspent work on other briefs.
walled_brief() {
  local brief="$1" verdicts=0 relation=0 accepted=0 d
  for d in "$RUNS"/open_space_plus10_r*/${~SIBLINGS}/$brief(N/); do
    [[ -f "$d/object_plan.json" ]] && (( accepted += 1 ))
    [[ -f "$d/case.result.json" ]] || continue
    jq -e '.status=="stage_a_schema_invalid" or .status=="placement_schema_invalid" or .status=="complete"' \
      "$d/case.result.json" >/dev/null 2>&1 || continue
    (( verdicts += 1 ))
    jq -e '(.reason // "") | test("relation endpoints must reference public object slots")' \
      "$d/case.result.json" >/dev/null 2>&1 && (( relation += 1 ))
  done
  (( accepted == 0 && verdicts >= 3 && relation == verdicts ))
}

# ---- loop -------------------------------------------------------------------
STOP=exhausted
backoffs=0
barren=0
round=0
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

print -- "gap loop: campaign=$CAMPAIGN family=$FAMILY"
print -- "  requested briefs : $(requested_briefs | tr '\n' ' ')"
print -- "  rounds already spent on this campaign: $(rounds_consumed) (cap $MAX_ROUNDS)"
print -- "  state: $STATE"

while true; do
  typeset -a rem
  rem=(${(f)"$(remaining_briefs)"})
  if (( ${#rem} == 0 )); then STOP=success; break; fi

  consumed=$(rounds_consumed)
  if (( consumed >= MAX_ROUNDS )); then STOP=exhausted; break; fi

  if (( ${#rem} == 1 )) && [[ -z "${CONTINUE_PAST_GATE:-}" ]] && walled_brief "${rem[1]}"; then
    print -- "review gate: ${rem[1]} has only ever failed the relation-endpoint contract"
    print -- "  and has never produced an accepted object plan; set CONTINUE_PAST_GATE=1 to override"
    STOP=review_gate
    break
  fi

  # Delay before a round rather than after one, so landing the last brief does not
  # cost a gratuitous wait.
  if (( round > 0 )); then
    print -- "  sleeping ${ROUND_DELAY}s"
    sleep "$ROUND_DELAY"
  fi
  (( round += 1 ))

  base=$(free_base) || { STOP=no_free_base; break }
  round_log="$STATE/round.$(date -u +%Y%m%dT%H%M%SZ).log"
  print -- "round $round (consumed $consumed/$MAX_ROUNDS): remaining ${rem[*]} -> ${base:t}"

  "$LAUNCHER" "$FAMILY" --campaign "$CAMPAIGN" --output-base "$base" >"$round_log" 2>&1 &
  CHILD_PID=$!
  wait "$CHILD_PID" || true
  CHILD_PID=""

  # Classify on artifacts, never on the exit status: the launcher exits 2 from nine
  # unrelated places, several in under a second with no network, so keying on $? is
  # an unthrottled spin.
  if credit_exhausted "$base"; then
    print -u2 -- "  upstream reports the credit balance is too low; this never succeeds on retry"
    STOP=credit
    break
  fi

  status=$(preflight_status "$round_log")
  case "$status" in
    200)
      : ;;
    400|401|403)
      print -u2 -- "  preflight http=$status is permanent (credential or entitlement)"
      STOP=preflight_permanent
      break ;;
    none)
      # The launcher never reached its preflight stage, so nothing was sent and
      # nothing was spent.  This is always local -- interpreter, bindings, the
      # config gate, an occupied output base -- and retrying cannot fix it.
      print -u2 -- "  launcher aborted before preflight; see $round_log"
      tail -n 3 "$round_log" >&2
      STOP=local_abort
      break ;;
    *)
      (( backoffs += 1 ))
      print -- "  preflight http=$status is transient (backoff $backoffs/$MAX_PREFLIGHT_BACKOFFS); round not counted"
      if (( backoffs > MAX_PREFLIGHT_BACKOFFS )); then STOP=backoff_exhausted; break; fi
      sleep "$PREFLIGHT_BACKOFF"
      continue ;;
  esac
  backoffs=0

  verdicts=$(content_verdicts "$base")
  if (( verdicts == 0 )); then
    (( barren += 1 ))
    print -- "  no brief reached a content verdict (barren $barren/2)"
    if (( barren >= 2 )); then STOP=circuit_breaker; break; fi
  else
    barren=0
    print -- "  $verdicts brief(s) reached a content verdict"
  fi
done

# ---- terminal summary -------------------------------------------------------
typeset -a final
final=(${(f)"$(remaining_briefs)"})
typeset -A CODES
CODES=(success 0 exhausted 1 no_free_base 3 credit 4 preflight_permanent 5
       circuit_breaker 6 review_gate 7 backoff_exhausted 8 local_abort 9)
code=${CODES[$STOP]:-1}

jq -n --arg campaign "$CAMPAIGN" --arg stop "$STOP" --arg started "$started_at" \
      --arg finished "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --argjson code "$code" \
      --argjson rounds_this_loop "$round" --argjson rounds_total "$(rounds_consumed)" \
      --arg remaining "${final[*]}" \
  '{schema_version:"gap_until_success_summary_v1", campaign_id:$campaign,
    stop_reason:$stop, exit_code:$code, started_at:$started, finished_at:$finished,
    rounds_this_loop:$rounds_this_loop, rounds_total:$rounds_total,
    remaining_briefs:($remaining | split(" ") | map(select(length>0)))}' \
  > "$STATE/summary.json"

print -- ""
print -- "stop: $STOP (exit $code)"
print -- "rounds this loop: $round   total for this campaign: $(rounds_consumed)"
if (( ${#final} == 0 )); then
  print -- "all requested briefs are banked"
else
  print -- "still missing: ${final[*]}"
fi
print -- "summary: $STATE/summary.json"
exit "$code"
