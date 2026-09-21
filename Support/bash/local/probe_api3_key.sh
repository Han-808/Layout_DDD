#!/bin/zsh

set -euo pipefail
set +x

# Minimal API3 credential probe.  Answers exactly one question — does the key
# in this terminal authenticate against the API3 LiteLLM gateway — using the
# cheapest authenticated endpoint (`/v1/models`).  No model is invoked, nothing
# is generated, and the key is read hidden and unset on exit.
#
#   Support/bash/local/probe_api3_key.sh
#
# Output is limited to HTTP status, whether the three benchmark aliases are
# listed, and the response's error `type`/`message` on failure.  Reasoning and
# body content are never printed in full.

BASE_URL="${API3_BASE_URL:-http://21.214.33.175:4000}"
BASE_URL="${BASE_URL%/}"

cleanup() { unset API3_API_KEY; }
trap cleanup EXIT INT TERM

if [[ -z "${API3_API_KEY:-}" ]]; then
  read -r -s "API3_API_KEY?API3 key (hidden): "
  print
fi
[[ -n "$API3_API_KEY" ]] || { print -u2 -- "API3 key input was empty"; exit 2; }
export API3_API_KEY

# Shape hints that cost nothing and leak nothing.
KEY_LENGTH=${#API3_API_KEY}
KEY_PREFIX="${API3_API_KEY[1,3]}"
[[ "$KEY_PREFIX" == "sk-" ]] && PREFIX_NOTE="starts with sk- (LiteLLM virtual/master key shape)" || PREFIX_NOTE="does not start with sk- (not the usual LiteLLM key shape)"
if [[ "$API3_API_KEY" == *[[:space:]]* ]]; then
  print -u2 -- "Key contains whitespace; likely a paste error"
fi
print -r -- "key length: $KEY_LENGTH; $PREFIX_NOTE"

print -r -- "GET $BASE_URL/v1/models"
RESPONSE_FILE=$(mktemp "${TMPDIR:-/tmp}/api3-probe.XXXXXX")
trap 'cleanup; command rm -f -- "$RESPONSE_FILE"' EXIT
HTTP_STATUS=$(curl --noproxy '*' -sS --max-time 20 -o "$RESPONSE_FILE" -w '%{http_code}' \
  -H "Authorization: Bearer $API3_API_KEY" \
  -H "Accept: application/json" \
  "$BASE_URL/v1/models" || print -r -- "000")
print -r -- "http: $HTTP_STATUS"

if [[ "$HTTP_STATUS" == "200" ]]; then
  for alias in claude-sonnet-5-aihub claude-opus-5-aihub claude-fable-5-aihub; do
    if jq -e --arg a "$alias" '.data[]? | select(.id == $a)' "$RESPONSE_FILE" >/dev/null 2>&1; then
      print -r -- "  listed : $alias"
    else
      print -r -- "  MISSING: $alias"
    fi
  done
  print -r -- "  total aliases: $(jq -r '.data | length' "$RESPONSE_FILE" 2>/dev/null || print '?')"
  exit 0
fi

# Failure: show only the structured error type/message, truncated.
jq -r '"  error.type   : \(.error.type // "-")\n  error.message: \((.error.message // "-") | tostring | .[0:160])"' "$RESPONSE_FILE" 2>/dev/null \
  || print -r -- "  (non-JSON response, $(wc -c < "$RESPONSE_FILE" | tr -d ' ') bytes)"
case "$HTTP_STATUS" in
  401) print -r -- "verdict: gateway rejected this key (invalid, expired or revoked). Ask the API3 admin for a fresh virtual key." ;;
  000) print -r -- "verdict: no HTTP response (network/VPN/route). Not a key problem." ;;
  *)   print -r -- "verdict: authenticated request reached the gateway but failed with $HTTP_STATUS." ;;
esac
exit 2
