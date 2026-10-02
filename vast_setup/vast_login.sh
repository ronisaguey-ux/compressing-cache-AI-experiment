#!/usr/bin/env bash
# Exchange the API key + a 2FA code for a session_key. Renting requires this; a plain key does not.
#
#   ./vast_login.sh totp  <6-digit-code>     # from your authenticator app (valid ~30s)
#   ./vast_login.sh email <code> <secret>    # if Vast emails you a code
#
# Writes the session_key to $VAST_HOME/session_key (600).
set -eu
D="${VAST_HOME:-$HOME/.vast}"
KEY=$(tr -d '[:space:]' < "$D/api_key")
METHOD="${1:-totp}"; CODE="${2:-}"; SECRET="${3:-}"
[ -n "$CODE" ] || { echo "usage: $0 totp <code>   |   $0 email <code> <secret>"; exit 2; }

if [ "$METHOD" = "email" ]; then
  BODY=$(python3 -c 'import json,sys;print(json.dumps({"tfa_method":"email","code":sys.argv[1],"secret":sys.argv[2]}))' "$CODE" "$SECRET")
else
  BODY=$(python3 -c 'import json,sys;print(json.dumps({"tfa_method":"totp","code":sys.argv[1]}))' "$CODE")
fi

R=$(curl -s --max-time 30 -X POST "https://console.vast.ai/api/v0/tfa/" \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "$BODY")
SK=$(printf '%s' "$R" | python3 -c '
import sys,json
try: print(json.load(sys.stdin).get("session_key","") or "")
except Exception: print("")')
[ -n "$SK" ] || { echo "NO session_key. Body was:"; printf '%s\n' "$R" | head -c 300; exit 1; }
printf '%s' "$SK" > "$D/session_key"; chmod 600 "$D/session_key"
echo "session_key stored (len ${#SK})"

# verify it actually authenticates -- a key that searches but cannot list is not a session key
for ep in /api/v0/users/current/ /api/v0/instances/; do
  C=$(curl -s --max-time 20 -o /tmp/vs.txt -w '%{http_code}' -H "Authorization: Bearer $SK" "https://console.vast.ai$ep")
  printf '  %-24s HTTP=%s  %s\n' "$ep" "$C" "$(head -c 100 /tmp/vs.txt | tr -d '\n')"
  sleep 2
done
