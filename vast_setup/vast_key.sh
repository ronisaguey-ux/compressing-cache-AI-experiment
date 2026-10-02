#!/usr/bin/env bash
# Store a Vast.ai API key. The key is a BARE 64-char string -- no "VAR=", no newline.
# Get it from: console.vast.ai -> Account -> API Keys.
#   ./vast_key.sh <the-key>
set -eu
[ $# -ge 1 ] || { echo "usage: $0 <api-key>"; exit 2; }
D="${VAST_HOME:-$HOME/.vast}"
mkdir -p "$D"; chmod 700 "$D"
printf '%s' "$1" > "$D/api_key"
chmod 600 "$D/api_key"
echo "stored $D/api_key ($(wc -c < "$D/api_key") bytes)"
echo "NOTE: a plain API key can SEARCH but not RENT. Renting needs a 2FA session_key -- see vast_login.sh"
