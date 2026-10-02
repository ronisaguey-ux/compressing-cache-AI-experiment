#!/usr/bin/env bash
for pat in "runfix.sh" "vast_entry.py" "incremental_coding"; do
  for p in $(ps -eo pid,cmd | grep "$pat" | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
done
sleep 2
echo "remaining: $(ps -eo cmd | grep -c '[v]ast_entry.py')"
