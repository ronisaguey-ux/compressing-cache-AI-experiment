#!/usr/bin/env python3
"""Print an instance's state and its ssh command.

★ JUDGE A BOX BY THE API, NOT BY SSH. Measured: ssh gave "Connection refused" for ~5 minutes while
the API reported cur_state=running and gpu_util=98%. A refused ssh port is NOT a dead instance.
"""
import json, os, sys, urllib.request

D = os.environ.get("VAST_HOME", os.path.expanduser("~/.vast"))
key = next(open(os.path.join(D, n)).read().strip() for n in ("session_key", "api_key")
           if os.path.exists(os.path.join(D, n)))
iid = sys.argv[1] if len(sys.argv) > 1 else open(os.path.join(D, "last_instance")).read().strip()
req = urllib.request.Request("https://console.vast.ai/api/v1/instances/",
                             headers={"Authorization": "Bearer " + key})
with urllib.request.urlopen(req, timeout=40) as r:
    d = json.load(r)
ins = d.get("instances", d)
ins = ins if isinstance(ins, list) else [ins]
for i in ins:
    if str(i.get("id")) != str(iid):
        continue
    print("id        :", i.get("id"), "|", i.get("label"))
    print("status    :", i.get("actual_status"), "(intended %s)" % i.get("intended_status"))
    print("gpu       :", i.get("gpu_name"), "util", i.get("gpu_util"), "%")
    print("price     : $%.3f/hr" % (i.get("dph_total") or 0))
    print("uptime    : %.2f h" % ((i.get("duration") or 0) / 3600))
    print("ssh       : ssh -p %s root@%s -i <your-key>" % (i.get("ssh_port"), i.get("ssh_host")))
    sys.exit(0)
print("instance %s not found -- it may have been destroyed" % iid)
