#!/usr/bin/env python3
"""Find an offer AND create the instance in ONE shot.

★ WHY ONE SHOT: offers EXPIRE in minutes. A two-step "pick a machine, then create it" loses the
offer and returns `no_such_ask (404/3603)` -- which reads like a broken request and is really a
stale offer id.

Also registers an SSH key BEFORE creating (a key added afterwards does not get injected), and
writes the new instance id to $VAST_HOME/last_instance.

  ./vast_rent.py --gpu "RTX A6000" --max-price 0.60
  ./vast_rent.py --gpu "RTX 4090" --vrgb 44 --ramgb 48 --disk 80
"""
import argparse, json, os, subprocess, sys, time, urllib.request

D = os.environ.get("VAST_HOME", os.path.expanduser("~/.vast"))
API = "https://console.vast.ai/api/v0"


def key():
    # prefer the 2FA session_key; a plain api_key cannot rent
    for name in ("session_key", "api_key"):
        p = os.path.join(D, name)
        if os.path.exists(p):
            v = open(p).read().strip()
            if v:
                return v
    sys.exit("no key at %s/session_key -- run vast_login.sh first" % D)


def call(method, path, body=None, tries=3):
    for i in range(tries):
        req = urllib.request.Request(
            API + path, method=method,
            headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"},
            data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            txt = e.read().decode()[:300]
            # ★ 429 is a RATE LIMIT, not "no results". Sleep and retry rather than concluding
            # there is no stock -- a wrong conclusion here wastes a whole session.
            if e.code == 429 and i < tries - 1:
                print("  429 rate limited, sleeping 12s"); time.sleep(12); continue
            sys.exit("HTTP %s on %s %s: %s" % (e.code, method, path, txt))
    sys.exit("gave up on %s %s" % (method, path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTX A6000")
    ap.add_argument("--max-price", type=float, default=0.60)
    ap.add_argument("--vrgb", type=float, default=44.0)
    ap.add_argument("--ramgb", type=float, default=48.0)
    ap.add_argument("--disk", type=float, default=80.0)
    ap.add_argument("--rel", type=float, default=0.985)
    ap.add_argument("--image", default="vastai/pytorch:cuda-12.4.1-auto")
    ap.add_argument("--label", default="agent-run")
    ap.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/id_ed25519.pub"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # register the ssh key FIRST -- a key added after creation is not injected into that box
    if os.path.exists(a.ssh_key):
        pub = open(a.ssh_key).read().strip()
        call("POST", "/ssh/", {"ssh_key": pub})
        print("ssh key registered")

    # ★ THE FILTER NEEDS A SPACE IN THE GPU NAME: "RTX A6000" works, "RTX_A6000" silently
    # returns ZERO offers -- a wrong name looks exactly like no stock.
    q = {"gpu_name": {"eq": a.gpu}, "num_gpus": {"eq": 1},
         "dph_total": {"lte": a.max_price},
         "gpu_total_ram": {"gte": int(a.vrgb * 1024)},
         "cpu_ram": {"gte": int(a.ramgb * 1024)},
         "disk_space": {"gte": a.disk},
         "reliability2": {"gte": a.rel}}
    import urllib.parse
    offers = call("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)))
    offs = offers.get("offers", offers if isinstance(offers, list) else [])
    if not offs:
        sys.exit("no offers matched. Loosen --max-price / --rel, or try another --gpu.")
    offs.sort(key=lambda o: o.get("dph_total", 9e9))
    o = offs[0]
    print("picked offer %s  %s  $%.3f/hr  %.0fGB vram  %.0fGB ram  %.0fGB disk  rel=%.4f"
          % (o.get("id"), a.gpu, o.get("dph_total", 0), o.get("gpu_total_ram", 0) / 1024,
             o.get("cpu_ram", 0) / 1024, o.get("disk_space", 0), o.get("reliability2", 0)))
    if a.dry_run:
        return
    r = call("PUT", "/asks/%s/" % o["id"], {
        "client_id": "me", "image": a.image, "disk": a.disk, "label": a.label,
        "runtype": "ssh", "onstart": "true"})
    iid = r.get("new_contract")
    if not iid:
        sys.exit("create failed: %s" % json.dumps(r)[:300])
    os.makedirs(D, exist_ok=True)
    open(os.path.join(D, "last_instance"), "w").write(str(iid))
    print("\nCREATED instance %s -> %s/last_instance" % (iid, D))
    print("wait ~60-180s, then:  python3 vast_status.py %s   (prints the ssh command)" % iid)


if __name__ == "__main__":
    main()
