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
    # ★ EXCLUDE A HOST. MEASURED: picking "the cheapest offer" repeatedly returns the SAME offer
    # id, so every instance lands on the SAME host -- and that host answered
    # `Permission denied (publickey)` to a key that was registered on the account and had worked on
    # an older instance. Two boxes, both on host 519195, both denied. Re-renting must be able to
    # avoid a host rather than rolling the dice on the same one again.
    ap.add_argument("--exclude-host", default="", help="comma-separated host_ids to skip")
    ap.add_argument("--offer-index", type=int, default=0, help="pick the Nth cheapest offer")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # register the ssh key FIRST -- a key added after creation is not injected into that box
    # ★ A DUPLICATE KEY IS SUCCESS, NOT FAILURE. The account keeps a single key list, so the
    # second and every later rental returns HTTP 400 {"error":"duplicate"} -- and `call()` exits
    # on any HTTPError, so a normal repeat rental aborted before it ever searched for a box.
    # MEASURED: this blocked a re-rent outright. Tolerate `duplicate`; fail on anything else.
    if os.path.exists(a.ssh_key):
        pub = open(a.ssh_key).read().strip()
        try:
            call("POST", "/ssh/", {"ssh_key": pub})
            print("ssh key registered")
        except SystemExit as e:
            if "duplicate" in str(e):
                print("ssh key already registered (fine)")
            else:
                raise

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
    _skip = {h.strip() for h in a.exclude_host.split(",") if h.strip()}
    if _skip:
        before = len(offs)
        offs = [o for o in offs if str(o.get("host_id")) not in _skip]
        print("excluded host(s) %s: %d -> %d offers" % (",".join(sorted(_skip)), before, len(offs)))
    if not offs:
        sys.exit("no offers left after exclusions -- loosen --max-price/--rel or drop --exclude-host")
    if a.offer_index:
        if a.offer_index >= len(offs):
            sys.exit("--offer-index %d out of range (%d offers)" % (a.offer_index, len(offs)))
        offs = offs[a.offer_index:]
    o = offs[0]
    print("host_id=%s" % o.get("host_id"))
    print("picked offer %s  %s  $%.3f/hr  %.0fGB vram  %.0fGB ram  %.0fGB disk  rel=%.4f"
          % (o.get("id"), a.gpu, o.get("dph_total", 0), o.get("gpu_total_ram", 0) / 1024,
             o.get("cpu_ram", 0) / 1024, o.get("disk_space", 0), o.get("reliability2", 0)))
    if a.dry_run:
        return
    # ★★ FALL THROUGH STALE OFFERS INSTEAD OF DYING ON THE FIRST ONE.
    # MEASURED: the same offer id came back on four consecutive searches and every create answered
    # `no_such_ask` -- the bundles list is served from a cache, so the "cheapest" entry can be a
    # machine that has been rented out. Retrying the search returns the same dead id, so the only
    # way forward is to try the NEXT offer from the list already in hand. That keeps the one-shot
    # property (no re-search, so the list cannot go stale again) while tolerating dead entries.
    _made = None
    _last = ""
    for _i, _o in enumerate(offs[:20]):
        try:
            r = call("PUT", "/asks/%s/" % _o["id"], {
                "client_id": "me", "image": a.image, "disk": a.disk, "label": a.label,
                "runtype": "ssh", "onstart": "true",
                # ★ THE KEY MUST BE PASSED AT CREATE TIME. MEASURED: an instance created without it,
                # then "associated" afterwards via POST /instances/{id}/ssh/ (which answers "already
                # associated with instance"), STILL rejects the key -- sshd answers
                # `Permission denied (publickey)` on both the proxy host and the direct port.
                "ssh_key": (open(a.ssh_key).read().strip() if os.path.exists(a.ssh_key) else None),
            })
        except SystemExit as e:
            _last = str(e)
            if "no_such_ask" in _last or "not available" in _last:
                print("  offer %s is gone (host %s), trying the next"
                      % (_o["id"], _o.get("host_id")))
                continue
            raise
        _made = (r.get("new_contract"), _o)
        break
    if not _made:
        sys.exit("every offer in the list failed to create; last error: %s" % _last[:200])
    iid, o = _made
    if not iid:
        sys.exit("create failed: %s" % json.dumps(r)[:300])
    os.makedirs(D, exist_ok=True)
    open(os.path.join(D, "last_instance"), "w").write(str(iid))
    print("\nCREATED instance %s -> %s/last_instance" % (iid, D))
    print("wait ~60-180s, then:  python3 vast_status.py %s   (prints the ssh command)" % iid)


if __name__ == "__main__":
    main()
