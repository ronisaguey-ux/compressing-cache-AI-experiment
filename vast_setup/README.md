# Renting and driving a Vast.ai GPU

Everything here is what I actually used, including the traps that cost me time. Copy this folder,
run four commands, and you have a working GPU box.

## For the human (one time)

1. **Get an API key** — `console.vast.ai` → Account → API Keys → create.
2. **Turn on 2FA if it is not already** — Vast requires a 2FA-verified session to **create or
   manage** instances. A plain API key can *search* but returns
   `401 ... requires you to have logged in using 2FA` for everything else.
   - **A 401 on Vast is not a bad key.** Read the BODY — it names the real cause.
3. **Give the agent the key** and, if TOTP is enabled, be ready to read out a 6-digit code when it
   asks (the code is valid ~30 seconds).

## For the agent

```bash
export VAST_HOME=~/.vast                 # wherever you like
./vast_key.sh <the-api-key>              # stores it 600
./vast_login.sh totp <6-digit-code>      # human reads this from their authenticator
```

`vast_login.sh` exchanges the key + code for a **session_key** and verifies it against
`/users/current/` and `/instances/`. Renting uses the session_key, not the API key.

Then find and create a box in **one shot**:

```bash
./vast_rent.py --gpu "RTX A6000" --max-price 0.60
# -> CREATED instance 12345678
python3 vast_status.py 12345678          # prints the ssh -p ... root@... command
```

SSH in and run:

```bash
ssh -p <port> root@<host> -i ~/.ssh/id_ed25519
```

## Traps — each of these cost real time

| symptom | what is actually happening |
|---|---|
| `no_such_ask (404/3603)` on create | **Offers expire in minutes.** Fetch and create in one shot — never "pick a machine, then create it later". |
| `429 Too Many Requests` | Rate limit, not "no results". Sleep ~12s and retry. |
| 0 offers for a GPU you know exists | **`gpu_name` needs a SPACE**: `"RTX A6000"` works, `"RTX_A6000"` silently returns nothing. |
| `401` on `/instances/` | The account needs 2FA. Read the message body, it says so. |
| ssh `Connection refused` | **Not necessarily dead.** Judge by the API's `actual_status` + `gpu_util` — I saw ssh refuse for 5 minutes while GPU sat at 98%. |
| Two boxes billing at once | **Check the instance list, not your memory.** I "stopped" a box that kept running at 0% GPU for hours. `actual_status` is the oracle. |
| `resources_unavailable` on restart | The host slot is taken. It is queued, not done. |

## Choosing a card

- **A plain API key cannot rent** — see above.
- **`dph_total` is $/hr**; `gpu_total_ram` and `cpu_ram` are in **MB**; `disk_space` is GB;
  `reliability2` is 0–1.
- **Rank by `GB/s per $`, not by `DLPerf/$`.** For memory-bandwidth-bound work (any transformer
  inference) `DLPerf` points at the wrong card.
- **System RAM is a separate gate from VRAM.** A 51.6 GB checkpoint must be *readable* before it is
  quantised, so a 24 GB card with only 32 GB of host RAM is a trap.
- **Disk must exceed the download**, e.g. a 62 GB model needs >100 GB even though the weights end up
  far smaller.

## Cost discipline

- **Stop or destroy the moment a run ends.** A 48 GB card idles at ~$0.3–0.5/hr.
- **Destroy, do not stop**, when the disk is not worth keeping — a stopped box still bills for its
  allocation on many hosts, and a destroyed one certainly does not.
- **List instances before and after every run** — that is the only place a stray shows up.

## Useful API shapes

```
GET  /api/v0/bundles/?q=<url-encoded JSON filter>   # offers (cheapest first)
PUT  /api/v0/asks/{offer_id}/                        # CREATE from an offer, in one shot
GET  /api/v1/instances/                              # your instances
DELETE /api/v0/instances/{id}/                       # destroy
PUT  /api/v0/instances/{id}/  {"state":"running"|"stopped"}
POST /api/v0/ssh/                                    # register an ssh key (do this BEFORE creating)
POST /api/v0/tfa/                                    # api key + code -> session_key
```

Header on every call: `Authorization: Bearer <session_key>`.
