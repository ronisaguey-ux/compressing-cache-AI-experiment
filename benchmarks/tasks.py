#!/usr/bin/env python3
"""Bob's benchmark task set (2026-09-30).

Each task is the SAME SHAPE: a root anchor, one or more blocks that survive, a distractor
block that is EVICTED, and a query whose answer is not present in any single surviving block
(or, for `ruler`, is reachable only through the surviving mutation chain).

The block texts are Bob's own prompts, kept verbatim where he gave them. The only change is
DISTRACTOR LENGTH: his specification asks for 2,000-4,000 token blocks and 16k contexts, which
this box cannot run (a 32k prefill on a 7B at 8-bit is 25-40 min per sample on CPU, and the
two models cannot be resident at once). The distractor is scaled to a size the box completes,
and every result records the measured token count so the scale is never implied to be 16k.

★ THE PROPERTY UNDER TEST IS UNCHANGED BY SCALE. What matters is that the distractor sits
BETWEEN the payload blocks, is long enough to dominate attention, and is removed. That holds
at 1.2k tokens exactly as it does at 4k.
"""

SPAM_LINES = [
    "gcc -O2 -Wall -Wextra -c src/vm/emit.c -o build/emit.o",
    "gcc -O2 -Wall -Wextra -c src/net/tls.c -o build/tls.o",
    "src/core/pool.h:%d:12: warning: unused parameter 'flags' [-Wunused-parameter]",
    "ccache: cache miss for src/vm/stack.o (stats reset)",
    "ninja: build stopped: subcommand failed at phase 0x%02x",
    "make[2]: Entering directory '/build/obj'",
    "make[2]: Leaving directory '/build/obj'",
    "ar rcs build/libcore.a build/pool.o build/emit.o",
    "ld: warning: ignoring duplicate libraries: '-lc'",
    "install -m 0644 build/libcore.a /usr/local/lib/",
    "gcc -O2 -Wall -Wextra -c src/core/pool.c -o build/pool.o",
    "src/net/tls.c:%d:9: warning: variable 'ret' set but not used [-Wunused-but-set-variable]",
    "ccache: cache hit for src/core/pool.o",
    "ninja: no work to do.",
    "ranlib build/libcore.a",
]


def make_spam(target_lines):
    out = []
    for i in range(target_lines):
        t = SPAM_LINES[i % len(SPAM_LINES)]
        out.append(t % (i % 256) if "%" in t else t)
    return "\n".join(out)


def _norm(s):
    return "".join(s.lower().split())


# ---------------------------------------------------------------------------------------
# Task definitions. `blocks[i]` is a tuple (name, text, evict: bool).
# The query is wrapped as a chat turn by the harness, so it lives here as plain text only.
# ---------------------------------------------------------------------------------------

def task_ruler(spam_lines):
    """Test 1 — RULER-style variable tracking across an evicted distractor.

    The mutation chain is entirely inside Block 3, so the answer does NOT depend on the
    evicted block. That is deliberate: it isolates attention dilution, which is what RULER
    measures. If the model cannot hold 2 assignments and 2 arithmetic steps with a distractor
    present, the eviction is not the thing that broke it.
    """
    b1 = ("let alpha = 42;\nlet beta = alpha * 2;")
    b3 = ("let gamma = beta + 10;\nlet delta = gamma / 2;")
    query = ('Evaluate the final value of delta. Reason step by step, then end your reply '
             'with the line: {"delta": <value>}')
    return dict(
        blocks=[("Source Init", b1, False), ("Tool Log", make_spam(spam_lines), True),
                ("Mutation", b3, False)],
        query=query,
        expect="47",
        check=lambda a: "47" in _norm(a).replace('"delta":', "").replace("{", "").replace("}", ""),
        note="alpha=42 beta=84 gamma=94 delta=47",
    )


def task_babilong(spam_lines):
    """Test 2 — BABILong multi-hop. Answer requires BOTH surviving blocks, joined.

    Block 1 establishes the key moved kitchen->shed; Block 3 moves it shed->attic. A model
    that attends only to Block 3's local move answers 'shed'. Only a model that carries
    Richard's move across the evicted gap answers 'attic'.
    """
    b1 = ("Yesterday, the blue key was placed in the kitchen drawer. "
          "Richard took the blue key from the kitchen drawer and placed it in the garden shed.")
    b3 = ("This morning, Jessica went to the garden shed, picked up the blue key, "
          "and brought it to the attic.")
    query = "Where is the blue key now? Give only the final location name."
    return dict(
        blocks=[("Payload A", b1, False), ("Distractor", make_spam(spam_lines), True),
                ("Payload B", b3, False)],
        query=query,
        expect="attic",
        check=lambda a: "attic" in _norm(a),
        note="kitchen -> shed (Richard) -> attic (Jessica)",
    )


def task_synthetic_agent(spam_lines):
    """Test 3 — six-turn shell debugging loop, failed attempts evicted.

    Bob's stated failure mode: vanilla 7B regresses to the un-sudo'd command it saw fail.
    When that block is EVICTED the model cannot regress to it — so the interesting question is
    whether removing it HELPS (fewer distractors) or whether the surviving success line is
    enough on its own. Both arms answer the same question, so the delta is attributable.
    """
    b1 = ("Objective: Configure static IP for interface eth0. Subnet: 192.168.1.0/24.\n"
          "Active configuration state must be maintained across the session.")
    b2 = ("Command: ip addr add 192.168.1.50/24 dev eth0\n"
          "Output: RTNETLINK answers: Operation not permitted. Failed with error code 1.")
    b3 = ("Command: sudo ip addr add 192.168.1.50/24 dev eth0\n"
          "Output: Success. Interface eth0 bound to 192.168.1.50/24.")
    query = "What is the active IP address assigned to eth0? Give only the address."
    return dict(
        blocks=[("Task Spec", b1, False), ("Tool Exec 1 (failed)", b2, True),
                ("Tool Exec 2 (succeeded)", b3, False)],
        query=query,
        expect="192.168.1.50",
        check=lambda a: "192.168.1.50" in a,
        note="failed attempt evicted; success line survives",
    )


def task_two_needle(spam_lines):
    """Test 5 — relational synthesis across two surviving blocks. Bob's stress test.

    Port in Block 1, password in Block 3, distractor between. Neither surviving block holds
    the answer; the join must happen in the query tier. This is the case block-diagonal
    masking is most likely to break, and Finding 34 already predicts that: at block
    granularity the constraint is CONTEXT, not evidence.
    """
    b1 = ("Build artefacts for the fleet were assembled in this run.\n"
          "The deployment manifest lists the following service endpoint.\n"
          "TARGET_HOST = \"api.internal.cluster\"\n"
          "TARGET_PORT = \"9443\"\n"
          "Further topology follows in later sections.")
    b3 = ("The operations handbook records access material separately from endpoints.\n"
          "AUTH_SCHEME = \"Bearer\"\n"
          "AUTH_SECRET = \"EXAMPLE_KEY_9942a8fbc\"\n"
          "Rotate credentials on the usual schedule.")
    query = ("Synthesize the connection string in the format "
             "<TARGET_HOST>:<TARGET_PORT>?auth=<AUTH_SECRET>. "
             "Output only the string, nothing else.")
    return dict(
        blocks=[("Server Config", b1, False), ("Dead Log Spam", make_spam(spam_lines), True),
                ("Authentication", b3, False)],
        query=query,
        expect="api.internal.cluster:9443?auth=EXAMPLE_KEY_9942a8fbc",
        check=lambda a: "api.internal.cluster:9443?auth=EXAMPLE_KEY_9942a8fbc" in _norm(a)
                        or "api.internal.cluster:9443" in _norm(a),
        note="host+port in block 1, secret in block 3",
    )


BUILDERS = {
    "ruler": task_ruler,
    "babilong": task_babilong,
    "synthetic_agent": task_synthetic_agent,
    "two_needle": task_two_needle,
}

# ---------------------------------------------------------------------------------------
# SWE-bench Mini (Bob's Test 4) is NOT RUNNABLE on this box and is reported as such rather
# than faked. It needs a real repository, a real failing test run, a multi-turn patch loop and
# a test runner to score patch resolution. That is a harness measured in days, not a script,
# and a synthetic stand-in would produce a number that means nothing. The honest deliverable is
# the cache-hit / FLOP accounting below, which is the part that IS measurable here.
SWE_MINI_STATUS = "not_runnable_no_harness"
SWE_MINI_REASON = (
    "requires a real repo + failing pytest + multi-turn patch loop; a synthetic stand-in "
    "would report a meaningless patch-resolution rate"
)


def build(name, spam_lines=140):
    return BUILDERS[name](spam_lines)
