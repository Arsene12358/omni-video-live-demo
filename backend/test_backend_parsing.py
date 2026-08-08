# SPDX-License-Identifier: Apache-2.0
"""Pure-parsing tests for demo_backend's server-log telemetry. No server needed.

Samples are byte-exact server-log lines from the migration validation evidence
(~/Vault/streaming-v026-migration/validation, S2/S5 runs of 2026-08-05), plus the
FROZEN rebase contract:

  "[streaming-kv] rebase req=%s delta=%d new_base=%d recent_tokens=%d"

Run: pytest backend/test_backend_parsing.py
"""
from demo_backend import _EV, _REBASE, _SESS, parse_kv_tail

# --- byte-exact evidence lines ------------------------------------------------
# S5 demo run (validation/s5/s5-summary.md): two consecutive eviction lines.
EVICTION_S5_A = (
    "(StageEngineCoreProc_stage0_replica0 pid=694365) INFO 08-05 21:21:28 "
    "[single_type_kv_cache_manager.py:1163] [streaming-kv] eviction "
    "req=vsess-58e8b2cd-0-9fb85dfd computed=10790 total_blocks=675 alive=673 "
    "(start_pinned=160) freed=2"
)
EVICTION_S5_B = (
    "(StageEngineCoreProc_stage0_replica0 pid=694365) INFO 08-05 21:21:28 "
    "[single_type_kv_cache_manager.py:1163] [streaming-kv] eviction "
    "req=vsess-58e8b2cd-0-9fb85dfd computed=11033 total_blocks=690 alive=673 "
    "(start_pinned=160) freed=15"
)
# S5 demo run after one refresh: request id stepped to epoch 1.
EVICTION_S5_EPOCH1 = (
    "(StageEngineCoreProc_stage0_replica0 pid=694365) INFO 08-05 21:21:50 "
    "[single_type_kv_cache_manager.py:1163] [streaming-kv] eviction "
    "req=vsess-ec3fd6e3-1-afd6ecc2 computed=10949 total_blocks=685 alive=673 "
    "(start_pinned=160) freed=12"
)
# S2 omni stream: in-omni per-stage-suffix ids vsess-<8hex>-<epoch>-<8hex>.
EVICTION_S2 = (
    "(StageEngineCoreProc_stage0_replica0 pid=409684) INFO 08-05 17:38:54 "
    "[v1/core/single_type_kv_cache_manager.py:1163] [streaming-kv] eviction "
    "req=vsess-13307d90-0-963c7dab computed=20938 total_blocks=1309 alive=673 "
    "(start_pinned=160) freed=15"
)
S2_EPOCH1_STATS = (
    "(APIServer pid=340339) INFO 08-05 12:42:59 [stats.py:791] "
    "[RequestE2EStats [request_id=vsess-426813f1-1-a995d2f7]]"
)
# FROZEN rebase contract rendered with a realistic engine-rebase request id
# (epoch pinned at 0; the id never steps) inside the usual server-log framing.
REBASE_LINE = (
    "(StageEngineCoreProc_stage0_replica0 pid=694365) INFO 08-05 21:30:00 "
    "[streaming_rebase.py:378] [streaming-kv] rebase req=vsess-58e8b2cd-0-9fb85dfd "
    "delta=45056 new_base=4096 recent_tokens=8192"
)


# --- (a) rebase line: fields extracted, count increments ----------------------
def test_rebase_line_fields():
    m = _REBASE.search(REBASE_LINE)
    assert m is not None
    assert m.group(1) == "vsess-58e8b2cd-0-9fb85dfd"  # req
    assert m.group(2) == "45056"  # delta
    assert m.group(3) == "4096"  # new_base
    assert m.group(4) == "8192"  # recent_tokens


def test_rebase_count_increments():
    assert parse_kv_tail("")[3] == 0
    assert parse_kv_tail(REBASE_LINE)[3] == 1
    assert parse_kv_tail(REBASE_LINE + "\n" + REBASE_LINE)[3] == 2


# --- (b) eviction line: frozen contract still parsed --------------------------
def test_eviction_line_byte_exact():
    comp, alive, epoch, rebases = parse_kv_tail(EVICTION_S5_A)
    assert (comp, alive) == (10790, 673)
    assert epoch == 0  # from req=vsess-58e8b2cd-0-... (suffix form)
    assert rebases == 0
    m = _EV.search(EVICTION_S5_A)
    assert m.groups() == ("10790", "675", "673")


def test_eviction_last_line_wins():
    comp, alive, _, _ = parse_kv_tail(EVICTION_S5_A + "\n" + EVICTION_S5_B)
    assert (comp, alive) == (11033, 673)


# --- (c) vsess epoch path: bare and per-stage-suffix forms --------------------
def test_vsess_epoch_bare_form():
    assert _SESS.findall("S2D_LAST_REQ=vsess-13307d90-0") == ["0"]
    assert parse_kv_tail("req=vsess-a7af1f23-9 finished")[2] == 9


def test_vsess_epoch_stage_suffix_form():
    # real S2 trace shape: vsess-<8hex>-<epoch>-<8hex>
    assert _SESS.findall("vsess-13307d90-0-963c7dab") == ["0"]
    assert _SESS.findall("vsess-426813f1-1-a995d2f7") == ["1"]
    assert parse_kv_tail(EVICTION_S2)[2] == 0
    assert parse_kv_tail(S2_EPOCH1_STATS)[2] == 1


def test_vsess_epoch_max_wins():
    tail = "\n".join([EVICTION_S2, S2_EPOCH1_STATS, EVICTION_S5_A])
    assert parse_kv_tail(tail)[2] == 1


# --- (d) mixed log: eviction + rebase + vsess combined ------------------------
def test_mixed_engine_rebase_mode():
    # engine-rebase mode: epoch stays 0, rebase lines are the boundedness signal
    tail = "\n".join([EVICTION_S5_A, REBASE_LINE, EVICTION_S5_B, REBASE_LINE])
    comp, alive, epoch, rebases = parse_kv_tail(tail)
    assert (comp, alive) == (11033, 673)  # newest eviction wins
    assert epoch == 0  # id never steps in engine-rebase mode
    assert rebases == 2


def test_mixed_refresh_mode_unaffected():
    # refresh mode: no rebase lines; epoch steps via new vsess ids as before
    tail = "\n".join([EVICTION_S5_A, EVICTION_S5_B, EVICTION_S5_EPOCH1])
    comp, alive, epoch, rebases = parse_kv_tail(tail)
    assert (comp, alive) == (10949, 673)
    assert epoch == 1
    assert rebases == 0


def test_empty_tail():
    assert parse_kv_tail("") == (None, None, None, 0)
