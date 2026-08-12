# SPDX-License-Identifier: Apache-2.0
"""Pure tests for demo_backend: server-log telemetry + session-config passthrough.
No server needed.

Samples are byte-exact server-log lines from the migration validation evidence
(~/Vault/streaming-v026-migration/validation, S2/S5 runs of 2026-08-05) and from
the unbounded-streaming V3 wave (~/Vault/streaming-unbounded/validation/v3), so
they carry the FROZEN rebase contract as the engine actually emits it:

  "[streaming-kv] rebase req=%s delta=%d new_base=%d recent_tokens=%d"

Run: pytest backend/test_backend_parsing.py
"""
from demo_backend import _EV, _REBASE, _SESS, build_session_config, parse_kv_tail

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
# Real engine-rebase line from the V3 multi-stream wave
# (~/Vault/streaming-unbounded/validation/v3/real1/child-62252736-job393802516.txt:1957),
# minus the srun task prefix that run captured it under — the demo reads the
# server's own stdout log, which carries no such prefix. Epoch pinned at 0: in
# engine-rebase mode the request id never steps.
REBASE_LINE = (
    "(StageEngineCoreProc_stage0_replica0 pid=597052) INFO 08-11 19:21:59 "
    "[streaming_rebase.py:452] [streaming-kv] rebase req=vsess-ad5f9994-0-8362cd2d "
    "delta=5500 new_base=640 recent_tokens=2046"
)


# --- (a) rebase line: fields extracted, count increments ----------------------
def test_rebase_line_fields():
    m = _REBASE.search(REBASE_LINE)
    assert m is not None
    assert m.group(1) == "vsess-ad5f9994-0-8362cd2d"  # req
    assert m.group(2) == "5500"  # delta
    assert m.group(3) == "640"  # new_base (tiny-R wave geometry: start_size 640)
    assert m.group(4) == "2046"  # recent_tokens


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


# --- (e) session.config passthrough: the engine-rebase checkbox ---------------
_START = {"type": "start", "clip": "c.mp4", "sampling_fps": 2, "sink_frames": 6,
          "num_frames": 10, "refresh_at": 2000, "evs": False}


def test_refresh_mode_config_unchanged():
    """Checkbox off (or absent): the shipped driver-side refresh payload."""
    for msg in (_START, {**_START, "engine_rebase": False}):
        cfg = build_session_config(msg)
        assert cfg["persistent"] is True
        assert cfg["engine_rebase"] is False
        assert cfg["refresh_at_position"] == 2000
        assert cfg["sink_frames"] == 6 and cfg["num_frames"] == 10
        assert cfg["enable_frame_filter"] is False


def test_engine_rebase_mode_omits_refresh_at_position():
    """Checkbox on: the engine owns boundedness, so refresh_at_position is not
    just unused but omitted — the server warns when it is set alongside
    engine_rebase, and a warning in the demo log reads like a misconfiguration."""
    cfg = build_session_config({**_START, "engine_rebase": True})
    assert cfg["engine_rebase"] is True
    assert "refresh_at_position" not in cfg
    # Everything else is the same session as refresh mode.
    assert cfg["persistent"] is True
    assert cfg["sink_frames"] == 6 and cfg["num_frames"] == 10


def test_ui_knobs_are_forwarded_in_both_modes():
    for rebase in (False, True):
        cfg = build_session_config(
            {**_START, "engine_rebase": rebase, "sink_frames": 0, "num_frames": 4, "evs": True}
        )
        assert cfg["sink_frames"] == 0
        assert cfg["num_frames"] == 4
        assert cfg["enable_frame_filter"] is True


def test_config_defaults_when_the_browser_omits_knobs():
    cfg = build_session_config({"type": "start", "clip": "c.mp4"})
    assert (cfg["sink_frames"], cfg["num_frames"]) == (6, 10)
    assert cfg["refresh_at_position"] == 2000
    assert cfg["engine_rebase"] is False
