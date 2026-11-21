"""Run with: pytest"""
import numpy as np

from src.RubiksCube import RubiksCube
from src.CubeLog import MoveLogger, read_log


def test_logger_roundtrip(tmp_path):
    p = tmp_path / "run.jsonl"
    with MoveLogger(p) as log:
        log.reset([1, 2, 3])
        log.step(4, -0.01)
        log.step(5)
        log.reset([])
        log.step(0, 1.0)
    eps = read_log(p)
    assert [e["id"] for e in eps] == [0, 1]
    assert eps[0]["scramble"] == [1, 2, 3]
    assert eps[0]["steps"] == [(4, -0.01), (5, None)]
    assert eps[1]["steps"] == [(0, 1.0)]


def test_numpy_ints_are_serialisable(tmp_path):
    p = tmp_path / "np.jsonl"
    with MoveLogger(p) as log:
        log.reset(np.array([3, 4]))
        log.step(np.int64(7), np.float32(0.5))
    assert read_log(p)[0]["steps"] == [(7, 0.5)]


def test_step_without_reset_starts_episode(tmp_path):
    p = tmp_path / "x.jsonl"
    with MoveLogger(p) as log:
        log.step(2)
    assert read_log(p)[0]["steps"] == [(2, None)]


def test_reader_skips_partial_last_line(tmp_path):
    p = tmp_path / "partial.jsonl"
    with MoveLogger(p) as log:
        log.reset([])
        log.step(1)
    with open(p, "a") as f:
        f.write('{"type": "step", "acti')
    assert len(read_log(p)[0]["steps"]) == 1


def test_lines_are_flushed_immediately(tmp_path):
    p = tmp_path / "live.jsonl"
    log = MoveLogger(p)
    log.reset([])
    log.step(3)
    assert len(p.read_text().splitlines()) == 2   # readable before close()
    log.close()


def test_replaying_log_reproduces_env_state(tmp_path):
    rng = np.random.default_rng(0)
    scramble = rng.integers(0, 12, 10)
    actions = rng.integers(0, 12, 25)
    p = tmp_path / "r.jsonl"
    with MoveLogger(p) as log:
        log.reset(scramble)
        for a in actions:
            log.step(a)
    ep = read_log(p)[0]
    c, c2 = RubiksCube(), RubiksCube()
    for a in scramble: c = c.step(a)
    for a in actions: c = c.step(a)
    for a in ep["scramble"]: c2 = c2.step(a)
    for a, _ in ep["steps"]: c2 = c2.step(a)
    assert np.array_equal(c.state, c2.state)