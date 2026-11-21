"""Run with: pytest test_RubiksCube.py"""
import numpy as np
import pytest

from src.RubiksCube import RubiksCube, perms, solved, action_names

CENTERS = [4, 13, 22, 31, 40, 49]
IDX = {n: i for i, n in enumerate(action_names)}


def inverse(a):
    return (a + 6) % 12


def apply(RubiksCube, actions):
    for a in actions:
        RubiksCube = RubiksCube.step(a)
    return RubiksCube


def test_solved_state_layout():
    assert solved.shape == (54,)
    assert np.all(np.bincount(solved) == 9)
    assert RubiksCube().is_solved()


def test_perms_shape_and_validity():
    assert perms.shape == (12, 54)
    for p in perms:
        assert sorted(p) == list(range(54))  # each is a true permutation


@pytest.mark.parametrize("a", range(12))
def test_each_move_touches_20_stickers(a):
    assert np.sum(perms[a] != np.arange(54)) == 20


@pytest.mark.parametrize("a", range(12))
def test_move_four_times_is_identity(a):
    assert apply(RubiksCube(), [a] * 4).is_solved()


@pytest.mark.parametrize("a", range(12))
def test_move_not_identity_before_four(a):
    for k in (1, 2, 3):
        assert not apply(RubiksCube(), [a] * k).is_solved()


@pytest.mark.parametrize("a", range(6))
def test_move_then_inverse_is_identity(a):
    assert apply(RubiksCube(), [a, inverse(a)]).is_solved()
    assert apply(RubiksCube(), [inverse(a), a]).is_solved()


@pytest.mark.parametrize("a", range(12))
def test_centers_never_move(a):
    c = RubiksCube().step(a)
    assert np.all(c.state[CENTERS] == np.arange(6))


def test_sexy_move_has_order_six():
    sexy = [IDX[m] for m in ("R", "U", "R'", "U'")]
    c = RubiksCube()
    for i in range(1, 7):
        c = apply(c, sexy)
        assert c.is_solved() == (i == 6)


def test_sune_has_order_six():
    sune = [IDX[m] for m in ("R", "U", "R'", "U", "R", "U", "U", "R'")]
    c = RubiksCube()
    for i in range(1, 7):
        c = apply(c, sune)
        assert c.is_solved() == (i == 6)


def test_u_clockwise_geometry():
    # U clockwise: F top row -> L, L -> B, B -> R, R -> F. Colours: U0 R1 F2 D3 L4 B5
    s = RubiksCube().step(IDX["U"]).state
    assert np.all(s[36:39] == 2)  # L top row now has F's colour
    assert np.all(s[45:48] == 4)  # B top row now has L's colour
    assert np.all(s[9:12] == 5)  # R top row now has B's colour
    assert np.all(s[18:21] == 1)  # F top row now has R's colour
    assert np.all(s[0:9] == 0)  # U face colour unchanged (uniform)


def test_r_clockwise_geometry():
    # R clockwise: F right col -> U, U -> B (reversed), B -> D, D -> F
    s = RubiksCube().step(IDX["R"]).state
    assert np.all(s[[2, 5, 8]] == 2)  # U right col gets F's colour
    assert np.all(s[[45 + 6, 45 + 3, 45 + 0]] == 0)  # B gets U's colour on its (mirrored) left col
    assert np.all(s[[27 + 2, 27 + 5, 27 + 8]] == 5)  # D right col gets B's colour
    assert np.all(s[[18 + 2, 18 + 5, 18 + 8]] == 3)  # F right col gets D's colour


def test_face_turn_cycles_own_stickers():
    # Mark each sticker on U with a unique value and check the in-face rotation.
    base = RubiksCube(np.arange(54, dtype=np.int8))
    s = base.step(IDX["U"]).state
    # clockwise: new[2] = old[0], new[8] = old[2], new[6] = old[8], new[0] = old[6]
    assert (s[2], s[8], s[6], s[0]) == (0, 2, 8, 6)
    assert (s[5], s[7], s[3], s[1]) == (1, 5, 7, 3)
    assert s[4] == 4


def test_step_does_not_mutate():
    c = RubiksCube()
    c.step(0)
    assert c.is_solved()


def test_commutation_of_opposite_faces():
    # U and D turn independent layers, so they commute
    for a, b in ((IDX["U"], IDX["D"]), (IDX["R"], IDX["L"]), (IDX["F"], IDX["B"])):
        assert np.array_equal(apply(RubiksCube(), [a, b]).state, apply(RubiksCube(), [b, a]).state)


def test_adjacent_faces_do_not_commute():
    assert not np.array_equal(apply(RubiksCube(), [IDX["U"], IDX["R"]]).state,
                              apply(RubiksCube(), [IDX["R"], IDX["U"]]).state)


@pytest.mark.parametrize("seed", range(5))
def test_random_scramble_invariants_and_undo(seed):
    rng = np.random.default_rng(seed)
    moves = rng.integers(0, 12, 100)
    c = RubiksCube()
    for a in moves:
        c = c.step(a)
        assert np.all(np.bincount(c.state, minlength=6) == 9)
        assert np.all(c.state[CENTERS] == np.arange(6))
    for a in moves[::-1]:
        c = c.step(inverse(a))
    assert c.is_solved()


def test_one_hot():
    oh = RubiksCube().one_hot()
    assert oh.shape == (324,)
    assert oh.sum() == 54
    assert oh.dtype == np.int8
