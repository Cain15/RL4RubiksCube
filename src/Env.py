import numpy as np

from src.RubiksCube import RubiksCube, get_inverse_actions

rng = np.random.default_rng(42)


def random_cubes(n, K):
    """
    n solvable states, random-walk depth uniform in 0...K.
    """
    cubes = []
    for i in range(n):
        cube = RubiksCube()
        depth = rng.integers(0, K + 1)
        cube, _ = cube.scramble(int(depth))
        cubes.append(cube)
    return cubes


inverse_actions = get_inverse_actions()


def demo_transitions(n, K):
    """
    Scramble n cubes for K moves, return the reversed path (s, a, s', upper_bound)
    """
    S = []
    A = []
    S2 = []
    UB = []
    for i in range(n):
        cube = RubiksCube()
        _, scramble = cube.scramble(K, rng)
        for step, a in enumerate(scramble):
            new_cube = cube.step(a)
            upper_bound = step + 1
            S.append(new_cube.one_hot())
            A.append(inverse_actions[a])
            S2.append(cube.one_hot())
            UB.append(upper_bound)
            cube = new_cube
    return S, A, S2, UB
