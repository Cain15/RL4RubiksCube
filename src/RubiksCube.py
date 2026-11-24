import numpy as np

faces = "URFDLB"  # up, right, front, down, left, back
normals = {
    "U": (0, 1, 0),
    "R": (1, 0, 0),
    "F": (0, 0, 1),
    "D": (0, -1, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
}


def _all_stickers():
    """
    Generate all stickers
    """
    # Iterate over each face
    for f in faces:
        n = np.array(normals[f])
        # Iterate over all possible cubies
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    p = np.array([x, y, z])
                    if p @ n == 1:  # this cubie shows a sticker on face f
                        yield p, n


def _sticker_index(pos, normal):
    """
    Map a sticker to its index in [0,53]
    :param pos: The x,y,z position of the sticker
    :param normal: The normal of the face of the sticker
    :return: index of the sticker
    """
    x, y, z = pos
    # Get the face of the sticker
    f = faces[
        [normals[c] for c in faces].index(tuple(int(v) for v in normal))
    ]

    #       U
    #     L F R B
    #       D

    if f == "U":
        # y = 1 gives UP face
        # z = -1 gives first row: +1 to get index in [0,2]
        # x = -1 gives the first column: +1 to get index in [0,2]
        row, col = z + 1, x + 1
    elif f == "R":
        # x = 1 gives RIGHT face
        # y = 1 gives the first row: 1-y to get index in [0,2]
        # z = 1 gives the first column: 1-z to get index in [0,2]
        row, col = 1 - y, 1 - z  #
    elif f == "F":
        # z = 1 gives FRONT face
        # y = 1 gives the first row: 1-y to get index in [0,2]
        # x = -1 gives the first col: +1 to get index in [0,2]
        row, col = 1 - y, x + 1
    elif f == "D":
        # y = -1 gives DOWN face
        # z = 1 gives the first row: 1-z to get index in [0,2]
        # x = -1 gives the first col: +1 to get index in [0,2]
        row, col = 1 - z, x + 1
    elif f == "L":
        # x = -1 gives LEFT face
        # y = 1 gives the first row: 1-y to get index in [0,2]
        # z = -1 gives the first col: +1 to get index in [0,2]
        row, col = 1 - y, z + 1
    else:
        # z = -1 gives BACK face
        # y = 1 gives the first row: 1-y to get index in [0,2]
        # x = 1 gives the first col: 1-x to get index in [0,2]
        row, col = 1 - y, 1 - x

    return faces.index(f) * 9 + row * 3 + col  # Each face takes 9 places, each row takes 3 places


def _rotate_cw(v, axis):
    """
    Rotate the vector v by -90 degrees around the axis (clockwise as seen from outside)
    :param v: The vector to rotate
    :param axis: The axis to rotate around
    :return: rotated vector
    """
    return -np.cross(axis, v) + axis * (axis @ v)


def _build_perm(face):
    """
    Build a permutation vector
    :param face: The face that gets rotated
    :return: The permutation vector
    """
    axis = np.array(normals[face])
    perm = np.arange(54)
    for p, n in _all_stickers():
        if p @ axis == 1:  # sticker is in the turning layer
            old = _sticker_index(p, n)
            new = _sticker_index(_rotate_cw(p, axis), _rotate_cw(n, axis))
            perm[new] = old  # new_state[i] = old_state[perm[i]]
    return perm


_cw = np.stack([_build_perm(face) for face in faces])  # Clockwise moves
_ccw = np.argsort(_cw, axis=1)  # Counterclockwise moves
_half = np.stack([p[p] for p in _cw]) # Half turns: apply each clockwise quarter turn twice
perms = np.concatenate([_cw, _ccw, _half])  # All 18 possible moves
action_names = list(faces) + [f + "'" for f in faces] + [f + "2" for f in faces]
solved = np.repeat(np.arange(6), 9).astype(np.int8)  # Solved state


def get_inverse_actions():
    """
    Return the inverse actions
    :return: A map from action to inverse action
    """
    inverse_actions = {}

    for action, permutation in enumerate(perms):
        inverse_permutation = np.argsort(permutation)

        matches = [
            i for i, p in enumerate(perms)
            if np.array_equal(p, inverse_permutation)
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Could not find unique inverse for action {action}"
            )

        inverse_actions[action] = matches[0]

    return inverse_actions


class RubiksCube:
    def __init__(self, state=None):
        """
        Construct a RubiksCube object
        :param state: The state of the cube
        """
        self.state = solved.copy() if state is None else state

    def step(self, action):
        """
        Take the given action and return the new Cube (immutable-style state transitions)
        :param action: The action to take
        :return: A cube object with the new state
        """
        return RubiksCube(self.state[perms[action]])

    def is_solved(self):
        """
        Check if the cube is solved
        :return: True if solved, False otherwise
        """
        return np.array_equal(self.state, solved)

    def one_hot(self):
        """
        Give the one_hot representation of the cube
        :return: The one_hot representation of the cube
        """
        return np.eye(6, dtype=np.int8)[self.state].ravel()

    def copy(self):
        """
        Create a copy of the cube
        :return: Object with the same state
        """
        return RubiksCube(self.state)

    def scramble(self, n: int = 0, rng=None):
        """
        Scramble the cube randomly with n random steps
        :param n: The amount of steps to scramble
        :param rng: The random number generator
        :return: A tuple of the new object with a scrambled state and the scramble steps
        """
        if rng is None:
            rng = np.random.default_rng()

        scramble = []
        previous_action = None

        inverse_actions = get_inverse_actions()

        for _ in range(n):
            possible_actions = list(range(18))

            # Do not allow the inverse action of the previous action
            if previous_action is not None:
                inverse_action = inverse_actions[previous_action]
                possible_actions.remove(inverse_action)

            action = int(rng.choice(possible_actions))
            scramble.append(action)
            previous_action = action

        scramble = np.array(scramble)

        c = self.copy()

        for a in scramble:
            c = c.step(int(a))

        return c, scramble
