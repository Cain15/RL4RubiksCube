import torch
import torch.nn as nn
import torch.nn.functional as F

from src.CubeLog import MoveLogger
from src.Env import *

BIG = 1000

solved_state = np.asarray(RubiksCube().one_hot())

torch.set_num_threads(8)


class ResBlock(nn.Module):
    """
    A residual block for the Q-net
    """

    def __init__(self, w):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(w, w), nn.BatchNorm1d(w), nn.ReLU(),
                               nn.Linear(w, w), nn.BatchNorm1d(w))

    def forward(self, x):
        return torch.relu(x + self.f(x))


class QNet(nn.Module):
    """
    Q(s,a) = estimated moves to solve if action a is taken first
    """

    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(324, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )

        self.out = nn.Linear(256, 18)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.out(x)


@torch.no_grad()
def q_values(net, states):
    states = torch.as_tensor(states, dtype=torch.float32)
    if states.ndim == 1:
        states = states.unsqueeze(0)
    return net(states)


def heuristic(net, state):
    if np.array_equal(state, solved_state):
        return 0.0

    return q_values(net, state).min(dim=1).values.item()


def load_net(path="qnet.pt"):
    net = QNet()
    net.load_state_dict(torch.load(path))
    return net.eval()


class ReplayBuffer:
    """
    Will store transitions with an upper bound on the number of actions until completion
    """

    def __init__(self, capacity, state_dim=324):
        self.capacity = capacity
        self.size = 0
        self.position = 0

        # Pre-allocate memory
        self.states = torch.zeros(
            (capacity, state_dim), dtype=torch.float32
        )
        self.actions = torch.zeros(
            capacity, dtype=torch.long
        )
        self.next_states = torch.zeros(
            (capacity, state_dim), dtype=torch.float32
        )
        self.upper_bounds = torch.zeros(
            capacity, dtype=torch.float32
        )

    def add(self, state, action, next_state, upper_bound):
        """Add one transition."""
        self.add_batch(
            state, action, next_state, upper_bound
        )

    def add_batch(self, states, actions, next_states, upper_bounds):
        """Add multiple transitions."""
        states = torch.as_tensor(
            np.asarray(states), dtype=torch.float32
        )
        actions = torch.as_tensor(
            actions, dtype=torch.long
        ).reshape(-1)
        next_states = torch.as_tensor(
            np.asarray(next_states), dtype=torch.float32
        )
        upper_bounds = torch.as_tensor(
            upper_bounds, dtype=torch.float32
        ).reshape(-1)

        if states.ndim == 1:
            states = states.unsqueeze(0)

        if next_states.ndim == 1:
            next_states = next_states.unsqueeze(0)

        n = len(states)

        assert len(actions) == n
        assert len(next_states) == n
        assert len(upper_bounds) == n
        assert n <= self.capacity

        indices = (
                          torch.arange(n) + self.position
                  ) % self.capacity

        self.states[indices] = states
        self.actions[indices] = actions
        self.next_states[indices] = next_states
        self.upper_bounds[indices] = upper_bounds

        self.position = (self.position + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        """Sample a random batch of transitions."""

        indices = torch.randint(
            0, self.size, (batch_size,)
        )

        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
            self.upper_bounds[indices]
        )

    def __len__(self):
        return self.size


@torch.no_grad()
def rollout(net, buf, n_env, K, T, eps):
    """
    Run epsilon-greedy episodes from random scrambles, stored with no upperbounds
    """
    net.eval()

    cubes = random_cubes(n_env, K)
    for _ in range(T):
        # Find cubes that are not solved
        active = [
            i for i, cube in enumerate(cubes)
            if not cube.is_solved()
        ]

        if not active:
            break

        # Collect all active states
        states = np.asarray([
            cubes[i].one_hot() for i in active
        ])

        # Predict all actions in one forward pass
        greedy_actions = (
            q_values(net, states)
            .argmin(dim=1)
            .cpu()
            .numpy()
        )

        # Epsilon-greedy exploration
        random_mask = rng.random(len(active)) < eps
        random_actions = rng.integers(
            0, 18, size=len(active)
        )

        actions = np.where(
            random_mask,
            random_actions,
            greedy_actions
        )

        # Execute actions and collect transitions
        next_cubes = []
        next_states = []

        for idx, action in zip(active, actions):
            cube = cubes[idx]
            next_cube = cube.step(int(action))

            next_cubes.append(next_cube)
            next_states.append(next_cube.one_hot())

        buf.add_batch(
            states,
            actions,
            next_states,
            np.full(len(active), BIG)
        )

        # Update cubes for the next timestep
        for idx, next_cube in zip(active, next_cubes):
            cubes[idx] = next_cube


def train(iterations=100_000, K=26, warmup=500, target_every=250, n_demo=256,
          n_env=256, T=8, eps=0.2, path="qnet.pt", log_every=200):
    """
    Train the Q-network to estimate the remaining number of moves.
    """
    lr = 1e-4
    batch_size = 2048

    net, tgt = QNet(), QNet()
    tgt.load_state_dict(net.state_dict())
    tgt.eval()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    buf = ReplayBuffer(500_000)
    t0 = time.time()

    for it in range(1, iterations + 1):
        buf.add_batch(*demo_transitions(n_demo, K))

        if it > warmup:
            rollout(net, buf, n_env, K, T, eps)

        s, a, s2, ub = buf.sample(batch_size)

        with torch.no_grad():
            if it <= warmup:
                y = ub
            else:
                next_q = tgt(s2)
                min_next_q = next_q.min(dim=1).values.clamp(min=0)

                # A solved next state has zero remaining moves
                solved_mask = torch.all(s2 == solved_state, dim=1)

                min_next_q[solved_mask] = 0.0

                # Bellman update, limited by the known upper bound
                y = torch.minimum(1.0 + min_next_q, ub)

        net.train()

        q = net(s).gather(
            1, a.unsqueeze(1)
        ).squeeze(1)

        # --------------------------------------------------
        # 6. Calculate loss and update network
        # --------------------------------------------------

        loss = F.smooth_l1_loss(q, y)

        opt.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=10.0
        )

        opt.step()

        # --------------------------------------------------
        # 7. Update target network
        # --------------------------------------------------

        if it % target_every == 0:
            tgt.load_state_dict(net.state_dict())

        # --------------------------------------------------
        # 8. Logging
        # --------------------------------------------------

        if it % log_every == 0:
            elapsed = time.time() - t0

            print(
                f"Iteration {it:>7}/{iterations} | "
                f"Loss: {loss.item():.4f} | "
                f"Buffer: {len(buf)} | "
                f"Time: {elapsed:.1f}s"
            )

            if it % (log_every * 10) == 0:
                report(net)

        # --------------------------------------------------
        # 9. Save network
        # --------------------------------------------------

        if it % 5000 == 0:
            torch.save(net.state_dict(), path)

    torch.save(net.state_dict(), path)

    return net


def report(net, n=500, depths=(1, 2, 4, 6, 8, 10, 12, 14, 16, 20)):
    net.eval()

    results = []

    for depth in depths:
        estimates = []

        for _ in range(n):
            cube, _ = RubiksCube().scramble(depth, rng)
            estimates.append(
                heuristic(net, cube.one_hot())
            )

        mean_h = np.mean(estimates)
        results.append(f"{depth}: {mean_h:.2f}")

    print("Mean heuristic by scramble depth:")
    print(" | ".join(results), flush=True)


def is_solved(state):
    return np.array_equal(state, solved)


import time
import numpy as np

from RubiksCube import perms, solved

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Beam search device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(DEVICE))
    torch.backends.cuda.matmul.allow_tf32 = True

# Permutations used to generate child states
PERMS_GPU = torch.as_tensor(
    perms,
    dtype=torch.long,
    device=DEVICE
)

# One-hot lookup table
ONE_HOT_GPU = torch.eye(
    6,
    dtype=torch.float32,
    device=DEVICE
)

SOLVED_GPU = torch.as_tensor(
    solved,
    dtype=torch.long,
    device=DEVICE
)

# Opposite faces
opposite = {
    0: 3,
    1: 4,
    2: 5
}

# Allowed actions for each previous face
allowed_actions = {
    last_face: [
        action for action in range(18)
        if last_face == -1 or (
                action % 6 != last_face
                and opposite.get(action % 6) != last_face
        )
    ]
    for last_face in range(-1, 6)
}

ALLOWED_MASK_GPU = torch.tensor(
    [
        [
            action in allowed_actions[last_face]
            for action in range(18)
        ]
        for last_face in range(-1, 6)
    ],
    dtype=torch.bool,
    device=DEVICE
)


@torch.no_grad()
def evaluate_parents(net, states):
    """
    Evaluate parent states.

    states: (B, 54), integer cube states
    returns: (B, 18)
    """

    states = states.long()

    one_hot = ONE_HOT_GPU[states].reshape(
        states.shape[0], 324
    )

    return net(one_hot)


@torch.no_grad()
def evaluate_children(net, states, batch_size=65536):
    """
    Evaluate child states in batches.

    states: (N, 54)
    returns: (N,) heuristic scores

    Heuristic = minimum predicted Q-value.
    """
    states = states.long()

    scores = []

    for start in range(0, len(states), batch_size):
        batch = states[start:start + batch_size]

        one_hot = ONE_HOT_GPU[batch].reshape(
            len(batch), 324
        )

        q = net(one_hot)

        scores.append(q.min(dim=1).values)

    return torch.cat(scores)


@torch.no_grad()
def beam_search(net, cube, beam_width=1 << 14, max_depth=40, score_children=False, inference_batch_size=65536):
    """
    use beam search to find the solved state, optimised for GPU
    """
    net.to(DEVICE)
    net.eval()

    if cube.is_solved():
        return [], 0

    max_nodes = max_depth * beam_width + 1

    parent_ids = np.empty(
        max_nodes,
        dtype=np.int32
    )

    parent_actions = np.empty(
        max_nodes,
        dtype=np.int8
    )

    parent_ids[0] = -1
    parent_actions[0] = -1

    next_id = 1

    def reconstruct_path(node_id):

        path = []

        while node_id != 0:
            path.append(int(parent_actions[node_id]))
            node_id = int(parent_ids[node_id])

        return path[::-1]

    beam_states = torch.as_tensor(
        cube.state.copy(),
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    beam_node_ids = np.array(
        [0],
        dtype=np.int32
    )

    beam_last_faces = torch.tensor(
        [-1],
        dtype=torch.long,
        device=DEVICE
    )

    for depth in range(max_depth):

        B = len(beam_states)

        if B == 0:
            break

        children = beam_states[:, PERMS_GPU]

        # Determine valid actions for each parent
        valid_mask = ALLOWED_MASK_GPU[
            beam_last_faces + 1
            ]

        # Flatten children
        children = children.reshape(-1, 54)

        valid_flat = valid_mask.reshape(-1)

        # Keep only valid children
        children = children[valid_flat]

        # Corresponding parent indices and actions
        parent_indices = torch.arange(
            B,
            device=DEVICE
        ).unsqueeze(1).expand(B, 18).reshape(-1)

        actions = torch.arange(
            18,
            device=DEVICE
        ).unsqueeze(0).expand(B, 18).reshape(-1)

        parent_indices = parent_indices[valid_flat]
        actions = actions[valid_flat]

        faces = actions % 6

        solved_mask = (
                children == SOLVED_GPU
        ).all(dim=1)

        solved_indices = torch.nonzero(
            solved_mask,
            as_tuple=False
        ).flatten()

        if len(solved_indices) > 0:
            idx = int(solved_indices[0].item())

            parent_idx = int(parent_indices[idx].item())
            action = int(actions[idx].item())

            node_id = int(beam_node_ids[parent_idx])

            return (
                reconstruct_path(node_id) + [action],
                depth + 1
            )

        if score_children:

            # Evaluate resulting child states
            scores = evaluate_children(
                net,
                children,
                batch_size=inference_batch_size
            )

        else:

            # Evaluate parent states
            parent_q = evaluate_parents(
                net,
                beam_states
            )

            # Score each action using its parent's Q-values
            scores = parent_q[
                parent_indices,
                actions
            ]

        keys = torch.cat(
            [
                children,
                faces.to(torch.uint8).unsqueeze(1)
            ],
            dim=1
        )

        unique_keys, inverse = torch.unique(
            keys,
            dim=0,
            return_inverse=True
        )

        n_unique = len(unique_keys)
        n_candidates = len(scores)

        # Minimum score for each unique state
        unique_scores = torch.full(
            (n_unique,),
            float("inf"),
            dtype=scores.dtype,
            device=DEVICE
        )

        unique_scores.scatter_reduce_(
            0,
            inverse,
            scores,
            reduce="amin",
            include_self=True
        )

        # Find the candidate index that produced the
        # minimum score for each unique state.
        candidate_indices = torch.arange(
            n_candidates,
            dtype=torch.long,
            device=DEVICE
        )

        matching_indices = torch.where(
            scores == unique_scores[inverse],
            candidate_indices,
            n_candidates
        )

        best_candidate = torch.full(
            (n_unique,),
            n_candidates,
            dtype=torch.long,
            device=DEVICE
        )

        best_candidate.scatter_reduce_(
            0,
            inverse,
            matching_indices,
            reduce="amin",
            include_self=True
        )

        k = min(beam_width, n_unique)

        selected_scores, selected_groups = torch.topk(
            unique_scores,
            k=k,
            largest=False,
            sorted=True
        )

        selected_indices = best_candidate[selected_groups]

        # Select corresponding child states
        new_beam_states = children[selected_indices]

        new_beam_faces = faces[selected_indices]

        selected_parent_indices = (
            parent_indices[selected_indices]
            .cpu()
            .numpy()
        )

        selected_actions = (
            actions[selected_indices]
            .cpu()
            .numpy()
        )

        new_beam_node_ids = np.arange(
            next_id,
            next_id + k,
            dtype=np.int32
        )

        for i in range(k):
            node_id = next_id + i

            parent_idx = selected_parent_indices[i]

            parent_ids[node_id] = beam_node_ids[parent_idx]

            parent_actions[node_id] = selected_actions[i]

        next_id += k

        beam_states = new_beam_states

        beam_last_faces = new_beam_faces

        beam_node_ids = new_beam_node_ids

    return None


if __name__ == "__main__":
    # train()
    net = load_net("qnet_best.pt")
    net.to(DEVICE)
    report(net)

    t0 = time.time()
    lengths = []
    logger = MoveLogger("solution.jsonl", False)
    n = 100
    benchmark_rng = np.random.default_rng(42)

    scrambles = [
        RubiksCube().scramble(100, benchmark_rng)
        for _ in range(n)
    ]
    for cube, scramble in scrambles:
        result = beam_search(net, cube, beam_width=1 << 17)
        if result:
            solution, length = result
            print("Found solution of length:", length)
            logger.reset(scramble)
            for a in solution:
                cube = cube.step(a)
                logger.step(a)
            print("cube solved:", cube.is_solved())
            lengths.append(length)
    logger.close()
    elapsed = time.time() - t0
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    success = len(lengths)
    stats = (
        f"Time: {elapsed:.2f}s | "
        f"Average length: {avg_length:.2f} | "
        f"Success: {success}/{n} "
        f"({success / n * 100:.1f}%)\n"
    )

    print(stats)

    with open("benchmark_results.txt", "a", encoding="utf-8") as f:
        f.write(stats)

    # Time: 563.07s | Average length: 20.70 | Success: 10/10 (100.0%)

    # Time: 328.35s | Average length: 20.70 | Success: 10/10 (100.0%)
