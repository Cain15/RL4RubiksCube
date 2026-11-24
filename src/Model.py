import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.CubeLog import MoveLogger
from src.Env import *
from src.RubiksCube import solved

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


import heapq
from RubiksCube import perms, solved

opposite = {
    0: 3,  # U -> D
    1: 4,  # R -> L
    2: 5   # F -> B
}

one_hot_table = np.eye(6, dtype=np.float32)

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


@torch.no_grad()
def beam_search(net, cube, beam_width=1 << 14,
                max_depth=40, score_children=False):

    net.eval()

    if cube.is_solved():
        return [], 0

    # Parent records: only store parent ID and action
    max_nodes = max_depth * beam_width + 1

    parent_ids = np.empty(max_nodes, dtype=np.int32)
    parent_actions = np.empty(max_nodes, dtype=np.int8)

    # Root node
    parent_ids[0] = -1
    parent_actions[0] = -1

    next_id = 1

    def reconstruct_path(node_id):
        path = []

        while node_id != 0:
            path.append(int(parent_actions[node_id]))
            node_id = int(parent_ids[node_id])

        return path[::-1]

    # Node: (state, node_id, last_face)
    beam = [(cube.state.copy(), 0, -1)]

    for depth in range(max_depth):

        candidates = {}

        # Batch inference for all parents
        if not score_children:
            states = np.stack([
                state for state, _, _ in beam
            ])

            parent_states = one_hot_table[states].reshape(
                len(beam), 324
            )

            parent_q = q_values(net, parent_states).cpu().numpy()

        for i, (state, node_id, last_face) in enumerate(beam):

            for action in allowed_actions[last_face]:
                face = action % 6

                # Generate child state directly
                child_state = state[perms[action]]

                if np.array_equal(child_state, solved):
                    return reconstruct_path(node_id) + [action], depth + 1

                key = (child_state.tobytes(), face)

                if score_children:
                    child_one_hot = one_hot_table[child_state].ravel()
                    score = heuristic(net, child_one_hot)
                else:
                    score = float(parent_q[i, action])

                old = candidates.get(key)

                if old is None or score < old[1]:
                    candidates[key] = (
                        child_state,
                        score,
                        node_id,
                        action,
                        face
                    )

        if not candidates:
            break

        # Select the best candidates
        selected = heapq.nsmallest(
            beam_width,
            candidates.values(),
            key=lambda x: x[1]
        )

        new_beam = []

        for state, score, parent_id, action, face in selected:

            node_id = next_id
            next_id += 1

            parent_ids[node_id] = parent_id
            parent_actions[node_id] = action

            new_beam.append((state, node_id, face))

        beam = new_beam

    return None


if __name__ == "__main__":
    # train()
    net = load_net("qnet_best.pt")
    report(net)

    t0 = time.time()
    lengths = []
    logger = MoveLogger("solution.jsonl", False)
    n = 1000
    benchmark_rng = np.random.default_rng(42)

    scrambles = [
        RubiksCube().scramble(100, benchmark_rng)
        for _ in range(n)
    ]
    for cube, scramble in scrambles:
        result = beam_search(net, cube, beam_width=1 << 16)
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

