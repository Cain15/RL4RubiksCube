import json
from pathlib import Path


class MoveLogger:
    """
    Appends one JSON object per line and flushes each time for robustness
    """

    def __init__(self, path, append=False):
        """
        Constructs a MoveLogger object
        """
        self.path = Path(path)
        self._f = open(self.path, "a" if append else "w", buffering=1)
        self.episode = -1

    def reset(self, scramble=()):
        """
        The cube is reset, log a reset and show the scramble
        """
        self.episode += 1
        self._write({"type": "reset", "episode": self.episode, "scramble": [int(a) for a in scramble]})

    def step(self, action, reward=None):
        """
        Log an action
        """
        if self.episode < 0:
            self.reset()
        rec = {"type": "step", "episode": self.episode, "action": int(action)}
        if reward is not None:
            rec["reward"] = float(reward)
        self._write(rec)

    def _write(self, rec):
        """
        Writes a JSON object to a file
        """
        self._f.write(json.dumps(rec) + "\n")

    def close(self):
        """
        Closes the file
        """
        self._f.close()

    def __enter__(self):
        """
        Use the object on entry
        """
        return self

    def __exit__(self, *exc):
        """
        close the file on exit
        """
        self.close()


def read_log(path):
    """
    Return a list of episodes
    """
    episodes, cur = [], None
    for line in Path(path).read_text().splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") == "reset":
            cur = {"id": o.get("episode", len(episodes)), "scramble": o.get("scramble", []), "steps": []}
            episodes.append(cur)
        elif "action" in o:
            if cur is None:
                cur = {"id": 0, "scramble": [], "steps": []}
                episodes.append(cur)
            cur["steps"].append((o["action"], o.get("reward")))

    return episodes
