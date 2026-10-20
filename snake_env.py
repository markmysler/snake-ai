"""Headless, fast Snake environment with a Gymnasium-style interface.

    env = SnakeEnv(10, 10, reward="shaped", vision="rays:depth=4", seed=0)
    state = env.reset()
    state, reward, done, info = env.step(action)   # action: 0=left, 1=straight, 2=right

What the policy sees (the state) is defined in vision.py; the reward in rewards.py.
"""
from collections import deque

import numpy as np

import rewards
import vision as visions
from vision import DIRS

UP, RIGHT, DOWN, LEFT = range(4)
TURN_LEFT, STRAIGHT, TURN_RIGHT = 0, 1, 2
# How a game ended, one letter per game: hit a wall, hit its body, starved, filled the board
ENDS = {"w": "wall", "b": "body", "s": "starved", "f": "won"}


def end_code(info):
    """The ENDS letter for the info of a game's last step."""
    if info["collision"]:
        return "w" if info.get("crash") == "wall" else "b"
    return "f" if info["won"] else "s"


def end_counts(codes):
    """{"wall": n, "body": n, "starved": n, "won": n} from end letters."""
    codes = list(codes)
    return {name: codes.count(code) for code, name in ENDS.items()}
N_ACTIONS = 3
STATE_SIZE = visions.state_size("short")  # the original 11 features


class SnakeEnv:
    def __init__(self, width=10, height=10, reward="score", seed=None,
                 stall_factor=100, init_length=3, vision="short"):
        if min(width, height) < init_length + 2:
            raise ValueError("grid too small for the initial snake")
        self.width, self.height = width, height
        self.vision = visions.parse(vision)
        self.observe = visions.make(self.vision)
        self.state_size = visions.state_size(self.vision, width, height)
        self.reward_spec = rewards.parse(reward)
        self.rewarder = rewards.Rewarder(self.reward_spec)
        self.stall_factor = stall_factor
        self.init_length = init_length
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ---- core API -------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.direction = int(self.rng.integers(4))
        dx, dy = DIRS[self.direction]
        hx, hy = self.width // 2, self.height // 2
        self.snake = deque((hx - i * dx, hy - i * dy) for i in range(self.init_length))
        self.occupied = set(self.snake)
        self.score = self.steps = self.steps_since_food = 0
        self._place_food()
        return self.observe(self)

    def step(self, action):
        if action == TURN_LEFT:
            self.direction = (self.direction - 1) % 4
        elif action == TURN_RIGHT:
            self.direction = (self.direction + 1) % 4
        dx, dy = DIRS[self.direction]
        hx, hy = self.snake[0]
        new = (hx + dx, hy + dy)
        self.steps += 1
        self.steps_since_food += 1
        info = {"score": self.score, "steps": self.steps,
                "collision": False, "truncated": False, "won": False, "crash": None}
        before = self._food_distance(hx, hy)

        if self.is_danger(*new):
            info["collision"] = True
            info["crash"] = "wall" if not (0 <= new[0] < self.width and 0 <= new[1] < self.height) else "body"
            reward = self.rewarder(False, info["crash"], False, False, before, None)
            return self.observe(self), reward, True, info

        eat = new == self.food
        if not eat:  # free the tail first: moving into the old tail cell is legal
            self.occupied.discard(self.snake.pop())
        self.snake.appendleft(new)
        self.occupied.add(new)

        done = starved = won = False
        if eat:
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
            won = self.food is None  # board is full
        elif self.steps_since_food >= self.stall_factor * len(self.snake):
            starved = True
        done = won or starved
        info.update(score=self.score, truncated=starved, won=won)
        reward = self.rewarder(eat, False, starved, won, before, self._food_distance(*new))
        return self.observe(self), reward, done, info

    # ---- helpers --------------------------------------------------------
    def _food_distance(self, x, y):
        return None if self.food is None else abs(x - self.food[0]) + abs(y - self.food[1])

    def is_danger(self, x, y):
        """True if moving the head onto (x, y) next step would kill the snake."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        # The tail moves away this step (food never sits on the tail).
        return (x, y) in self.occupied and (x, y) != self.snake[-1]

    def _place_food(self):
        free = self.width * self.height - len(self.snake)
        if free == 0:
            self.food = None
            return
        for _ in range(32):  # rejection sampling is fast while the snake is short
            cell = (int(self.rng.integers(self.width)), int(self.rng.integers(self.height)))
            if cell not in self.occupied:
                self.food = cell
                return
        cells = [(x, y) for y in range(self.height) for x in range(self.width)
                 if (x, y) not in self.occupied]
        self.food = cells[int(self.rng.integers(len(cells)))]

    def set_state(self, snake, direction, food):
        """Place the snake (head first), heading and food explicitly (used by tests)."""
        self.snake = deque(snake)
        self.occupied = set(self.snake)
        self.direction = direction
        self.food = food
        self.steps_since_food = 0
        return self.observe(self)
