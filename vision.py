"""What the snake sees: functions that turn the game into the numbers a network reads.

Each vision is registered in VISIONS with a label, a description and its parameters, so the
command line and the dashboard's "New run" form can offer it without extra code. A vision is
written as a spec, e.g. "short", "full", "rays:depth=4", "window:radius=3" (or the same as a
dict: {"type": "window", "radius": 3}).

To add a vision: write view(env, **params) -> 1-D float32 array with values in [0, 1],
a size(width, height, **params) function, and register(Vision(...)) at the bottom.
"""
import numpy as np

# Directions in clockwise order; (dx, dy) with y growing downward.
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Vision:
    def __init__(self, name, label, description, view, size, params=()):
        self.name, self.label, self.description = name, label, description
        self.view, self.size, self.params = view, size, list(params)

    def info(self):
        return {"name": self.name, "label": self.label, "description": self.description, "params": self.params}


VISIONS = {}


def register(vision):
    VISIONS[vision.name] = vision


def int_param(name, label, default, lo, hi, help=""):
    return {"name": name, "label": label, "type": "int", "default": default, "min": lo, "max": hi, "help": help}


# ---- specs ------------------------------------------------------------------------
def parse(spec):
    """'window:radius=3' | 'full' | {'type': 'window', 'radius': 3} -> {'type': ..., every param}."""
    if isinstance(spec, dict):
        name, given = spec.get("type", "short"), {k: v for k, v in spec.items() if k != "type"}
    else:
        name, _, rest = str(spec).partition(":")
        given = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    if name not in VISIONS:
        raise ValueError(f"unknown vision {name!r}; choose from {sorted(VISIONS)}")
    out = {"type": name}
    for p in VISIONS[name].params:
        value = int(given.pop(p["name"], p["default"]))
        if not p["min"] <= value <= p["max"]:
            raise ValueError(f"{name} {p['name']} must be between {p['min']} and {p['max']}")
        out[p["name"]] = value
    if given:
        raise ValueError(f"unknown parameter(s) for vision {name!r}: {sorted(given)}")
    return out


def to_string(spec):
    spec = parse(spec)
    params = ",".join(f"{k}={v}" for k, v in spec.items() if k != "type")
    return spec["type"] + (f":{params}" if params else "")


def label(spec):
    spec = parse(spec)
    params = ", ".join(f"{k} {v}" for k, v in spec.items() if k != "type")
    return VISIONS[spec["type"]].label.split(" · ")[0] + (f" ({params})" if params else "")


def _params(spec):
    return {k: v for k, v in spec.items() if k != "type"}


def state_size(spec, width=10, height=10):
    spec = parse(spec)
    return VISIONS[spec["type"]].size(width, height, **_params(spec))


def make(spec):
    """The observation function env -> state for this spec."""
    spec = parse(spec)
    view, params = VISIONS[spec["type"]].view, _params(spec)
    return lambda env: view(env, **params)


# ---- shared helpers -----------------------------------------------------------------
def food_flags(env):
    """Is the apple ahead / behind / to the left / to the right of the head (snake's point of view)?"""
    if env.food is None:
        return np.zeros(4, dtype=np.float32)
    hx, hy = env.snake[0]
    rx, ry = env.food[0] - hx, env.food[1] - hy
    ax, ay = DIRS[env.direction]
    sx, sy = DIRS[(env.direction + 1) % 4]  # the snake's right
    ahead, right = rx * ax + ry * ay, rx * sx + ry * sy
    return np.array([ahead > 0, ahead < 0, right < 0, right > 0], dtype=np.float32)


# ---- short: 11 hand-made hints (lesson 01) --------------------------------------------
def short_view(env):
    """[danger straight, left, right,  heading up, right, down, left,  food left, right, up, down]"""
    hx, hy = env.snake[0]
    d = env.direction
    s = np.zeros(11, dtype=np.float32)
    for i, turn in enumerate((0, -1, 1)):  # straight, left, right
        dx, dy = DIRS[(d + turn) % 4]
        s[i] = env.is_danger(hx + dx, hy + dy)
    s[3 + d] = 1.0
    if env.food is not None:
        fx, fy = env.food
        s[7], s[8], s[9], s[10] = fx < hx, fx > hx, fy < hy, fy > hy
    return s


# ---- room: how much of the board each move keeps reachable (flood fill) ------------------
def room_shares(env):
    """For the moves straight, left, right: the share of the board's free squares the snake could still
    reach after making it (0 = the move crashes). Squares are blocked by walls and the body, except the
    tail, which moves away. A move into a small pocket scores low long before it becomes a crash."""
    w, h = env.width, env.height
    stride = w + 2  # a flat board with a border of wall squares: no bounds checks while flooding
    grid = bytearray(b"" * (stride * (h + 2)))  # 0 = free, 1 = blocked, 2+ = pocket label
    for y in range(h):
        row = (y + 1) * stride + 1
        grid[row:row + w] = bytes(w)
    for x, y in env.occupied:
        grid[(y + 1) * stride + x + 1] = 1
    tx, ty = env.snake[-1]
    grid[(ty + 1) * stride + tx + 1] = 0
    free = max(w * h - len(env.occupied) + 1, 1)
    steps = (1, -1, stride, -stride)
    hx, hy = env.snake[0]
    out = np.zeros(3, dtype=np.float32)
    sizes = [0, 0]  # pocket label -> number of squares
    for i, turn in enumerate((0, -1, 1)):
        dx, dy = DIRS[(env.direction + turn) % 4]
        start = (hy + dy + 1) * stride + hx + dx + 1
        if grid[start] == 1:
            continue  # wall or body: this move crashes
        if grid[start] == 0:  # a pocket not explored yet: flood it
            label, count, stack = len(sizes), 1, [start]
            grid[start] = label
            while stack:
                c = stack.pop()
                for d in steps:
                    if grid[c + d] == 0:
                        grid[c + d] = label
                        count += 1
                        stack.append(c + d)
            sizes.append(count)
        out[i] = sizes[grid[start]] / free
    return out


def room_view(env):
    """The 11 adjacent hints, then the room left after moving straight, left, right."""
    return np.concatenate([short_view(env), room_shares(env)])


# ---- window / full: pictures around the head ------------------------------------------
def window_picture(env, radius):
    """3 stacked (2r+1) x (2r+1) pictures centred on the head, rotated so the snake faces up:
      blocked  1 = wall (outside the board) or a body cell still there next move
      body     moves until each body cell is free again, divided by the length
      food     1 where the apple is
    """
    w, h, side = env.width, env.height, 2 * radius + 1
    hx, hy = env.snake[0]
    view = np.zeros((3, side, side), dtype=np.float32)
    view[0] = 1.0  # everything is wall ...
    x0, x1, y0, y1 = max(0, hx - radius), min(w, hx + radius + 1), max(0, hy - radius), min(h, hy + radius + 1)
    view[0, y0 - hy + radius:y1 - hy + radius, x0 - hx + radius:x1 - hx + radius] = 0.0  # ... except the board

    n = len(env.snake)
    if n > 1:
        body = np.array(env.snake, dtype=np.int64)[1:]  # skip the head (always the centre)
        xs, ys = body[:, 0] - hx + radius, body[:, 1] - hy + radius
        inside = (xs >= 0) & (xs < side) & (ys >= 0) & (ys < side)
        moves_left = np.arange(n - 1, 0, -1, dtype=np.float32) / n  # neck (n-1)/n ... tail 1/n
        view[1, ys[inside], xs[inside]] = moves_left[inside]
        blocked = inside.copy()
        blocked[-1] = False  # the tail moves away this step, so it does not block
        view[0, ys[blocked], xs[blocked]] = 1.0
    if env.food is not None:
        fx, fy = env.food[0] - hx + radius, env.food[1] - hy + radius
        if 0 <= fx < side and 0 <= fy < side:
            view[2, fy, fx] = 1.0
    # np.rot90 turns counter-clockwise: heading right (1) needs one quarter turn, down 2, left 3.
    return np.rot90(view, k=env.direction, axes=(1, 2)).ravel()


def full_radius(width, height):
    return max(width, height) - 1  # big enough to show the whole board wherever the head is


def full_view(env):
    return window_picture(env, full_radius(env.width, env.height))


def window_view(env, radius):
    return np.concatenate([window_picture(env, radius), food_flags(env)])


# ---- rays: look N squares along 8 lines ------------------------------------------------
def ray_directions(d):
    (ax, ay), (rx, ry) = DIRS[d], DIRS[(d + 1) % 4]
    return [(ax, ay), (ax + rx, ay + ry), (rx, ry), (rx - ax, ry - ay),
            (-ax, -ay), (-ax - rx, -ay - ry), (-rx, -ry), (ax - rx, ay - ry)]


def rays_view(env, depth):
    """For 8 lines (ahead, ahead-right, right, ... ahead-left), how close are the nearest wall,
    body and apple within `depth` squares (1 = adjacent, 0 = not seen); plus food direction."""
    out = np.zeros(8 * 3 + 4, dtype=np.float32)
    w, h, occupied, tail, food = env.width, env.height, env.occupied, env.snake[-1], env.food
    hx, hy = env.snake[0]
    for i, (dx, dy) in enumerate(ray_directions(env.direction)):
        x, y = hx, hy
        for k in range(1, depth + 1):
            x, y = x + dx, y + dy
            near = (depth - k + 1) / depth
            if not (0 <= x < w and 0 <= y < h):
                out[3 * i] = near
                break
            if not out[3 * i + 1] and (x, y) in occupied and (x, y) != tail:
                out[3 * i + 1] = near
            if (x, y) == food:
                out[3 * i + 2] = near
    out[24:] = food_flags(env)
    return out


register(Vision("short", "Adjacent · 11 hints",
                "Danger in the next square straight / left / right, the heading, and which way the "
                "apple is. Tiny and quick to learn from, but blind beyond one square (lesson 01).",
                lambda env: short_view(env), lambda w, h: 11))
register(Vision("room", "Adjacent + room to move",
                "The 11 adjacent hints plus, for each move, how much of the free board the snake could "
                "still reach after it (a flood fill). Warns about boxing itself in before it is too late.",
                lambda env: room_view(env), lambda w, h: 14))
register(Vision("rays", "Rays · N squares ahead",
                "Looks up to N squares along 8 lines around its heading and reports how close the "
                "nearest wall, body part and apple are on each line, plus which way the apple is.",
                rays_view, lambda w, h, depth: 28,
                [int_param("depth", "Squares ahead", 4, 1, 20, "how far each ray looks")]))
register(Vision("window", "Window · square around the head",
                "A picture of the squares within N steps of the head, rotated so the snake faces up, "
                "plus which way the apple is (it may be outside the window).",
                window_view, lambda w, h, radius: 3 * (2 * radius + 1) ** 2 + 4,
                [int_param("radius", "Radius (squares)", 2, 1, 9, "window side = 2 x radius + 1")]))
register(Vision("full", "Whole board",
                "The entire board as pictures centred on the head and rotated so the snake faces up. "
                "Sees everything, but has many inputs (lesson 02).",
                lambda env: full_view(env), lambda w, h: 3 * (2 * full_radius(w, h) + 1) ** 2))
