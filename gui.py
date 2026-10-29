"""Watch Value Iteration, Q-learning and SARSA learn FrozenLake, in a small winter-themed window.

    python gui.py              # 4x4 map
    python gui.py --map 8x8

Keys: 1 / 2 / 3 choose the method (restarts it), Space pauses, Up / Down change speed,
S skips 50 episodes (or the rest of planning), R restarts, Esc quits.

The environment, rewards and algorithms are the ones in ``frozenlake_rl``. Q-learning and SARSA run
the real ``train_td`` in a background thread, which waits after every step until the window has
animated it.
"""
import argparse
import math
import queue
import random
import threading

import numpy as np
import pygame

from frozenlake_rl import agents
from frozenlake_rl.config import CONFIG
from frozenlake_rl.env import build_model, grid_shape, make_env, play_episode, start_state, terminal_states, tiles
from frozenlake_rl.experiments import TD_METHODS

METHOD_KEYS = {pygame.K_1: "Value Iteration", pygame.K_2: "Q-learning", pygame.K_3: "SARSA"}
SPEEDS = [1, 2, 4, 8, 15, 30, 60]   # moves per second
SKIP_EPISODES = 50
RESULT_TIME = 0.9                        # seconds the episode result stays on screen (shorter at high speed)

NIGHT = (14, 22, 44)
BAR = (8, 13, 28)
TEXT = (226, 236, 248)
TEXT_DIM = (130, 150, 180)
ICE = (196, 226, 245)          # "no idea yet"
WARM = (255, 176, 102)         # looks good (leads to the goal)
COLD = (128, 150, 196)         # looks bad (leads to a hole)
WATER = (18, 42, 82)
GREEDY_STEP = (255, 205, 80)   # footprints: best-looking move
RANDOM_STEP = (235, 90, 205)   # footprints: random (exploring) move
DIRECTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # left, down, right, up
RESULTS = {"goal": "Reached the goal!", "hole": "Fell through the ice!", "timeout": "Ran out of steps"}


def lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def value_colour(v):
    """Cold -> icy -> warm as a tile looks worse -> unknown -> better (square root so small values show)."""
    t = math.copysign(min(abs(v), 1.0) ** 0.5, v)
    return lerp(ICE, WARM, t) if t >= 0 else lerp(ICE, COLD, -t)


class StopTraining(Exception):
    pass


class TDRunner:
    """Runs ``train_td`` in a thread, pausing after each step until the window calls ``release``."""

    def __init__(self, method, map_key, seed):
        cfg, td, ev = CONFIG["maps"][map_key], CONFIG["td"], CONFIG["evaluation"]
        self.n_episodes = cfg["n_episodes"]
        self.items = queue.Queue()
        self.go = threading.Semaphore(0)
        self.stop = False
        self.skip_to = 0   # don't wait for the window while episode <= skip_to
        kwargs = dict(n_episodes=cfg["n_episodes"], gamma=CONFIG["gamma"], alpha=td["alpha"],
                      epsilon_start=td["epsilon_start"], epsilon_end=td["epsilon_end"],
                      eps_decay_episodes=cfg["eps_decay_episodes"], q_init=td["q_init"], seed=seed,
                      eval_every=ev["eval_every"], eval_episodes=ev["eval_episodes"], step_callback=self._on_step)
        self.thread = threading.Thread(target=self._run, args=(TD_METHODS[method], map_key, kwargs), daemon=True)
        self.thread.start()

    def _run(self, method, map_key, kwargs):
        try:
            agents.train_td(method, make_env(map_key), make_env(map_key), **kwargs)
            self.items.put(("finished", None, None, False))
        except StopTraining:
            pass

    def _on_step(self, step, Q):
        if self.stop:
            raise StopTraining
        waits = step["episode"] > self.skip_to
        self.items.put(("step", step, Q.copy(), waits))
        if waits:
            self.go.acquire()
            if self.stop:
                raise StopTraining

    def release(self):
        self.go.release()

    def close(self):
        self.stop = True
        self.go.release()
        self.thread.join(timeout=1)


class App:
    def __init__(self, map_key="4x4", seed=0):
        pygame.init()
        pygame.display.set_caption("FrozenLake: watch the agent learn")
        self.map_key, self.seed = map_key, seed
        self.env = make_env(map_key)
        self.tiles, (self.rows, self.cols) = tiles(self.env), grid_shape(self.env)
        self.tile = 120 if max(self.rows, self.cols) <= 4 else 72
        grid_w, grid_h = self.cols * self.tile, self.rows * self.tile
        self.width = max(grid_w + 80, 840)
        self.height = 44 + grid_h + 30 + 58
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.grid_x, self.grid_y = (self.width - grid_w) // 2, 44
        self.font = pygame.font.SysFont("helvetica", 17)
        self.small = pygame.font.SysFont("helvetica", 14)
        self.big = pygame.font.SysFont("helvetica", 34, bold=True)
        self.clock = pygame.time.Clock()
        self.flakes = [[random.uniform(0, self.width), random.uniform(0, self.height), random.uniform(1, 2.6),
                        random.uniform(12, 35), random.uniform(0, 6.3)] for _ in range(70)]
        self.time = 0.0
        self.speed_i = 2
        self.paused = False
        self.running = True
        self.runner = None
        self.start("Q-learning")

    # ----------------------------------------------------------------------- state
    def start(self, method):
        if self.runner:
            self.runner.close()
            self.runner = None
        self.method = method
        n_states, n_actions = self.env.observation_space.n, self.env.action_space.n
        self.Q = np.zeros((n_states, n_actions))
        self.shown = np.zeros(n_states)            # displayed tile values, eased towards max Q
        self.agent = start_state(self.env)          # tile the agent stands on
        self.move = None                            # (from, to, progress 0..1) while walking
        self.footprints = []                        # [(x, y, explored)] in tile units
        self.result, self.result_timer = "", 0.0    # result banner
        self.last_result = "none yet"
        self.episode, self.steps, self.epsilon = 0, 0, None
        self.finished = False
        if method == "Value Iteration":
            P = build_model(self.env)
            res = agents.value_iteration(P, n_states, n_actions, CONFIG["gamma"],
                                         CONFIG["value_iteration"]["theta"], terminal_states(self.env))
            self.vi_q = [agents.q_from_v(P, V, CONFIG["gamma"], n_states, n_actions) for V in res.V_history]
            self.vi_path = play_episode(make_env(self.map_key), res.policy, seed=self.seed)
            self.sweep, self.sweep_timer, self.walk_i = 0, 0.0, None
        else:
            self.runner = TDRunner(method, self.map_key, self.seed)
            self.current = None                     # (step dict, Q after it) being animated

    @property
    def speed(self):
        return SPEEDS[self.speed_i]

    def skip(self):
        if self.method == "Value Iteration":
            self.sweep, self.Q = len(self.vi_q), self.vi_q[-1]
            return
        self.runner.skip_to = self.episode + SKIP_EPISODES
        if self.current is not None:      # the worker is waiting on this step: let it go
            self.current = None
            self.result_timer = 0.0
            self.runner.release()
        self.move = None

    # ----------------------------------------------------------------------- update
    def update(self, dt):
        self.time += dt
        for f in self.flakes:
            f[1] += f[3] * dt
            f[0] += math.sin(self.time * 0.8 + f[4]) * 0.3
            if f[1] > self.height:
                f[0], f[1] = random.uniform(0, self.width), -5
        self.shown += (self.Q.max(axis=1) - self.shown) * min(1.0, dt * 4)   # gentle tint changes
        if self.paused:
            return
        if self.method == "Value Iteration":
            self._update_vi(dt)
        else:
            self._update_td(dt)

    def _walk(self, dt):
        """Advance the current move by ``dt`` seconds. Returns the time left over once it ends, else None."""
        frm, to, t = self.move
        t += dt * self.speed
        if t < 1.0:
            self.move = (frm, to, t)
            return None
        self.agent, self.move = to, None
        return (t - 1.0) / self.speed

    def _update_vi(self, dt):
        self.result_timer -= dt
        if self.sweep < len(self.vi_q):
            self.sweep_timer += dt
            if self.sweep_timer >= 4 / self.speed:   # one sweep per 4 move-times
                self.sweep_timer = 0.0
                self.Q = self.vi_q[self.sweep]
                self.sweep += 1
            return
        if self.walk_i is None:          # planning done: play one episode with the final policy
            self.walk_i = 0
        if self.move and self._walk(dt) is None:
            return
        if self.walk_i + 1 < len(self.vi_path):
            frm, to = self.vi_path[self.walk_i], self.vi_path[self.walk_i + 1]
            self.footprints.append((*self._footprint(frm, to), False))
            self.move = (frm, to, 0.0)
            self.walk_i += 1
            self.steps = self.walk_i
        elif not self.finished:
            self.finished = True
            self.last_result = RESULTS[{"G": "goal", "H": "hole"}.get(self.tiles[self.agent], "timeout")]
            self.result, self.result_timer = self.last_result, RESULT_TIME * 2

    def _update_td(self, dt):
        """Spend this frame's time on result banners and moves, so fast speeds can do several moves per frame."""
        while True:
            if self.result_timer > 0:
                self.result_timer -= dt
                if self.result_timer <= 0 and self.current is not None:
                    self.current = None
                    self.runner.release()
                return
            if self.move:
                dt = self._walk(dt)
                if dt is None:
                    return
                self._finish_step()
            elif not self._next_step():
                return

    def _next_step(self):
        """Take the next step from the trainer and start animating it. False if none is ready yet."""
        while self.current is None:
            try:
                kind, step, Q, waits = self.runner.items.get_nowait()
            except queue.Empty:
                return False
            if kind == "finished":
                self.finished = True
                return False
            self.Q = Q
            self.episode, self.steps, self.epsilon = step["episode"], step["step"], step["epsilon"]
            if step["step"] == 1:
                self.footprints, self.agent = [], start_state(self.env)
            if step["done"]:
                self.last_result = self._result(step)
            if not waits:                 # skipping ahead: apply without animating
                self.agent = step["next_state"]
                continue
            self.current = (step, Q)
            s, s2 = step["state"], step["next_state"]
            self.footprints.append((*self._footprint(s, s2, step["action"]), step["explored"]))
            self.move = (s, s2, 0.0)
        return self.move is not None

    def _finish_step(self):
        step, _Q = self.current
        if step["done"]:
            self.result = self.last_result
            self.result_timer = RESULT_TIME if self.speed <= 8 else 0.25
        else:
            self.current = None
            self.runner.release()

    def _result(self, step):
        tile = self.tiles[step["next_state"]]
        return RESULTS["goal" if tile == "G" else "hole" if tile == "H" else "timeout"]

    def _footprint(self, s, s2, action=None):
        """Footprint between the two tiles (or against the edge, if the move bumped into it)."""
        (x1, y1), (x2, y2) = self._cell(s), self._cell(s2)
        if s == s2 and action is not None:
            dx, dy = DIRECTIONS[action]
            return x1 + 0.32 * dx, y1 + 0.32 * dy
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _cell(self, s):
        return s % self.cols, s // self.cols

    # ----------------------------------------------------------------------- drawing
    def _px(self, x, y):
        """Tile coordinates (centre of tile = integer) to screen pixels."""
        return self.grid_x + (x + 0.5) * self.tile, self.grid_y + (y + 0.5) * self.tile

    def draw(self):
        sc = self.screen
        sc.fill(NIGHT)
        for x, y, r, _v, _p in self.flakes:
            pygame.draw.circle(sc, (200, 215, 235) if r > 1.8 else (120, 140, 170), (x, y), r)
        for s, t in enumerate(self.tiles):
            self._draw_tile(s, t)
        if self.method == "Value Iteration" or self.Q.any():
            for s, t in enumerate(self.tiles):
                if t not in "GH" and np.ptp(self.Q[s]) > 1e-12:
                    self._draw_arrow(s, int(np.argmax(self.Q[s])))
        for x, y, explored in self.footprints:
            self._draw_footprint(x, y, RANDOM_STEP if explored else GREEDY_STEP)
        self._draw_agent()
        if self.result_timer > 0:
            self._draw_banner(self.result)
        if self.paused:
            self._draw_banner("Paused")
        self._draw_text()
        pygame.display.flip()

    def _draw_tile(self, s, t):
        x, y = self._cell(s)
        T = self.tile
        rect = pygame.Rect(self.grid_x + x * T + 2, self.grid_y + y * T + 2, T - 4, T - 4)
        if t == "H":
            pygame.draw.rect(self.screen, WATER, rect, border_radius=T // 8)
            cx, cy = rect.center
            for k in range(3):   # ripples
                w = T * (0.25 + 0.17 * k) + 4 * math.sin(self.time * 1.5 + k)
                pygame.draw.ellipse(self.screen, (40, 74, 122), (cx - w / 2, cy - w / 5, w, w / 2.5), 1)
            for a, b in [((0.05, 0.2), (0.22, 0.32)), ((0.95, 0.75), (0.75, 0.68)), ((0.6, 0.04), (0.55, 0.2))]:
                pygame.draw.line(self.screen, (170, 200, 230), (rect.x + a[0] * T, rect.y + a[1] * T),
                                 (rect.x + b[0] * T, rect.y + b[1] * T), 2)   # cracks at the edge
            return
        pygame.draw.rect(self.screen, value_colour(self.shown[s]), rect, border_radius=T // 8)
        # Snow texture: a faint highlight streak and a few specks (fixed per tile).
        hl = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.line(hl, (255, 255, 255, 70), (T * 0.12, T * 0.3), (T * 0.3, T * 0.12), 3)
        rnd = random.Random(s)
        for _ in range(5):
            pygame.draw.circle(hl, (255, 255, 255, 90), (rnd.uniform(6, T - 10), rnd.uniform(6, T - 10)), 1.5)
        self.screen.blit(hl, rect.topleft)
        cx, cy = rect.center
        if t == "S":   # snowy mound
            pygame.draw.ellipse(self.screen, (150, 175, 200), (cx - T * 0.34, cy + T * 0.16, T * 0.68, T * 0.24))
            pygame.draw.ellipse(self.screen, (250, 252, 255), (cx - T * 0.32, cy + T * 0.12, T * 0.64, T * 0.24))
        elif t == "G":  # flag on a mound
            pygame.draw.ellipse(self.screen, (250, 252, 255), (cx - T * 0.25, cy + T * 0.22, T * 0.5, T * 0.16))
            pole_x, top = cx - T * 0.05, cy - T * 0.36
            pygame.draw.line(self.screen, (70, 60, 55), (pole_x, cy + T * 0.28), (pole_x, top), 3)
            wave = 3 * math.sin(self.time * 4)
            pygame.draw.polygon(self.screen, (220, 50, 60), [(pole_x, top), (pole_x + T * 0.3, top + T * 0.08 + wave),
                                                             (pole_x, top + T * 0.18)])

    def _draw_arrow(self, s, a):
        cx, cy = self._px(*self._cell(s))
        dx, dy = DIRECTIONS[a]
        L = self.tile * 0.2
        tip = (cx + dx * L, cy + dy * L)
        tail = (cx - dx * L, cy - dy * L)
        side = (-dy * L * 0.45, dx * L * 0.45)
        colour = (40, 60, 95)
        pygame.draw.line(self.screen, colour, tail, (cx + dx * L * 0.4, cy + dy * L * 0.4), 3)
        pygame.draw.polygon(self.screen, colour, [tip, (cx + dx * L * 0.2 + side[0], cy + dy * L * 0.2 + side[1]),
                                                  (cx + dx * L * 0.2 - side[0], cy + dy * L * 0.2 - side[1])])

    def _draw_footprint(self, x, y, colour):
        px, py = self._px(x, y)
        r = max(3, self.tile // 22)
        pygame.draw.ellipse(self.screen, colour, (px - 2.6 * r, py - r * 1.4, r * 1.8, r * 2.6))
        pygame.draw.ellipse(self.screen, colour, (px + 0.8 * r, py - r * 1.0, r * 1.8, r * 2.6))

    def _draw_agent(self):
        if self.move:
            (x1, y1), (x2, y2), t = self._cell(self.move[0]), self._cell(self.move[1]), self.move[2]
            x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
            hop = -abs(math.sin(t * math.pi)) * 0.12
        else:
            x, y = self._cell(self.agent)
            hop = 0.0
        px, py = self._px(x, y + hop)
        T = self.tile
        if self.tiles[self.agent] == "H" and not self.move:   # splash rings instead of a penguin
            for k in range(3):
                r = T * (0.12 + 0.1 * k + 0.05 * math.sin(self.time * 6 + k))
                pygame.draw.circle(self.screen, (210, 230, 250), (px, py), r, 2)
            return
        if self.tiles[self.agent] == "G" and not self.move:
            py -= abs(math.sin(self.time * 6)) * T * 0.08    # happy bounce
        u = T * 0.13   # penguin size unit
        pygame.draw.ellipse(self.screen, (255, 150, 40), (px - 1.1 * u, py + 1.6 * u, u, 0.5 * u))   # feet
        pygame.draw.ellipse(self.screen, (255, 150, 40), (px + 0.1 * u, py + 1.6 * u, u, 0.5 * u))
        pygame.draw.ellipse(self.screen, (25, 28, 38), (px - 1.25 * u, py - 1.4 * u, 2.5 * u, 3.2 * u))  # body
        pygame.draw.ellipse(self.screen, (245, 245, 250), (px - 0.8 * u, py - 0.6 * u, 1.6 * u, 2.3 * u))  # belly
        pygame.draw.circle(self.screen, (25, 28, 38), (px, py - 1.4 * u), 0.95 * u)                       # head
        for ex in (-0.35, 0.35):
            pygame.draw.circle(self.screen, (255, 255, 255), (px + ex * u, py - 1.55 * u), 0.22 * u)
            pygame.draw.circle(self.screen, (0, 0, 0), (px + ex * u, py - 1.55 * u), 0.1 * u)
        pygame.draw.polygon(self.screen, (255, 150, 40), [(px - 0.25 * u, py - 1.2 * u), (px + 0.25 * u, py - 1.2 * u),
                                                          (px, py - 0.85 * u)])                          # beak
        pygame.draw.rect(self.screen, (200, 40, 50), (px - 0.9 * u, py - 0.75 * u, 1.8 * u, 0.3 * u),
                         border_radius=3)                                                               # scarf

    def _draw_banner(self, text):
        label = self.big.render(text, True, TEXT)
        box = label.get_rect(center=(self.width // 2, self.grid_y + self.rows * self.tile // 2)).inflate(36, 20)
        shade = pygame.Surface(box.size, pygame.SRCALPHA)
        pygame.draw.rect(shade, (8, 13, 28, 200), shade.get_rect(), border_radius=12)
        self.screen.blit(shade, box.topleft)
        self.screen.blit(label, label.get_rect(center=box.center))

    def _draw_text(self):
        controls = "1 Value Iteration   2 Q-learning   3 SARSA   Space pause   Up/Down speed   S skip 50   R restart   Esc quit"
        self.screen.blit(self.small.render(controls, True, TEXT_DIM), (12, 14))
        bar_y = self.height - 58
        pygame.draw.rect(self.screen, BAR, (0, bar_y, self.width, 58))
        if self.method == "Value Iteration":
            n = len(self.vi_q)
            progress = f"sweep {min(self.sweep, n)} / {n}" + ("  (done, walking the plan)" if self.sweep >= n else "")
            chance = "chance of random move: none (planning)"
        else:
            progress = f"episode {self.episode:,} / {self.runner.n_episodes:,}" + ("  (finished)" if self.finished else "")
            chance = f"chance of random move: {self.epsilon:.2f}" if self.epsilon is not None else "chance of random move: -"
        status = f"{self.method}   ·   {progress}   ·   steps: {self.steps}   ·   {chance}"
        self.screen.blit(self.font.render(status, True, TEXT), (12, bar_y + 8))
        x = 12
        for colour, label in [(GREEDY_STEP, "best-looking move"), (RANDOM_STEP, "random move"),
                              (WARM, "looks good"), (COLD, "looks bad")]:
            pygame.draw.circle(self.screen, colour, (x + 6, bar_y + 41), 6)
            text = self.small.render(label, True, TEXT_DIM)
            self.screen.blit(text, (x + 18, bar_y + 33))
            x += 30 + text.get_width()
        right = self.small.render(f"last: {self.last_result}   ·   {self.speed} moves/s", True, TEXT)
        self.screen.blit(right, (self.width - right.get_width() - 12, bar_y + 33))

    # ----------------------------------------------------------------------- loop
    def handle(self, event):
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in METHOD_KEYS:
                self.start(METHOD_KEYS[event.key])
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_UP:
                self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
            elif event.key == pygame.K_DOWN:
                self.speed_i = max(self.speed_i - 1, 0)
            elif event.key == pygame.K_s:
                self.skip()
            elif event.key == pygame.K_r:
                self.start(self.method)

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000
            for event in pygame.event.get():
                self.handle(event)
            self.update(dt)
            self.draw()
        if self.runner:
            self.runner.close()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", default="4x4", choices=list(CONFIG["maps"]))
    parser.add_argument("--seed", type=int, default=0, help="training seed for Q-learning / SARSA")
    args = parser.parse_args()
    App(args.map, args.seed).run()
