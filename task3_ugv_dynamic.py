"""
Task 3: UGV Navigation — Dynamic Obstacles (D* Lite Algorithm)
==============================================================
Extends Task 2: obstacles are NOT fully known a-priori.
The UGV has a limited sensor range and discovers obstacles as it moves.
When a new obstacle blocks the planned path, D* Lite REPLANS efficiently
without restarting from scratch.

Dynamic events simulated:
  • Sensor range: UGV sees obstacles within R cells of its current position.
  • New obstacles appear randomly mid-mission (simulating enemy movement / debris).

Measures of Effectiveness (MoE):
  • Total path length (km)
  • Number of replans triggered
  • Nodes re-expanded during replanning
  • Computation time (ms)
  • Percentage of path that changed due to dynamic obstacles
"""

import heapq
import random
import time
import math
from collections import defaultdict


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
GRID_ROWS   = 70
GRID_COLS   = 70
SENSOR_RANGE = 7          # cells the UGV can sense around itself
INITIAL_DENSITY = 0.15    # fraction of initially known obstacles
DYNAMIC_EVENTS  = 30      # extra obstacles appearing during mission
DYNAMIC_SEED    = 99

MOVES = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, 1.414),
    (-1,  1, 1.414),
    ( 1, -1, 1.414),
    ( 1,  1, 1.414),
]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def heuristic(a, b):
    dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
    return max(dr, dc) + (math.sqrt(2)-1)*min(dr, dc)

def neighbors(node, rows, cols):
    r, c = node
    for dr, dc, cost in MOVES:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc), cost

def path_length(path):
    total = 0.0
    for i in range(1, len(path)):
        r1,c1 = path[i-1]; r2,c2 = path[i]
        total += math.sqrt((r2-r1)**2+(c2-c1)**2)
    return round(total, 3)

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)


# ──────────────────────────────────────────────
# D* Lite
# ──────────────────────────────────────────────
class DStarLite:
    """
    D* Lite incremental replanning algorithm.
    Works backwards from goal → start so replanning is local.
    """
    def __init__(self, rows, cols, start, goal, known_obstacles):
        self.rows = rows
        self.cols  = cols
        self.start = start
        self.goal  = goal
        self.obstacles = set(known_obstacles)

        self.g   = defaultdict(lambda: float("inf"))
        self.rhs = defaultdict(lambda: float("inf"))
        self.km  = 0.0
        self.U   = []           # priority queue: (key, node)
        self._entry_finder = {}
        self.counter = 0

        self.rhs[goal] = 0.0
        self._push(goal, self._calc_key(goal))

    # ── priority queue helpers ──────────────────
    def _push(self, node, key):
        self.counter += 1
        entry = [key, self.counter, node]
        self._entry_finder[node] = entry
        heapq.heappush(self.U, entry)

    def _pop(self):
        while self.U:
            key, cnt, node = heapq.heappop(self.U)
            if node in self._entry_finder and self._entry_finder[node][1] == cnt:
                del self._entry_finder[node]
                return key, node
        return None, None

    def _top_key(self):
        while self.U:
            key, cnt, node = self.U[0]
            if node in self._entry_finder and self._entry_finder[node][1] == cnt:
                return key
            heapq.heappop(self.U)
        return (float("inf"), float("inf"))

    # ── D* Lite core ────────────────────────────
    def _calc_key(self, node):
        g_rhs = min(self.g[node], self.rhs[node])
        return (g_rhs + heuristic(self.start, node) + self.km,
                g_rhs)

    def _update_vertex(self, u):
        if u != self.goal:
            best = float("inf")
            for v, cost in neighbors(u, self.rows, self.cols):
                if v not in self.obstacles:
                    best = min(best, cost + self.g[v])
            self.rhs[u] = best

        if u in self._entry_finder:
            del self._entry_finder[u]

        if self.g[u] != self.rhs[u]:
            self._push(u, self._calc_key(u))

    def compute_shortest_path(self):
        nodes_expanded = 0
        while True:
            top_key = self._top_key()
            s_key   = self._calc_key(self.start)
            if top_key >= s_key and self.rhs[self.start] == self.g[self.start]:
                break

            k_old, u = self._pop()
            if u is None:
                break
            nodes_expanded += 1

            k_new = self._calc_key(u)
            if k_old < k_new:
                self._push(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for v, _ in neighbors(u, self.rows, self.cols):
                    self._update_vertex(v)
            else:
                self.g[u] = float("inf")
                self._update_vertex(u)
                for v, _ in neighbors(u, self.rows, self.cols):
                    self._update_vertex(v)

        return nodes_expanded

    def extract_path(self):
        """Greedily follows minimum g values from start to goal."""
        path = [self.start]
        current = self.start
        visited = {current}
        for _ in range(self.rows * self.cols):
            if current == self.goal:
                break
            best_cost = float("inf")
            best_next = None
            for v, cost in neighbors(current, self.rows, self.cols):
                if v not in self.obstacles and v not in visited:
                    if cost + self.g[v] < best_cost:
                        best_cost = cost + self.g[v]
                        best_next = v
            if best_next is None:
                return []    # stuck
            path.append(best_next)
            visited.add(best_next)
            current = best_next
        return path if current == self.goal else []

    def add_obstacle(self, node):
        """Notify D* Lite of a newly discovered obstacle."""
        if node in self.obstacles:
            return
        self.obstacles.add(node)
        # Edges to/from node are now blocked — update all predecessors
        self._update_vertex(node)
        for v, _ in neighbors(node, self.rows, self.cols):
            self._update_vertex(v)

    def move_start(self, new_start):
        """UGV moves one step; update km heuristic bias."""
        self.km += heuristic(self.start, new_start)
        self.start = new_start


# ──────────────────────────────────────────────
# Mission simulator
# ──────────────────────────────────────────────
def run_mission(start, goal, rows=GRID_ROWS, cols=GRID_COLS,
                initial_density=INITIAL_DENSITY,
                dynamic_events=DYNAMIC_EVENTS,
                sensor_range=SENSOR_RANGE,
                seed=DYNAMIC_SEED):

    rng = random.Random(seed)

    # ── Build initially known obstacles ──────────
    known_obstacles = set()
    for r in range(rows):
        for c in range(cols):
            if (r,c) not in (start,goal) and rng.random() < initial_density:
                known_obstacles.add((r,c))

    # ── Hidden dynamic obstacles (appear mid-mission) ──
    # Placed far from start so they're not visible initially
    hidden_obstacles = []
    attempts = 0
    while len(hidden_obstacles) < dynamic_events and attempts < 5000:
        attempts += 1
        r = rng.randint(0, rows-1)
        c = rng.randint(0, cols-1)
        candidate = (r, c)
        if (candidate not in known_obstacles and
                candidate not in (start, goal) and
                euclidean(candidate, start) > sensor_range * 2):
            hidden_obstacles.append(candidate)

    t0 = time.perf_counter()

    # ── Initialise D* Lite ──────────────────────
    planner = DStarLite(rows, cols, start, goal, known_obstacles)
    total_nodes_expanded = planner.compute_shortest_path()

    current = start
    full_trajectory = [current]
    total_replans   = 0
    nodes_replanned = 0
    steps_changed   = 0

    hidden_pool = hidden_obstacles[:]

    for step in range(rows * cols):
        if current == goal:
            break

        # Sense obstacles within sensor_range
        newly_found = []
        for obs in list(hidden_pool):
            if euclidean(current, obs) <= sensor_range:
                newly_found.append(obs)
                hidden_pool.remove(obs)

        # If new obstacles found → update planner and replan
        if newly_found:
            for obs in newly_found:
                planner.add_obstacle(obs)
            planner.km += heuristic(planner.start, current)
            planner.start = current
            extra = planner.compute_shortest_path()
            nodes_replanned += extra
            total_replans   += 1
            total_nodes_expanded += extra

        # Extract current best path from current position
        path_from_here = planner.extract_path()
        if not path_from_here or len(path_from_here) < 2:
            print("  ⚠️  Path blocked — no route to goal!")
            break

        # Move one step along path
        next_step = path_from_here[1]
        planner.move_start(next_step)
        planner.start = next_step
        current = next_step
        full_trajectory.append(current)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return full_trajectory, total_replans, nodes_replanned, total_nodes_expanded, elapsed_ms, planner.obstacles


# ──────────────────────────────────────────────
# Visualiser
# ──────────────────────────────────────────────
def visualize_dynamic(trajectory, obstacles, start, goal, max_display=35):
    rows = min(GRID_ROWS, max_display)
    cols = min(GRID_COLS, max_display)
    traj_set = set(trajectory)
    print("\n  Legend:  S=Start  G=Goal  *=Path  █=Obstacle  .=Free\n")
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            if (r,c) == start:       row_str += "S "
            elif (r,c) == goal:      row_str += "G "
            elif (r,c) in traj_set:  row_str += "* "
            elif (r,c) in obstacles: row_str += "█ "
            else:                    row_str += ". "
        print("  " + row_str)
    if GRID_ROWS > max_display:
        print(f"\n  (Truncated to {max_display}×{max_display}; full grid {GRID_ROWS}×{GRID_COLS})")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    START = (5, 5)
    GOAL  = (64, 64)

    print("\n" + "="*60)
    print("  UGV Navigation — Dynamic Obstacles (D* Lite)")
    print(f"  Grid: {GRID_ROWS}×{GRID_COLS} km")
    print(f"  Start: {START}  →  Goal: {GOAL}")
    print(f"  Sensor range: {SENSOR_RANGE} km")
    print(f"  Initial obstacle density: {INITIAL_DENSITY*100:.0f}%")
    print(f"  Dynamic obstacles injected mid-mission: {DYNAMIC_EVENTS}")
    print("="*60)

    trajectory, replans, nodes_replanned, total_nodes, elapsed_ms, all_obstacles = run_mission(
        START, GOAL
    )

    reached = trajectory[-1] == GOAL
    direct  = euclidean(START, GOAL)
    plen    = path_length(trajectory)
    ratio   = round(plen / direct, 4) if direct > 0 else 1.0

    print(f"\n{'═'*55}")
    print(f"  MoE Report — Dynamic Environment")
    print(f"{'═'*55}")
    print(f"  Goal reached            : {'Yes ✓' if reached else 'No ✗'}")
    print(f"  Total trajectory length : {plen:.3f} km")
    print(f"  Direct (Euclidean) dist : {direct:.3f} km")
    print(f"  Optimality ratio        : {ratio:.4f}")
    print(f"  Total steps taken       : {len(trajectory)}")
    print(f"  Replans triggered       : {replans}")
    print(f"  Nodes re-expanded       : {nodes_replanned:,}")
    print(f"  Total nodes expanded    : {total_nodes:,}")
    print(f"  Computation time        : {elapsed_ms:.3f} ms")
    print(f"  Final obstacle count    : {len(all_obstacles):,}")
    print(f"{'═'*55}")

    visualize_dynamic(trajectory, all_obstacles, START, GOAL)
