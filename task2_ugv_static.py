"""
Task 2: UGV Navigation — Static Obstacles (A* Algorithm)
=========================================================
Simulates a 70×70 km battlefield grid.
Each cell = 1 km².  UGV moves in 8 directions (including diagonals).
Obstacles are generated randomly at three density levels.

Measures of Effectiveness (MoE):
  • Path length (km)
  • Nodes explored
  • Computation time (ms)
  • Path optimality ratio
  • Obstacle density (%)
"""

import heapq
import random
import time
import math
import os


# ──────────────────────────────────────────────
# Grid configuration
# ──────────────────────────────────────────────
GRID_ROWS = 70
GRID_COLS = 70

OBSTACLE_DENSITIES = {
    "low":    0.10,   # 10%
    "medium": 0.25,   # 25%
    "high":   0.40,   # 40%
}

# 8-directional movement costs
MOVES = [
    (-1,  0, 1.0),   # N
    ( 1,  0, 1.0),   # S
    ( 0, -1, 1.0),   # W
    ( 0,  1, 1.0),   # E
    (-1, -1, 1.414), # NW
    (-1,  1, 1.414), # NE
    ( 1, -1, 1.414), # SW
    ( 1,  1, 1.414), # SE
]


# ──────────────────────────────────────────────
# Grid generation
# ──────────────────────────────────────────────
def generate_grid(rows: int, cols: int, density: float,
                  start: tuple, goal: tuple,
                  seed: int = 42) -> list:
    """Returns a 2D grid: 0 = free, 1 = obstacle."""
    random.seed(seed)
    grid = [[0] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in (start, goal) and random.random() < density:
                grid[r][c] = 1

    return grid


# ──────────────────────────────────────────────
# Heuristic
# ──────────────────────────────────────────────
def heuristic(a: tuple, b: tuple) -> float:
    """Octile distance — admissible for 8-directional grids."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


# ──────────────────────────────────────────────
# A* Search
# ──────────────────────────────────────────────
def astar(grid: list, start: tuple, goal: tuple) -> tuple:
    """
    A* search on a 2D grid.

    Returns
    -------
    path          : list of (row, col) from start to goal, or []
    nodes_explored: int
    """
    rows = len(grid)
    cols = len(grid[0])

    open_set = []           # min-heap: (f, g, node)
    heapq.heappush(open_set, (heuristic(start, goal), 0.0, start))

    g_score = {start: 0.0}
    came_from = {start: None}
    closed_set = set()
    nodes_explored = 0

    while open_set:
        f, g, current = heapq.heappop(open_set)

        if current in closed_set:
            continue
        closed_set.add(current)
        nodes_explored += 1

        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, nodes_explored

        r, c = current
        for dr, dc, cost in MOVES:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:
                continue
            if neighbor in closed_set:
                continue

            tentative_g = g + cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_new = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_new, tentative_g, neighbor))

    return [], nodes_explored   # no path found


# ──────────────────────────────────────────────
# Visualiser
# ──────────────────────────────────────────────
def visualize(grid: list, path: list, start: tuple, goal: tuple,
              max_display: int = 40):
    """Prints an ASCII map (truncated to max_display×max_display for readability)."""
    rows = min(len(grid), max_display)
    cols = min(len(grid[0]), max_display)

    path_set = set(path)

    print("\n  Legend:  S=Start  G=Goal  *=Path  █=Obstacle  .=Free\n")
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            if (r, c) == start:
                row_str += "S "
            elif (r, c) == goal:
                row_str += "G "
            elif (r, c) in path_set:
                row_str += "* "
            elif grid[r][c] == 1:
                row_str += "█ "
            else:
                row_str += ". "
        print("  " + row_str)

    if len(grid) > max_display or len(grid[0]) > max_display:
        print(f"\n  (Map truncated to {max_display}×{max_display} for display; full grid is {GRID_ROWS}×{GRID_COLS})")


# ──────────────────────────────────────────────
# MoE report
# ──────────────────────────────────────────────
def path_length(path: list) -> float:
    total = 0.0
    for i in range(1, len(path)):
        r1, c1 = path[i - 1]
        r2, c2 = path[i]
        total += math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)
    return round(total, 3)


def euclidean(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def print_moe(density_label: str, density: float, path: list,
              nodes: int, elapsed_ms: float,
              start: tuple, goal: tuple):
    print(f"\n{'═'*55}")
    print(f"  MoE Report — Obstacle Density: {density_label.upper()} ({density*100:.0f}%)")
    print(f"{'═'*55}")
    if path:
        plen   = path_length(path)
        direct = euclidean(start, goal)
        ratio  = round(plen / direct, 4) if direct > 0 else 1.0
        print(f"  Path found          : Yes")
        print(f"  Path length         : {plen:.3f} km")
        print(f"  Direct distance     : {direct:.3f} km")
        print(f"  Optimality ratio    : {ratio:.4f}  (1.0 = straight line)")
        print(f"  Steps in path       : {len(path)}")
    else:
        print(f"  Path found          : NO (goal unreachable)")

    print(f"  Nodes explored      : {nodes:,}")
    print(f"  Computation time    : {elapsed_ms:.3f} ms")
    obstacle_count = sum(grid[r][c] for r in range(GRID_ROWS) for c in range(GRID_COLS))
    print(f"  Obstacles on grid   : {obstacle_count:,} / {GRID_ROWS*GRID_COLS:,} cells")
    print(f"{'═'*55}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    START = (5, 5)
    GOAL  = (64, 64)

    print("\n" + "="*55)
    print("  UGV Navigation — Static Obstacles (A* Algorithm)")
    print(f"  Grid: {GRID_ROWS}×{GRID_COLS} km  |  Start: {START}  |  Goal: {GOAL}")
    print("="*55)

    for label, density in OBSTACLE_DENSITIES.items():
        grid = generate_grid(GRID_ROWS, GRID_COLS, density, START, GOAL, seed=42)

        t0 = time.perf_counter()
        path, nodes = astar(grid, START, GOAL)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print_moe(label, density, path, nodes, elapsed_ms, START, GOAL)

        if path:
            visualize(grid, path, START, GOAL, max_display=35)
        else:
            print("\n  ⚠️  No path found — try a different seed or lower density.\n")

        print()
