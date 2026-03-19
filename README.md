# AI Search Algorithms — Programming Assignment

Three standalone Python programs implementing classic and advanced AI search algorithms.  
**No external libraries required** — runs on pure Python 3.8+.

---

## Files

| File | Algorithm | Problem |
|------|-----------|---------|
| `task1_dijkstra_india.py` | Dijkstra's Algorithm | Shortest road distance between Indian cities |
| `task2_ugv_static.py` | A\* Search | UGV navigation on 70×70 grid with static obstacles |
| `task3_ugv_dynamic.py` | D\* Lite | UGV navigation with dynamic (unknown a-priori) obstacles |

---

## How to Run

```bash
# Task 1 — Dijkstra on Indian road network
python task1_dijkstra_india.py

# Task 2 — A* with static obstacles (3 density levels)
python task2_ugv_static.py

# Task 3 — D* Lite with dynamic obstacles
python task3_ugv_dynamic.py
```

---

## Task Descriptions

### Task 1 — Dijkstra's Algorithm (Indian Cities)
- Implements Dijkstra's uniform-cost search using a **min-heap priority queue**
- Graph covers **50+ major Indian cities** with approximate road distances (km)
- Finds shortest path from a source city to **all other cities**
- Includes point-to-point queries: Delhi→Chennai, Delhi→Thiruvananthapuram, etc.

### Task 2 — UGV Static Obstacle Navigation (A*)
- 70×70 km battlefield grid, each cell = 1 km²
- **8-directional movement** (N, S, E, W + diagonals)
- Obstacles generated randomly at **3 density levels**: Low (10%), Medium (25%), High (40%)
- Uses **A\*** with octile distance heuristic (admissible for 8-dir grids)
- ASCII map visualization + full **Measures of Effectiveness (MoE)** report

**MoE Outputs:**
- Path length (km)
- Direct Euclidean distance
- Optimality ratio (path/direct)
- Nodes explored
- Computation time (ms)

### Task 3 — UGV Dynamic Obstacle Navigation (D* Lite)
- Extends Task 2: obstacles are **not fully known a-priori**
- UGV has a **sensor range** (default 7 km) — only sees nearby obstacles
- **Dynamic obstacles** appear mid-mission (enemy movement / debris)
- **D\* Lite** (incremental replanning) efficiently updates the path without restarting
- Tracks replans triggered and nodes re-expanded per replan

**MoE Outputs (additional to Task 2):**
- Number of replans triggered
- Nodes re-expanded during replanning
- Total vs direct distance comparison

---

## Algorithm Summary

| Algorithm | Time Complexity | Space | Best For |
|-----------|----------------|-------|----------|
| Dijkstra  | O((V+E) log V) | O(V)  | Weighted graphs, no heuristic |
| A\*       | O(b^d)         | O(b^d)| Static grids with good heuristic |
| D\* Lite  | O(k log n) per replan | O(n) | Dynamic environments |

*V = vertices, E = edges, b = branching factor, d = depth, k = changed edges, n = nodes*

---

## Configuration (editable at top of each file)

**Task 2 & 3:**
```python
GRID_ROWS = 70          # grid height (km)
GRID_COLS = 70          # grid width  (km)
START     = (5, 5)      # UGV start position
GOAL      = (64, 64)    # mission objective
```

**Task 3 additional:**
```python
SENSOR_RANGE    = 7     # km UGV can see around itself
DYNAMIC_EVENTS  = 30    # obstacles appearing mid-mission
INITIAL_DENSITY = 0.15  # fraction of initially known obstacles
```
