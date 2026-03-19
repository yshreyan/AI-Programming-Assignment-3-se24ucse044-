"""
Task 1: Dijkstra's Algorithm - Shortest Path Between Indian Cities
===================================================================
Finds the shortest road distance from a source city to ALL other cities in India.
Road distances are approximate real-world values (in km) from open sources.
"""

import heapq

# ──────────────────────────────────────────────
# Road distance graph (undirected, km approx.)
# ──────────────────────────────────────────────
INDIA_ROADS = {
    "Delhi": [("Jaipur", 280), ("Agra", 233), ("Chandigarh", 250), ("Lucknow", 555), ("Amritsar", 450)],
    "Jaipur": [("Delhi", 280), ("Agra", 240), ("Ahmedabad", 670), ("Jodhpur", 345), ("Kota", 245)],
    "Agra": [("Delhi", 233), ("Jaipur", 240), ("Lucknow", 363), ("Gwalior", 118), ("Kanpur", 295)],
    "Chandigarh": [("Delhi", 250), ("Amritsar", 220), ("Shimla", 115), ("Dehradun", 170)],
    "Amritsar": [("Chandigarh", 220), ("Delhi", 450), ("Jammu", 210)],
    "Jammu": [("Amritsar", 210), ("Chandigarh", 310)],
    "Shimla": [("Chandigarh", 115), ("Dehradun", 235)],
    "Dehradun": [("Chandigarh", 170), ("Shimla", 235), ("Lucknow", 500), ("Haridwar", 55)],
    "Haridwar": [("Dehradun", 55), ("Lucknow", 440)],
    "Lucknow": [("Delhi", 555), ("Agra", 363), ("Kanpur", 90), ("Varanasi", 320), ("Dehradun", 500), ("Haridwar", 440)],
    "Kanpur": [("Lucknow", 90), ("Agra", 295), ("Allahabad", 193)],
    "Allahabad": [("Kanpur", 193), ("Varanasi", 121), ("Patna", 595)],
    "Varanasi": [("Lucknow", 320), ("Allahabad", 121), ("Patna", 300), ("Kolkata", 670)],
    "Patna": [("Varanasi", 300), ("Allahabad", 595), ("Kolkata", 600), ("Ranchi", 330), ("Gaya", 100)],
    "Gaya": [("Patna", 100), ("Ranchi", 175), ("Kolkata", 490)],
    "Ranchi": [("Patna", 330), ("Gaya", 175), ("Kolkata", 400), ("Bhubaneswar", 450)],
    "Kolkata": [("Patna", 600), ("Varanasi", 670), ("Ranchi", 400), ("Bhubaneswar", 480), ("Guwahati", 1000)],
    "Bhubaneswar": [("Kolkata", 480), ("Ranchi", 450), ("Visakhapatnam", 450), ("Raipur", 440)],
    "Visakhapatnam": [("Bhubaneswar", 450), ("Hyderabad", 580), ("Vijayawada", 350)],
    "Vijayawada": [("Visakhapatnam", 350), ("Hyderabad", 275), ("Chennai", 430), ("Tirupati", 400)],
    "Hyderabad": [("Visakhapatnam", 580), ("Vijayawada", 275), ("Nagpur", 500), ("Bengaluru", 575), ("Chennai", 630)],
    "Nagpur": [("Hyderabad", 500), ("Raipur", 295), ("Bhopal", 350), ("Mumbai", 840), ("Aurangabad", 540)],
    "Raipur": [("Nagpur", 295), ("Bhubaneswar", 440), ("Ranchi", 390)],
    "Bhopal": [("Nagpur", 350), ("Indore", 195), ("Gwalior", 430), ("Delhi", 775)],
    "Gwalior": [("Agra", 118), ("Bhopal", 430), ("Jhansi", 100)],
    "Jhansi": [("Gwalior", 100), ("Kanpur", 200), ("Bhopal", 345)],
    "Indore": [("Bhopal", 195), ("Ahmedabad", 390), ("Mumbai", 590), ("Ujjain", 55)],
    "Ujjain": [("Indore", 55), ("Bhopal", 185)],
    "Ahmedabad": [("Jaipur", 670), ("Indore", 390), ("Mumbai", 530), ("Surat", 265), ("Jodhpur", 490)],
    "Jodhpur": [("Jaipur", 345), ("Ahmedabad", 490), ("Udaipur", 250)],
    "Udaipur": [("Jodhpur", 250), ("Ahmedabad", 265), ("Kota", 250)],
    "Kota": [("Jaipur", 245), ("Udaipur", 250), ("Bhopal", 390)],
    "Surat": [("Ahmedabad", 265), ("Mumbai", 290)],
    "Mumbai": [("Surat", 290), ("Ahmedabad", 530), ("Pune", 150), ("Nagpur", 840), ("Goa", 590), ("Aurangabad", 340)],
    "Aurangabad": [("Mumbai", 340), ("Nagpur", 540), ("Pune", 235)],
    "Pune": [("Mumbai", 150), ("Aurangabad", 235), ("Goa", 450), ("Bengaluru", 840), ("Hyderabad", 560)],
    "Goa": [("Mumbai", 590), ("Pune", 450), ("Bengaluru", 560), ("Mangalore", 385)],
    "Mangalore": [("Goa", 385), ("Bengaluru", 360), ("Kozhikode", 225)],
    "Bengaluru": [("Hyderabad", 575), ("Chennai", 350), ("Mysuru", 150), ("Coimbatore", 365), ("Mangalore", 360), ("Goa", 560), ("Pune", 840)],
    "Mysuru": [("Bengaluru", 150), ("Coimbatore", 215), ("Ooty", 120)],
    "Ooty": [("Mysuru", 120), ("Coimbatore", 90)],
    "Coimbatore": [("Bengaluru", 365), ("Mysuru", 215), ("Ooty", 90), ("Chennai", 500), ("Kozhikode", 165), ("Madurai", 210)],
    "Kozhikode": [("Coimbatore", 165), ("Mangalore", 225), ("Kochi", 220), ("Thrissur", 150)],
    "Kochi": [("Kozhikode", 220), ("Thrissur", 80), ("Thiruvananthapuram", 225)],
    "Thrissur": [("Kochi", 80), ("Kozhikode", 150)],
    "Thiruvananthapuram": [("Kochi", 225), ("Madurai", 250)],
    "Madurai": [("Thiruvananthapuram", 250), ("Coimbatore", 210), ("Chennai", 460), ("Tirunelveli", 160)],
    "Tirunelveli": [("Madurai", 160), ("Thiruvananthapuram", 150)],
    "Chennai": [("Bengaluru", 350), ("Vijayawada", 430), ("Hyderabad", 630), ("Tirupati", 150), ("Madurai", 460), ("Coimbatore", 500)],
    "Tirupati": [("Chennai", 150), ("Vijayawada", 400)],
    "Guwahati": [("Kolkata", 1000), ("Shillong", 100), ("Silchar", 400)],
    "Shillong": [("Guwahati", 100)],
    "Silchar": [("Guwahati", 400)],
}


def dijkstra(graph: dict, source: str) -> tuple[dict, dict]:
    """
    Dijkstra's algorithm using a min-heap (priority queue).

    Returns
    -------
    dist  : dict  – shortest distance from source to every reachable city
    prev  : dict  – previous-node map for path reconstruction
    """
    dist = {city: float("inf") for city in graph}
    prev = {city: None for city in graph}
    dist[source] = 0

    # heap entries: (distance, city)
    heap = [(0, source)]

    visited = set()

    while heap:
        d, u = heapq.heappop(heap)

        if u in visited:
            continue
        visited.add(u)

        for v, weight in graph.get(u, []):
            new_dist = d + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, prev


def reconstruct_path(prev: dict, source: str, destination: str) -> list:
    """Traces back from destination to source using the prev map."""
    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    if path[0] != source:
        return []          # destination unreachable
    return path


def print_results(source: str, dist: dict, prev: dict):
    print(f"\n{'═'*65}")
    print(f"  Dijkstra's Algorithm — Source: {source}")
    print(f"{'═'*65}")
    print(f"{'Destination':<25} {'Distance (km)':>15}  Path")
    print(f"{'─'*65}")

    reachable = [(d, city) for city, d in dist.items() if d < float("inf") and city != source]
    reachable.sort()

    for d, city in reachable:
        path = reconstruct_path(prev, source, city)
        path_str = " → ".join(path)
        print(f"{city:<25} {d:>15,}  {path_str}")

    unreachable = [c for c, d in dist.items() if d == float("inf")]
    if unreachable:
        print(f"\n  Unreachable cities: {', '.join(unreachable)}")
    print(f"{'═'*65}\n")


def single_query(source: str, destination: str, dist: dict, prev: dict):
    """Print shortest path between two specific cities."""
    if dist[destination] == float("inf"):
        print(f"\nNo path found from {source} to {destination}.")
        return
    path = reconstruct_path(prev, source, destination)
    print(f"\n  Shortest path : {' → '.join(path)}")
    print(f"  Total distance: {dist[destination]:,} km\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    SOURCE = "Delhi"

    if SOURCE not in INDIA_ROADS:
        print(f"City '{SOURCE}' not found in the graph.")
    else:
        dist, prev = dijkstra(INDIA_ROADS, SOURCE)
        print_results(SOURCE, dist, prev)

        # --- Example point-to-point queries ---
        queries = [
            ("Delhi", "Chennai"),
            ("Delhi", "Thiruvananthapuram"),
            ("Delhi", "Guwahati"),
            ("Delhi", "Mumbai"),
            ("Delhi", "Bengaluru"),
        ]

        print("  ── Point-to-Point Queries ──")
        for src, dst in queries:
            d, p = dijkstra(INDIA_ROADS, src)
            single_query(src, dst, d, p)
