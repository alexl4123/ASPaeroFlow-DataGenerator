#!/usr/bin/env python3
"""
Check whether each sector forms a connected subgraph.

Usage (default filenames in current directory):
    python check_sector_connectivity.py

Or with custom paths:
    python check_sector_connectivity.py \
        --edges graph_edges.csv \
        --assignments navaid_sector_assignment.csv
"""

import argparse
import csv
from collections import defaultdict

import networkx as nx


def load_graph(edges_path: str) -> nx.Graph:
    """Load an undirected graph from graph_edges.csv."""
    G = nx.Graph()
    with open(edges_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row["source"])
            tgt = int(row["target"])
            dist = float(row["dist_m"])
            G.add_edge(src, tgt, dist_m=dist)
    return G


def load_assignments(assignments_path: str):
    """
    Load navaid -> sector assignments.

    Returns:
        nav_to_sec: dict[navaid_id] = sector_id
        sec_to_navs: dict[sector_id] = set of navaid_ids
    """
    nav_to_sec = {}
    sec_to_navs = defaultdict(set)

    with open(assignments_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            navaid_id = int(row["Navaid_ID"])
            sector_id = int(row["Sector_ID"])
            nav_to_sec[navaid_id] = sector_id
            sec_to_navs[sector_id].add(navaid_id)

    return nav_to_sec, sec_to_navs


def find_disconnected_sectors(G: nx.Graph, sec_to_navs):
    """
    For each sector, check if the induced subgraph is connected.

    Returns:
        List of (sector_id, num_components, component_sizes_list).
    """
    disconnected = []

    for sector_id in sorted(sec_to_navs.keys()):
        nodes = sec_to_navs[sector_id]
        if not nodes:
            # No navaids in this sector -> ignore
            continue

        # Ensure all navaids appear in the graph (isolated nodes if necessary)
        for n in nodes:
            if n not in G:
                G.add_node(n)

        if len(nodes) <= 1:
            # Single-node sectors are trivially connected
            continue

        subG = G.subgraph(nodes)
        if not nx.is_connected(subG):
            components = list(nx.connected_components(subG))
            sizes = sorted((len(c) for c in components), reverse=True)
            disconnected.append((sector_id, len(components), sizes))

    return disconnected


def main():
    parser = argparse.ArgumentParser(
        description="Check whether each sector forms a connected subgraph."
    )
    parser.add_argument(
        "--edges",
        default="graph_edges.csv",
        help="Path to graph_edges.csv (default: graph_edges.csv)",
    )
    parser.add_argument(
        "--assignments",
        default="navaid_sector_assignment.csv",
        help="Path to navaid_sector_assignment.csv (default: navaid_sector_assignment.csv)",
    )

    args = parser.parse_args()

    print(f"Reading graph from: {args.edges}")
    G = load_graph(args.edges)
    print(f"  -> Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print(f"Reading navaid-sector assignments from: {args.assignments}")
    _, sec_to_navs = load_assignments(args.assignments)
    print(f"  -> Found {len(sec_to_navs)} sectors.\n")

    disconnected = find_disconnected_sectors(G, sec_to_navs)

    if not disconnected:
        print("✅ All sectors form connected subgraphs.")
    else:
        print("⚠️ Disconnected sectors detected:")
        for sector_id, n_components, sizes in disconnected:
            size_str = ", ".join(str(s) for s in sizes)
            print(f"  - Sector {sector_id}: {n_components} components (sizes: {size_str})")


if __name__ == "__main__":
    main()
