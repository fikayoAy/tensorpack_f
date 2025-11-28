# tensorpack-f

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE.md)
[![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](./docs/INSTALL.md)
[![Status](https://img.shields.io/badge/status-beta-yellow)]()
[![Discord](https://img.shields.io/discord/1285677307163574322?color=7289da&label=Discord&logo=discord&logoColor=white)](https://discord.gg/uJyNEeYX2U)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](./docs/README.md)

> [!NOTE]
> **Beta Status:** tensorpack-f is under active development. Core algorithms are usable for experimentation; documentation and tooling are being improved.

**Tensorpack-f is a graph-based semantic relationship discovery and analysis system that operates in hyperdimensional space to find, rank, and traverse meaningful connections between data entities.**

It is a Semantic Graph Engine which builds and queries connection graphs from high-dimensional embeddings and transformations, a Multi-Modal Search Framework that provides 4 specialized search algorithms for different relationship discovery patterns, a Hyperdimensional Navigation System that uses geometric projections and energy-based metrics to traverse semantic spaces, and a Connection Intelligence Platform which enriches raw connections with metadata, previews, and diagnostic information.

## Quick Example

Basic usage with a precomputed payload:

```py
from connection import load_precomputed_connections, entity_search, search_directional

payload = load_precomputed_connections('connections_linked.json')
# Find top-5 neighbors for node '42'
neighbors = entity_search(payload, node_id='42', k=5)

# Directional path between two nodes (requires payload['connections'])
path = search_directional(payload.get('connections', {}), start_node='42', target_node='128', payload=payload)
```

## Primary Functions

- Entity-Centric Discovery (`entity_search`) - Find strongest semantic neighbors for any node
- Directional Pathfinding (`search_directional`) - Navigate optimal paths between specific entities using cluster-based algorithms
- Energy Flow Analysis (`search_energy_flow`) - Follow energy gradients to discover high-activation pathways
- Bidirectional Relationship Mining (`search_reciprocal`) - Identify mutually reinforcing semantic connections
- Graph Construction (`discover_connections`) - Build enriched adjacency from `MatrixTransformer` results

### `entity_search`

It finds the strongest semantic neighbors for any node from precomputed payload data.

**Steps:**
- It resolves node identifiers from integer, string, or composite key formats to consistent lookup keys.
- It searches multiple connection indices (`connections`, `by_signature`, `by_source_row`) for matching edges.
- It supports partial matching on `source_row` for flexible entity lookup.
- It ranks neighbors by `combined_score` or `strength` and deduplicates while preserving order.
- It returns top-k results or all matches when `k` is `None`.

**Return Value:** Returns a list of edge dictionaries sorted by strength, each containing the original edge metadata and identification fields.

### `search_directional`

It finds the shortest path from a source node to a target node using cluster-based navigation to traverse a hyperdimensional connection graph.

**Steps:**
- It resolves start and target nodes from string/numeric identifiers to consistent keys (unique id).
- It uses `DynamicGraph` (from `graph.py`) to create adjacency and node metadata.
- From `DynamicGraph` it calls `graph.navigate_cluster(all_nodes)`, which uses a divide-and-conquer traversal strategy.
- It finds the positions of start and target nodes in the visited order produced by cluster navigation and extracts the path segment between them (handling both forward and reverse directions).
- It builds results by constructing detailed path entries with per-hop edge information, metadata, previews, and scores.
- It returns a list of dictionaries representing each edge in the path.

**Return Value:** Returns a list of edge dictionaries, each containing:
- `src`, `tgt`: source and target node IDs
- `score`: connection strength score
- `edge`: original edge data with all metrics
- `source_metadata`, `target_metadata`: node metadata
- `source_preview`, `target_preview`: content previews

### `search_energy_flow`

It follows energy gradients through the connection graph to surface high-activation pathways.

**Steps:**
- It resolves the start node to a consistent key.
- It gathers neighbors and computes energy scores using `energy_gradient` norms (with fallbacks to `target_energy` or `local_energy`).
- It builds result entries with full edge data, metadata, and previews for each neighbor and ranks candidates by energy score.
- It returns all neighbors or top-k based on the `top_k` parameter.

**Return Value:** Returns a list of edge dictionaries sorted by energy score, each containing `from`, `to`, `score`, `edge`, metadata, previews, and `projection_diagnostics` when available.

### `search_reciprocal`

It identifies mutually reinforcing semantic connections by examining reciprocal edges in both directions.

**Steps:**
- Iterates through normalized adjacency edges and checks for `reciprocal_angle`.
- Locates the reverse edge and verifies both directions fall below the configured angle threshold.
- Collects comprehensive metadata for forward and reverse edges and returns only strong bidirectional relationships.

**Return Value:** Returns a list of edge dictionaries including `edge` and `rev_edge`, reciprocal angle metrics, metadata and previews for both directions.

### `discover_connections`

It builds enriched adjacency mappings from `MatrixTransformer` traversal output.

**Steps:**
- Accesses the traversal adapter via `mt._traverse_graph.adjacency()`.
- Extracts payload helpers (`matrix_metadata`, `projected_points`, `proj_dists`).
- Normalizes numeric values to JSON-serializable formats, enriches each edge with metadata and projections, and returns a ready-to-query adjacency mapping.

**Return Value:** Returns a dictionary mapping source node keys to lists of enriched edge dictionaries with `numeric`, metadata, previews and projection coordinates.

## Use Cases

- Knowledge Discovery - Finding hidden relationships in complex datasets
- Semantic Search - Content recommendation based on hyperdimensional similarity
- Network Analysis - Understanding information flow and cluster structures
- AI/ML Pipeline Integration - Processing `MatrixTransformer` outputs for downstream analysis
