# tensorpack-f

Tensorpack-f is a graph-based semantic relationship discovery and analysis system that operates in hyperdimensional space to find, rank, and traverse meaningful connections between data entities.

It is a Semantic Graph Engine which builds and queries connection graphs from high-dimensional embeddings and transformations, a Multi-Modal Search Framework that provides 4 specialized search algorithms for different relationship discovery patterns, a Hyperdimensional Navigation System that uses geometric projections and energy-based metrics to traverse semantic spaces, and a Connection Intelligence Platform which enriches raw connections with metadata, previews, and diagnostic information.

## Primary Functions

- Entity-Centric Discovery (entity_search) - Find strongest semantic neighbors for any node
- Directional Pathfinding (search_directional) - Navigate optimal paths between specific entities using cluster-based algorithms
- Energy Flow Analysis (search_energy_flow) - Follow energy gradients to discover high-activation pathways
- Bidirectional Relationship Mining (search_reciprocal) - Identify mutually reinforcing semantic connections
- Graph Construction (discover_connections) - Build enriched adjacency from MatrixTransformer results

### entity_search

It finds the strongest semantic neighbors for any node from precomputed payload data.

**Steps:**
- It resolves node identifiers from integer, string, or composite key formats to consistent lookup keys
- It searches multiple connection indices (connections, by_signature, by_source_row) for matching edges
- It supports partial matching on source_row for flexible entity lookup
- It ranks all neighbors by combined_score or strength metrics
- It deduplicates results while preserving order
- It returns top-k results or all matches when k is None

**Return Value:** Returns a list of edge dictionaries sorted by strength, each containing:
- All original edge metadata and metrics
- `combined_score`, `strength`: connection strength scores
- `source_dataset_id`, `source_element_index`, `target_element_index`: identification fields

### search_directional

It finds the shortest path from a source node to a target node using cluster-based navigation to traverse a hyperdimensional connection graph.

**Steps:**
- It resolves start and target nodes from string/numeric identifiers to consistent keys (to the unique id)
- It uses DynamicGraph (from graph.py) to create an adjacency data
- From DynamicGraph it calls graph.navigate_cluster(all_nodes) which uses a divide-and-conquer traversal strategy
- It finds the positions of start and target nodes in the visited order from cluster navigation and then extracts the path segment between them (handling both forward and reverse directions)
- It builds results by constructing detailed path results with edge information for each step
- It includes metadata, previews and scores
- It returns a list of dictionaries representing each edge in the path

**Return Value:** Returns a list of edge dictionaries, each containing:
- `src`, `tgt`: source and target node IDs
- `score`: connection strength score
- `edge`: original edge data with all metrics
- `source_metadata`, `target_metadata`: node metadata
- `source_preview`, `target_preview`: content previews

### search_energy_flow

It follows energy gradients through the connection graph to surface high-activation pathways.

**Steps:**
- It resolves the start node from string/numeric identifiers to a consistent key
- It gathers all neighbors for the starting node from the adjacency data
- It computes energy scores using energy_gradient norms (with fallbacks to target_energy or local_energy)
- It builds result entries with full edge data, metadata, and previews for each neighbor
- It ranks candidates by energy score in descending order
- It returns all neighbors or top-k based on the top_k parameter

**Return Value:** Returns a list of edge dictionaries sorted by energy score, each containing:
- `from`, `to`: source and target node IDs
- `score`: computed energy flow score
- `edge`: original edge data with all metrics
- `source_metadata`, `target_metadata`: node metadata
- `source_preview`, `target_preview`: content previews
- `projection_diagnostics`: projection-related diagnostic information

### search_reciprocal

It identifies mutually reinforcing semantic connections by examining reciprocal edges in both directions.

**Steps:**
- It iterates through all edges in the normalized adjacency data
- It checks if each edge has a reciprocal_angle metric
- It looks up the reverse edge (from target back to source)
- It verifies both directions have reciprocal_angle below the threshold
- It collects comprehensive metadata for both forward and reverse edges
- It returns only relationships that are strong in both directions

**Return Value:** Returns a list of edge dictionaries, each containing:
- `src`, `tgt`: source and target node IDs
- `reciprocal_angle`, `reciprocal_angle_rev`: angular metrics for both directions
- `edge`, `rev_edge`: complete edge data for both directions
- `source_metadata`, `target_metadata`: metadata for both nodes
- `source_preview`, `target_preview`: content previews
- `rev_source_metadata`, `rev_target_metadata`: reverse direction metadata
- `rev_source_preview`, `rev_target_preview`: reverse direction content previews

### discover_connections

It builds enriched adjacency mappings from MatrixTransformer traversal output.

**Steps:**
- It accesses the traversal adapter from MatrixTransformer via mt._traverse_graph.adjacency()
- It extracts payload lookup helpers (matrix_metadata, projected_points, proj_dists)
- It normalizes all numeric values to JSON-serializable formats (converting numpy types)
- It iterates through raw adjacency and enriches each edge with metadata
- It attaches source and target metadata from edge-level or payload-level lookups
- It includes projected coordinates and preview snippets for both nodes
- It produces ready-to-query semantic graphs with consistent string keys

**Return Value:** Returns a dictionary mapping source node keys to lists of enriched edge dictionaries, each containing:
- `src`, `tgt`, `target_idx`: source and target node string keys
- `original`: original edge data from traversal
- `numeric`: normalized numeric representation
- `source_metadata`, `target_metadata`: rich metadata for both nodes
- `source_preview`, `target_preview`: content snippets for quick review
- `source_projected`, `target_projected`: geometric projection coordinates
- `projection_diagnostics`: projection quality and diagnostic information

## Use Cases

- Knowledge Discovery - Finding hidden relationships in complex datasets
- Semantic Search - Content recommendation based on hyperdimensional similarity
- Network Analysis - Understanding information flow and cluster structures
- AI/ML Pipeline Integration - Processing MatrixTransformer outputs for downstream analysis
