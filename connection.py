"""connection.py

Simplified, coherent module for building and querying hyperdimensional
connections using MatrixTransformer results.

This file provides:
- Utilities for flexible key mapping between strings and ints
- Entity resolution helpers (resolve_entity_id)
- Serialization helpers for edges
- Query functions: entity_search, discover_connections
- Advanced traversal/search helpers: search_directional, search_energy_flow,
  search_reciprocal
- An ingestion driver: process_dc_results that produces enriched connection
  metadata and saves the `dc_results_output.json` output.

Design goals:
- Robust imports & fallbacks, friendly logging, and resilient parsing
- Try to call external optimized metrics/builders (if present), falling back
  to lightweight Python logic when they aren't available.
"""

from __future__ import annotations


import json
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple
from pathlib import Path
from itertools import chain
from collections import defaultdict
import bisect

try:
    import numpy as np
except Exception:
    np = None  # Numpy required for numeric ops; many functions will check

try:
    from matrixtransformer import MatrixTransformer
except Exception:
    # Keep a local fallback to avoid hard import-time failure
    MatrixTransformer = None

try:
    from load import load_payload, get_byte_metadata
except Exception:
    load_payload = None
    get_byte_metadata = None



# Configure module logger
logger = logging.getLogger(__name__)
if logging.getLogger().handlers:
    # Use existing configuration
    pass
else:
    logging.basicConfig(level=logging.INFO)


# Global maps for label resolution (optional, can be populated by user)
id_to_label: Dict[str, str] = {}
label_to_id: Dict[str, str] = {}

# --- Loader for precomputed connections_linked.json ---
def load_precomputed_connections(json_path: str | Path) -> Dict[str, Any]:
    """
    Load and normalize precomputed connections from connections_linked.json.
    Returns a payload dict with 'connections' mapping: {source_idx: [edge_dict, ...]}
    and optional metadata.
    """
    path = Path(json_path)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    connections = {}
    by_source_row = {}
    by_signature = {}
    by_composite_key = {}  # Unique key: (dataset_id, source_element_index)
    for row in rows:
        src = row.get("source_element_index")
        tgt = row.get("target_element_index")
        if src is None or tgt is None:
            continue
        src = int(src)
        tgt = int(tgt)
        # Build edge dict, keeping all metrics and metadata
        edge = dict(row)
        edge["target_idx"] = tgt
        # add formats for convenience
        src_row = row.get("source_row") or ""
        src_dataset = row.get("source_dataset_id") or ""
        # parse signature token from source_row (first tab-separated field, strip quotes)
        sig = ""
        if isinstance(src_row, str) and src_row:
            first = src_row.split('\t', 1)[0]
            sig = first.strip().strip('"')
        # Create composite key for unique identification across datasets
        composite_key = f"{src_dataset}:{src}"
        # Optionally parse/repair fields here if needed
        connections.setdefault(src, []).append(edge)
        by_composite_key.setdefault(composite_key, []).append(edge)
        if src_row:
            by_source_row.setdefault(src_row, []).append(edge)
        if sig:
            by_signature.setdefault(sig, []).append(edge)
    payload = {"connections": connections,
               "by_composite_key": by_composite_key,
               "by_source_row": by_source_row,
               "by_signature": by_signature}
    return payload


def _get_by_string_key(m: Mapping, key: Any):
    """Return value by flexible lookup (string or numeric key).

    Accepts dicts keyed by integers or strings and attempts both.
    Returns None if not found.
    """
    try:
        if not isinstance(m, Mapping):
            return None
        if key in m:
            return m[key]
        key_str = str(key)
        if key_str in m:
            return m[key_str]
        try:
            key_int = int(key_str)
            if key_int in m:
                return m[key_int]
        except Exception:
            pass
    except Exception:
        logger.exception("_get_by_string_key: unexpected error")
    return None


def _neighbors_for_key(adj_or_connections: Mapping, key: Any) -> List[Tuple[Any, Dict]]:
    """Return list of (neighbor_key, edge_dict) for the given key.

    Supports three shapes:
    - adjacency mapping: {src: [(tgt, edge), ...]}
    - connections mapping: {src: [edge_dict, ...]} where edge_dict has target_idx/target_element_index
    - payload dict containing 'connections' or 'by_composite_key'
    
    The key can be:
    - An integer (source_element_index)
    - A string representation of an integer
    - A composite key "dataset_id:source_element_index"
    """
    if adj_or_connections is None:
        return []
    # allow passing payload directly - prefer by_composite_key for unique identification
    if isinstance(adj_or_connections, dict) and not isinstance(key, dict):
        if 'by_composite_key' in adj_or_connections:
            adj = adj_or_connections.get('by_composite_key', {})
        elif 'connections' in adj_or_connections:
            adj = adj_or_connections.get('connections', {})
        else:
            adj = adj_or_connections
    else:
        adj = adj_or_connections

    # try direct key or flexible lookup
    val = None
    if isinstance(adj, Mapping):
        if key in adj:
            val = adj[key]
        else:
            val = _get_by_string_key(adj, key)

    if not val:
        return []

    # if already adjacency style (list of (tgt, edge))
    if isinstance(val, list) and val and isinstance(val[0], (list, tuple)):
        return [(v, e) for v, e in val]

    # otherwise we expect list of dicts
    out = []
    for e in (val or []):
        if not isinstance(e, dict):
            continue
        tgt = e.get('target_idx') or e.get('target_element_index') or e.get('target')
        # Create composite key for target if dataset info is available
        tgt_dataset = e.get('target_dataset_id') or ''
        if tgt_dataset:
            tgt_composite = f"{tgt_dataset}:{tgt}"
            out.append((tgt_composite, e))
        else:
            out.append((tgt, e))
    return out


def _iter_normalized_adj(adj_or_connections: Mapping):
    """Yield (src_key_str, neighbors_list_of_(tgt, edge)) for any supported mapping."""
    if adj_or_connections is None:
        return
    if isinstance(adj_or_connections, dict) and 'connections' in adj_or_connections:
        adj = adj_or_connections['connections']
    else:
        adj = adj_or_connections

    if not isinstance(adj, Mapping):
        return

    for src, val in adj.items():
        src_key = str(src)
        if isinstance(val, list) and val and isinstance(val[0], (list, tuple)):
            yield src_key, [(v, e) for v, e in val]
        else:
            neighbors = []
            for e in (val or []):
                if not isinstance(e, dict):
                    continue
                tgt = e.get('target_idx') or e.get('target_element_index') or e.get('target')
                neighbors.append((tgt, e))
            yield src_key, neighbors


def _build_reverse_adjacency(adj_or_connections: Mapping):
    """Build a reverse adjacency mapping: target_key_str -> list of (source_key_str, edge_dict).

    Keys are strings for consistent lookup. Edge objects are passed through as-is.
    """
    rev = defaultdict(list)
    try:
        for src_key, neighbors in _iter_normalized_adj(adj_or_connections):
            for tgt, edge in neighbors:
                try:
                    tkey = str(tgt)
                except Exception:
                    tkey = str(tgt)
                rev[tkey].append((src_key, edge))
    except Exception:
        logger.exception("_build_reverse_adjacency failed")
    return rev


def resolve_entity_id(entity: Any, label_map: Optional[Mapping[str, str]] = None) -> int:
    """Resolve an entity identifier to an integer ID.

    Accepts: int, string containing digits, or named label. Raises KeyError if label
    not found in mapping. If mapping is None, uses `label_to_id` (global).
    """
    if label_map is None:
        label_map = label_to_id
    if isinstance(entity, int):
        return entity
    if isinstance(entity, str):
        s = entity.strip()
        if s.isdigit():
            return int(s)
        # Lookup label
        if s in label_map:
            return int(label_map[s]) if isinstance(label_map[s], str) and label_map[s].isdigit() else label_map[s]
        # fallback to flexible lookup
        v = _get_by_string_key(label_map, s)
        if v is not None:
            return int(v) if isinstance(v, str) and v.isdigit() else v
        raise KeyError(f"Unknown entity label: {entity}")
    raise TypeError("Entity must be int or str")


def serialize_edge(src_id: Any, tgt_id: Any, metrics: Optional[Mapping] = None, id_label_map: Optional[Mapping] = None) -> Dict:
    """Return JSON-serializable edge representation with human readable labels.
    """
    if id_label_map is None:
        id_label_map = id_to_label
    s, t = str(src_id), str(tgt_id)
    out = {
        "source_idx": s,
        "target_idx": t,
        "source_label": _get_by_string_key(id_label_map, s) or s,
        "target_label": _get_by_string_key(id_label_map, t) or t,
    }
    if isinstance(metrics, Mapping):
        out.update({k: v for k, v in metrics.items()})
    return out


def _resolve_node_string_key(payload: Dict, node_key: Any) -> Optional[str]:
    """Resolve a `node_key` to a consistent string key for metadata lookup.

    Searches: payload.node_id_map, node_texts, matrix_metadata common fields.
    Returns string key to be used in subsequent lookup; returns None if invalid.
    """
    try:
        if not isinstance(payload, dict):
            return None
        if node_key is None:
            return None
        kstr = str(node_key).strip()
        # Return None for empty/whitespace-only strings
        if not kstr:
            return None
        # Direct payload node_id_map
        nid_map = payload.get("node_id_map") or {}
        if isinstance(nid_map, dict):
            if kstr in nid_map:
                return str(nid_map[kstr])
            v = _get_by_string_key(nid_map, kstr)
            if v is not None:
                return str(v)
        # Node texts
        node_texts = payload.get("node_texts") or {}
        if isinstance(node_texts, dict):
            for k, v in node_texts.items():
                if str(k).strip() == kstr or str(v).strip() == kstr:
                    return str(k)
        # matrix_metadata fields
        meta = payload.get("matrix_metadata") or {}
        if isinstance(meta, dict):
            for k, m in meta.items():
                if not isinstance(m, dict):
                    continue
                for fld in ("manifest_id", "chunk_id", "chunk_index", "dataset_id", "id"):
                    val = m.get(fld)
                    if val is not None and str(val).strip() == kstr:
                        return str(k)
            # direct key match
            if kstr in meta:
                return str(kstr)
        # check precomputed connection indices built by loader
        # by_signature maps signature -> [edge, ...]; CID can appear as source or target
        sig_index = payload.get('by_signature') if isinstance(payload, dict) else {}
        if isinstance(sig_index, dict) and kstr in sig_index:
            lst = sig_index.get(kstr) or []
            for edge in lst:
                # Check if CID appears in source_row (meaning this node is the source)
                src_row = edge.get('source_row', '')
                if kstr in str(src_row):
                    src = edge.get('source_element_index') or edge.get('source_idx')
                    src_dataset = edge.get('source_dataset_id') or ''
                    if src is not None:
                        # Return composite key for unique identification
                        return f"{src_dataset}:{src}"
                # Check if CID appears in target_row (meaning this node is the target)
                tgt_row = edge.get('target_row', '')
                if kstr in str(tgt_row):
                    tgt = edge.get('target_element_index') or edge.get('target_idx')
                    tgt_dataset = edge.get('target_dataset_id') or ''
                    if tgt is not None:
                        # Return composite key for target node
                        return f"{tgt_dataset}:{tgt}"
        
        # by_source_row may contain the full source_row string; prefer exact match then substring
        row_index = payload.get('by_source_row') if isinstance(payload, dict) else {}
        if isinstance(row_index, dict):
            if kstr in row_index:
                lst = row_index.get(kstr) or []
                if lst:
                    e0 = lst[0]
                    src = e0.get('source_element_index') or e0.get('source_idx') or e0.get('source')
                    src_dataset = e0.get('source_dataset_id') or ''
                    if src is not None:
                        return f"{src_dataset}:{src}"
            # substring match
            for src_row_key, lst in row_index.items():
                if kstr in str(src_row_key):
                    if lst:
                        e0 = lst[0]
                        src = e0.get('source_element_index') or e0.get('source_idx') or e0.get('source')
                        src_dataset = e0.get('source_dataset_id') or ''
                        if src is not None:
                            return f"{src_dataset}:{src}"
        # not found -> return normalized string for safe use in JSON-style keys
        return kstr
    except Exception:
        logger.exception("_resolve_node_string_key failed")
        return None


def _get_edge_field(edge: Dict, name: str, default=None):
    try:
        if not isinstance(edge, dict):
            return default
        if 'numeric' in edge and isinstance(edge['numeric'], dict) and name in edge['numeric']:
            return edge['numeric'][name]
        if 'original' in edge and isinstance(edge['original'], dict) and name in edge['original']:
            return edge['original'][name]
        if name in edge:
            return edge[name]
    except Exception:
        logger.exception("_get_edge_field error")
    return default


def _preview_snippet(meta: Optional[dict], length: Optional[int] = None) -> Optional[str]:
    """Return preview string from metadata's `content_preview`.

    By default (`length is None`) return the full preview text. If a
    positive `length` is provided, return a truncated snippet.
    Returns None if no preview is available.
    """
    try:
        if not isinstance(meta, dict):
            return None
        pv = meta.get('content_preview')
        if pv is None:
            return None
        s = str(pv)
        if length is None or length <= 0:
            return s
        if len(s) <= length:
            return s
        return s[:length]
    except Exception:
        return None


def compute_metric_direction(payload: Dict, node_key: Any, method: str = 'composite_metrics', weight_by: str = 'combined_score', neighborhood_k: int = 10) -> Optional[List[float]]:
    """Compute a metric-driven direction vector for `node_key`.

    Method supported:
    - 'composite_metrics': build a composite weight from scalar metrics
      (log_map_norm, local_energy, |local_curvature|, ||energy_gradient||,
      hyperdimensional_dist) and compute a weighted centroid (or average
      energy_gradient vectors if projections are absent).

    Returns a list(float) direction (not normalized). May return None if no data.
    """
    try:
        if not isinstance(payload, dict):
            return None
        conns = payload.get('connections', {}) or {}
        node = _resolve_node_string_key(payload, node_key)
        if node is None:
            return None

        # gather neighbors
        neighbors = _neighbors_for_key(conns, node)
        if not neighbors:
            return None

        # We no longer inspect per-edge 'log_map' or 'transported_log_map' here;
        # direction is derived via the composite-metrics strategy below.

        # Composite-metrics method: synthesize weights from scalar metrics
        if method == 'composite_metrics':
            # metrics to use (order-insensitive)
            metric_names = [
                'log_map_norm',
                'local_energy',
                'local_curvature',
                'energy_gradient',
                'hyperdimensional_dist',
            ]
            # collect values per neighbor
            vals = {m: [] for m in metric_names}
            for v, edge in neighbors:
                # log_map_norm
                vals['log_map_norm'].append(float(_get_edge_field(edge, 'log_map_norm') or 0.0))
                vals['local_energy'].append(float(_get_edge_field(edge, 'local_energy') or 0.0))
                # curvature use abs()
                try:
                    cv = _get_edge_field(edge, 'local_curvature')
                    vals['local_curvature'].append(abs(float(cv)) if cv is not None else 0.0)
                except Exception:
                    vals['local_curvature'].append(0.0)
                # energy_gradient: if vector use norm, else scalar
                eg = _get_edge_field(edge, 'energy_gradient')
                try:
                    if isinstance(eg, (list, tuple)) or (np is not None and isinstance(eg, np.ndarray)):
                        if np is not None:
                            vals['energy_gradient'].append(float(np.linalg.norm(np.asarray(eg, dtype=float))))
                        else:
                            vals['energy_gradient'].append(float(math.sqrt(sum((float(x) ** 2) for x in eg))))
                    else:
                        vals['energy_gradient'].append(float(eg or 0.0))
                except Exception:
                    vals['energy_gradient'].append(0.0)
                vals['hyperdimensional_dist'].append(float(_get_edge_field(edge, 'hyperdimensional_dist') or 0.0))

            # normalize per-metric (min-max) and combine with default alphas
            eps = 1e-12
            normed = {}
            for m in metric_names:
                arr = None
                try:
                    if np is not None:
                        arr = np.asarray(vals[m], dtype=float)
                        mn = float(arr.min())
                        mx = float(arr.max())
                        norm = (arr - mn) / (mx - mn + eps)
                        normed[m] = norm.tolist()
                    else:
                        lst = vals[m]
                        mn = min(lst) if lst else 0.0
                        mx = max(lst) if lst else 0.0
                        normed[m] = [ (x - mn) / (mx - mn + eps) for x in lst ]
                except Exception:
                    normed[m] = [0.0] * len(vals[m])

            # default alpha weights: invert hyperdimensional_dist (smaller=closer)
            alphas = {
                'log_map_norm': 1.0,
                'local_energy': 1.0,
                'local_curvature': 1.0,
                'energy_gradient': 1.0,
                'hyperdimensional_dist': -1.0,
            }

            # build composite weights
            comp = [0.0] * len(neighbors)
            for m in metric_names:
                wlist = normed.get(m, [0.0] * len(neighbors))
                alpha = float(alphas.get(m, 1.0))
                for i, v in enumerate(wlist):
                    # invert hyperdimensional_dist contribution by using negative alpha
                    if m == 'hyperdimensional_dist':
                        comp[i] += alpha * (1.0 - v)
                    else:
                        comp[i] += alpha * v

            # clamp negatives to zero
            comp = [max(0.0, float(x)) for x in comp]
            total = sum(comp)

            # If projections are available, compute weighted centroid direction
            proj = payload.get('projected_points', {}) or {}
            node_proj = _get_by_string_key(proj, node)
            if node_proj is not None and total > 0.0:
                weighted_sum = None
                for (v, edge), w in zip(neighbors, comp):
                    if w <= 0.0:
                        continue
                    p = _get_by_string_key(proj, str(v))
                    if p is None:
                        continue
                    if weighted_sum is None:
                        weighted_sum = [float(x) * w for x in p]
                    else:
                        weighted_sum = [a + float(b) * w for a, b in zip(weighted_sum, p)]
                if weighted_sum is None:
                    return None
                centroid = [x / total for x in weighted_sum]
                return [float(t - s) for s, t in zip(node_proj, centroid)]

            # Fallback: if any vector-like metric exists (energy_gradient as vector), average those vectors
            vecs = []
            for (v, edge), w in zip(neighbors, comp):
                if w <= 0.0:
                    continue
                eg = _get_edge_field(edge, 'energy_gradient')
                if eg is None:
                    continue
                try:
                    if np is not None:
                        arr = np.asarray(eg, dtype=float)
                        vecs.append(arr * w)
                    else:
                        vecs.append([float(x) * w for x in eg])
                except Exception:
                    continue
            if vecs:
                if np is not None:
                    s = np.sum(np.vstack(vecs), axis=0)
                    if np.linalg.norm(s) == 0:
                        return None
                    return [float(x) for x in s.tolist()]
                else:
                    summed = None
                    for vec in vecs:
                        if summed is None:
                            summed = list(vec)
                        else:
                            summed = [a + b for a, b in zip(summed, vec)]
                    return [float(x) for x in summed]
        # No other methods supported; composite_metrics handled above.
    except Exception:
        logger.exception('compute_metric_direction failed')
    return None


def discover_connections(mt: Any) -> Dict[str, List[Tuple[str, Dict]]]:
    """Return enriched adjacency mapping (string keys) from `mt` traversal adapter.

    If `build_adjacency` or `mt._traverse_graph.adjacency` is not available, returns
    empty mapping.
    """
    try:
        if mt is None or not hasattr(mt, '_traverse_graph'):
            logger.info("discover_connections: missing traversal adapter")
            return {}
        if not hasattr(mt._traverse_graph, 'adjacency'):
            logger.info("discover_connections: traversal adapter has no adjacency()");
            return {}
        logger.info("discover_connections: calling adjacency()")
        raw_adj = mt._traverse_graph.adjacency()
        logger.info(f"discover_connections: raw_adj has {len(raw_adj)} nodes")
        if raw_adj:
            sample_node = list(raw_adj.keys())[0]
            logger.info(f"discover_connections: sample node {sample_node} has {len(raw_adj[sample_node])} neighbors")
        # Build payload lookup helpers
        payload = getattr(mt._traverse_graph, 'payload', {})
        matrix_meta = payload.get('matrix_metadata', {}) if isinstance(payload, dict) else {}
        proj_points = payload.get('projected_points', {}) if isinstance(payload, dict) else {}
        proj_dists = payload.get('proj_dists', {}) if isinstance(payload, dict) else {}

        def _to_numeric_value(x):
            try:
                if np is not None:
                    if isinstance(x, (np.integer, np.floating)):
                        return float(x)
                if isinstance(x, (int, float)):
                    return float(x)
            except Exception:
                pass
            return x

        def _numeric_summary(edge: Dict):
            if not isinstance(edge, dict):
                return edge
            out = {}
            for k, v in edge.items():
                try:
                    if isinstance(v, (list, tuple)):
                        out[k] = [ _to_numeric_value(x) for x in v ]
                    elif np is not None and isinstance(v, np.ndarray):
                        out[k] = [ _to_numeric_value(x) for x in v.flatten().tolist() ]
                    elif isinstance(v, (int, float)):
                        out[k] = float(v)
                    elif np is not None and isinstance(v, (np.integer, np.floating)):
                        out[k] = float(v)
                    else:
                        out[k] = v
                except Exception:
                    out[k] = v
            return out

        enriched = {}
        for src, neighbors in (raw_adj or {}).items():
            src_key = str(src)
            enriched[src_key] = []
            for tgt, edge in neighbors:
                tgt_key = str(tgt)
                numeric = _numeric_summary(edge) if isinstance(edge, dict) else edge
                # Prefer metadata attached to the edge itself; fall back to
                # payload-level matrix_metadata if absent.
                edge_src_meta = None
                edge_tgt_meta = None
                if isinstance(edge, dict):
                    edge_src_meta = edge.get('source_metadata')
                    edge_tgt_meta = edge.get('target_metadata')
                src_meta = edge_src_meta or _get_by_string_key(matrix_meta, src_key)
                tgt_meta = edge_tgt_meta or _get_by_string_key(matrix_meta, tgt_key)
                src_proj = _get_by_string_key(proj_points, src_key)
                tgt_proj = _get_by_string_key(proj_points, tgt_key)
                enriched_entry = {
                    'src': src_key,
                    'tgt': tgt_key,
                    'target_idx': tgt_key,
                    'original': edge,
                    'numeric': numeric,
                    'source_metadata': src_meta,
                    'target_metadata': tgt_meta,
                    'source_preview': _preview_snippet(src_meta),
                    'target_preview': _preview_snippet(tgt_meta),
                    'source_projected': src_proj,
                    'target_projected': tgt_proj,
                    'projection_diagnostics': proj_dists,
                }
                enriched[src_key].append(enriched_entry)
        logger.debug("discover_connections: found adjacency for %d src nodes", len(enriched))
        return enriched
    except Exception:
        logger.exception("discover_connections: unexpected error")
        return {}



def entity_search(payload: dict, node_id: Any, k: Optional[int] = None, use_labels: bool = True, label_map: Optional[Mapping]=None, id_map: Optional[Mapping]=None) -> List[Dict]:
    """
    Return top-k neighbors of a node from precomputed payload.
    """
    try:
        connections = payload.get("connections", {})
        # Try integer, then string, then fallback to flexible lookup
        src_candidates = []
        if isinstance(node_id, int):
            src_candidates = [node_id, str(node_id)]
        elif isinstance(node_id, str):
            if node_id.isdigit():
                src_candidates = [int(node_id), node_id]
            else:
                src_candidates = [node_id]
        else:
            src_candidates = [node_id]

        neighbors = []
        for key in src_candidates:
            if key in connections:
                neighbors = connections[key]
                break
        if not neighbors:
            # Try flexible lookup
            neighbors = _get_by_string_key(connections, node_id) or []

        # If we still have no neighbors and node_id is a string, try the
        # additional indices created by the loader: by_signature and by_source_row
        if not neighbors and isinstance(node_id, str):
            sig_index = payload.get('by_signature', {}) if isinstance(payload, dict) else {}
            row_index = payload.get('by_source_row', {}) if isinstance(payload, dict) else {}
            # exact signature match
            neighbors = sig_index.get(node_id, [])
            if not neighbors:
                # exact source_row match
                neighbors = row_index.get(node_id, [])
            if not neighbors:
                # partial match on source_row (substring search)
                for src_row_key, lst in row_index.items():
                    if node_id in src_row_key:
                        neighbors.extend(lst)
                # deduplicate while preserving order
                if neighbors:
                    seen = set()
                    dedup = []
                    for e in neighbors:
                        key = (e.get('source_dataset_id'), e.get('source_element_index'), e.get('target_element_index'))
                        if key not in seen:
                            seen.add(key)
                            dedup.append(e)
                    neighbors = dedup

        # Sort by strength or combined_score if present
        def score(e):
            return float(e.get("combined_score", e.get("strength", 0.0)))
        sorted_neighbors = sorted(neighbors, key=score, reverse=True)
        # When k is None return all matches; otherwise return top-k
        if k is None:
            return sorted_neighbors
        return sorted_neighbors[:k]
    except Exception:
        logger.exception("entity_search failed")
        return []



def _convert_traverse_path_to_results(transformation_path, attention_scores, adj, start_key, target_key, payload):
    """
    Convert MatrixTransformer._traverse_graph results to connection search results format.
    
    Args:
        transformation_path: Path from MatrixTransformer
        attention_scores: Attention scores from traversal
        adj: Adjacency mapping
        start_key: Start node key
        target_key: Target node key  
        payload: Data payload
        
    Returns:
        List of connection results in expected format
    """
    try:
        logger.info(f"[PATH_CONVERT] Converting MatrixTransformer path to connection results")
        logger.info(f"[PATH_CONVERT] Path: {transformation_path}")
        logger.info(f"[PATH_CONVERT] Attention: {attention_scores}")
        
        results = []
        if not transformation_path:
            return results
            
        # For now, create a simple result based on the transformation path
        # This is a placeholder - actual implementation would depend on the
        # structure of transformation_path from MatrixTransformer
        
        # Extract matrix metadata and projections
        matrix_meta = payload.get('matrix_metadata', {}) if isinstance(payload, dict) else {}
        proj_points = payload.get('projected_points', {}) if isinstance(payload, dict) else {}
        
        # Create result entry for the path found
        result = {
            'source_element_index': start_key,
            'target_element_index': target_key,
            'transformation_path': transformation_path,
            'attention_scores': attention_scores,
            'path_length': len(transformation_path) if isinstance(transformation_path, (list, tuple)) else 1
        }
        
        # Add projection data if available
        try:
            start_proj = _get_by_string_key(proj_points, start_key)
            target_proj = _get_by_string_key(proj_points, target_key)
            if start_proj is not None:
                result['source_projection'] = start_proj
            if target_proj is not None:
                result['target_projection'] = target_proj
        except Exception:
            pass
            
        results.append(result)
        logger.info(f"[PATH_CONVERT] Created {len(results)} result entries")
        return results
        
    except Exception as e:
        logger.exception(f"[PATH_CONVERT] Failed to convert path: {e}")
        return []


def search_directional(adj: Mapping, start_node: Any, target_node: Any, payload: Optional[Dict] = None) -> List[Dict]:
    """
    Find the shortest path from source to target using navigate_cluster for pathfinding.
    Uses cluster-based navigation to traverse the connection graph.
    """
    import numpy as _np
    try:
        from graph import DynamicGraph
        
        logger.info(f"[DIRECTIONAL_SEARCH] Starting search from {start_node} to {target_node}")
        
        # Resolve start node
        start_key = _resolve_node_string_key(payload or {}, start_node)
        if start_key is None:
            logger.error(f"[DIRECTIONAL_SEARCH] Could not resolve start node: {start_node}")
            return []
        logger.info(f"[DIRECTIONAL_SEARCH] Start node resolved to: {start_key}")
        
        # Resolve target node
        tgt_key = _resolve_node_string_key(payload or {}, target_node)
        if tgt_key is None:
            logger.error(f"[DIRECTIONAL_SEARCH] Could not resolve target node: {target_node}")
            return []
        logger.info(f"[DIRECTIONAL_SEARCH] Target node resolved to: {tgt_key}")
        
        # Build a DynamicGraph from adjacency data
        logger.info("[DIRECTIONAL_SEARCH] Building DynamicGraph from adjacency data")
        graph = DynamicGraph(directed=True)
        
        # Add all nodes and edges from adjacency
        matrix_meta = payload.get('matrix_metadata', {}) if isinstance(payload, dict) else {}
        proj_dists = payload.get('proj_dists', {}) if isinstance(payload, dict) else {}
        
        for src_str, neighbors in _iter_normalized_adj(adj):
            if not graph.has_node(src_str):
                src_meta = _get_by_string_key(matrix_meta, src_str)
                graph.add_node(src_str, {'metadata': src_meta})
            
            for tgt, edge in neighbors:
                tgt_str = str(tgt)
                if not graph.has_node(tgt_str):
                    tgt_meta = _get_by_string_key(matrix_meta, tgt_str)
                    graph.add_node(tgt_str, {'metadata': tgt_meta})
                
                # Extract edge weight from edge data
                weight = float(_get_edge_field(edge, 'combined_score') or 
                             _get_edge_field(edge, 'log_map_norm') or 
                             _get_edge_field(edge, 'strength') or 1.0)
                
                graph.add_edge(src_str, tgt_str, weight=weight, edge_data=edge)
        
        logger.info(f"[DIRECTIONAL_SEARCH] Graph built with {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges")
        
        # Collect all nodes into a cluster and use navigate_cluster
        all_nodes = list(graph.graph.nodes())
        logger.info(f"[DIRECTIONAL_SEARCH] Navigating cluster with {len(all_nodes)} nodes")
        
        visited_order = graph.navigate_cluster(all_nodes)
        logger.info(f"[DIRECTIONAL_SEARCH] Cluster navigation complete, visited {len(visited_order)} nodes")
        
        # Find start and target positions in visited order
        try:
            start_idx = visited_order.index(start_key)
            target_idx = visited_order.index(tgt_key)
            logger.info(f"[DIRECTIONAL_SEARCH] Start at position {start_idx}, target at position {target_idx}")
        except ValueError as e:
            logger.error(f"[DIRECTIONAL_SEARCH] Could not find start or target in visited order: {e}")
            return []
        
        # Extract the path segment from start to target
        if start_idx < target_idx:
            path_nodes = visited_order[start_idx:target_idx + 1]
        else:
            path_nodes = visited_order[target_idx:start_idx + 1]
            path_nodes.reverse()
        
        logger.info(f"[DIRECTIONAL_SEARCH] Path found with {len(path_nodes)} nodes: {' -> '.join(path_nodes)}")
        
        # Build path results with edges
        path_results = []
        for i in range(len(path_nodes) - 1):
            s = path_nodes[i]
            t = path_nodes[i + 1]
            
            # Get edge data
            edge_data = graph.get_edge_data(s, t)
            if edge_data is None:
                logger.warning(f"[DIRECTIONAL_SEARCH] No edge found for {s} -> {t}, skipping")
                continue
            
            edge = edge_data.get('edge_data', {})
            edge_src_meta = (edge.get('source_metadata') if isinstance(edge, dict) else None) or _get_by_string_key(matrix_meta, s)
            edge_tgt_meta = (edge.get('target_metadata') if isinstance(edge, dict) else None) or _get_by_string_key(matrix_meta, t)
            
            score = float(_get_edge_field(edge, 'combined_score') or 
                        _get_edge_field(edge, 'log_map_norm') or 
                        _get_edge_field(edge, 'strength') or 0.0)
            
            path_results.append({
                'src': s,
                'tgt': t,
                'score': score,
                'edge': edge,
                'source_metadata': edge_src_meta,
                'target_metadata': edge_tgt_meta,
                'source_preview': _preview_snippet(edge_src_meta),
                'target_preview': _preview_snippet(edge_tgt_meta),
                'projection_diagnostics': proj_dists,
            })
        
        logger.info(f"[DIRECTIONAL_SEARCH] Returning path with {len(path_results)} edges")
        return path_results
        
    except Exception:
        logger.exception("search_directional failed")
        return []


def search_energy_flow(adj: Mapping, start_node: Any, steps: int = 5, deterministic: bool = True, payload: Optional[Dict] = None, top_k: Optional[int] = None) -> List[Dict]:
    import numpy as _np
    try:
        node = _resolve_node_string_key(payload or {}, start_node)
        if node is None:
            return []
        matrix_meta = payload.get('matrix_metadata', {}) if isinstance(payload, dict) else {}
        proj_points = payload.get('projected_points', {}) if isinstance(payload, dict) else {}
        proj_dists = payload.get('proj_dists', {}) if isinstance(payload, dict) else {}

        # Always produce neighbor-ranked results for the start node. When
        # `top_k is None` return all neighbors sorted by energy score; when an
        # integer `top_k` is provided return that many entries.
        neighbors = _neighbors_for_key(adj, node)
        if not neighbors:
            return []
        scores = []
        for v, edge in neighbors:
            grad = _get_edge_field(edge, 'energy_gradient')
            if grad is None:
                s = float(_get_edge_field(edge, 'target_energy') or _get_edge_field(edge, 'local_energy') or 0.0)
            else:
                s = float(_np.linalg.norm(_np.asarray(grad, dtype=float)))
            scores.append(max(0.0, s))
        entries = []
        for (v, edge), sc in zip(neighbors, scores):
            v_str = str(v)
            edge_src_meta = (edge.get('source_metadata') if isinstance(edge, dict) else None) or _get_by_string_key(matrix_meta, node)
            edge_tgt_meta = (edge.get('target_metadata') if isinstance(edge, dict) else None) or _get_by_string_key(matrix_meta, v_str)
            entries.append({
                'from': node,
                'to': v_str,
                'score': float(sc),
                'edge': edge,
                'source_metadata': edge_src_meta,
                'target_metadata': edge_tgt_meta,
                'source_preview': _preview_snippet(edge_src_meta),
                'target_preview': _preview_snippet(edge_tgt_meta),
                'projection_diagnostics': proj_dists
            })
        entries_sorted = sorted(entries, key=lambda x: x.get('score', 0.0), reverse=True)
        if top_k is None:
            return entries_sorted
        return entries_sorted[:top_k]
        for _ in range(steps):
            neighbors = _neighbors_for_key(adj, node)
            if not neighbors:
                break
            scores = []
            for v, edge in neighbors:
                grad = _get_edge_field(edge, 'energy_gradient')
                if grad is None:
                    s = float(_get_edge_field(edge, 'target_energy') or _get_edge_field(edge, 'local_energy') or 0.0)
                else:
                    s = float(_np.linalg.norm(_np.asarray(grad, dtype=float)))
                scores.append(max(0.0, s))
            if all(s == 0.0 for s in scores):
                break
            if deterministic:
                idx = int(_np.argmax(scores))
            else:
                exps = _np.exp(_np.asarray(scores) - max(scores))
                probs = exps / (exps.sum() + 1e-12)
                idx = int(_np.random.choice(len(neighbors), p=probs))
            chosen = neighbors[idx]
            v_str = str(chosen[0])
            edge = chosen[1]
            edge_src_meta = (edge.get('source_metadata') if isinstance(edge, dict) else None) or _get_by_string_key(matrix_meta, node)
            edge_tgt_meta = (edge.get('target_metadata') if isinstance(edge, dict) else None) or _get_by_string_key(matrix_meta, v_str)
            path.append({
                'from': node,
                'to': v_str,
                'score': float(scores[idx]),
                'edge': edge,
                'source_metadata': edge_src_meta,
                'target_metadata': edge_tgt_meta,
                'source_preview': _preview_snippet(edge_src_meta),
                'target_preview': _preview_snippet(edge_tgt_meta),
                'projection_diagnostics': proj_dists
            })
            node = v_str
        return path
    except Exception:
        logger.exception("search_energy_flow failed")
        return []



def search_reciprocal(adj: Mapping, angle_threshold: float = 0.5, payload: Optional[Dict] = None) -> List[Dict]:
    try:
        out = []
        matrix_meta = payload.get('matrix_metadata', {}) if isinstance(payload, dict) else {}
        proj_points = payload.get('projected_points', {}) if isinstance(payload, dict) else {}
        proj_dists = payload.get('proj_dists', {}) if isinstance(payload, dict) else {}
        for u_str, lst in _iter_normalized_adj(adj):
            for v, edge in lst:
                v_str = str(v)
                ra = _get_edge_field(edge, 'reciprocal_angle')
                if ra is None:
                    continue
                rev_edges = [e2 for vv, e2 in _neighbors_for_key(adj, v_str) if str(vv) == u_str]
                if not rev_edges:
                    continue
                rev_ra = _get_edge_field(rev_edges[0], 'reciprocal_angle')
                if rev_ra is None:
                    continue
                try:
                    ok = float(ra) < float(angle_threshold) and float(rev_ra) < float(angle_threshold)
                except Exception:
                    ok = False
                if ok:
                    rev = rev_edges[0]
                    edge_src_meta = (edge.get('source_metadata') if isinstance(edge, dict) else None) or _get_by_string_key(matrix_meta, u_str)
                    edge_tgt_meta = (edge.get('target_metadata') if isinstance(edge, dict) else None) or _get_by_string_key(matrix_meta, v_str)
                    rev_src_meta = (rev.get('source_metadata') if isinstance(rev, dict) else None) or _get_by_string_key(matrix_meta, v_str)
                    rev_tgt_meta = (rev.get('target_metadata') if isinstance(rev, dict) else None) or _get_by_string_key(matrix_meta, u_str)
                    out.append({
                        'src': u_str,
                        'tgt': v_str,
                        'reciprocal_angle': float(ra),
                        'reciprocal_angle_rev': float(rev_ra),
                        'edge': edge,
                        'rev_edge': rev,
                        'source_metadata': edge_src_meta,
                        'target_metadata': edge_tgt_meta,
                        'source_preview': _preview_snippet(edge_src_meta),
                        'target_preview': _preview_snippet(edge_tgt_meta),
                        'rev_source_metadata': rev_src_meta,
                        'rev_target_metadata': rev_tgt_meta,
                        'rev_source_preview': _preview_snippet(rev_src_meta),
                        'rev_target_preview': _preview_snippet(rev_tgt_meta),
                        'projection_diagnostics': proj_dists
                    })
        return out
    except Exception:
        logger.exception("search_reciprocal failed")
        return []


def setup_ingest_from_payload(payload: Dict, dataset_id: str = "samples"):
    if load_payload is None:
        raise RuntimeError("load_payload not available")
    ingest = load_payload(payload, dataset_id=dataset_id)
    features = ingest.features
    chunk_meta = list(ingest.chunk_metadata)
    return ingest, features, chunk_meta


def relationship_tier_connection(connection: List[Any]):
    try:
        vectors = []
        for v in connection:
            if np is None:
                raise RuntimeError("Numpy not available")
            arr = np.asarray(v, dtype=float)
            norm = np.linalg.norm(arr)
            vectors.append(arr / norm if norm > 0 else arr)
        vectors = np.array(vectors)
        if vectors.size == 0:
            return np.array([[]]), np.array([])
        R = np.dot(vectors, vectors.T)
        tiers = np.where(R > 0.7, 1, np.where(R > 0.4, 2, 3))
        return R, tiers
    except Exception:
        logger.exception("relationship_tier_connection failed")
        if np is not None:
            return np.array([[]]), np.array([])
        return [], []




if __name__ == '__main__':
    import argparse
    import sys
    import os

    def _dump_json(obj, fp):
        json.dump(obj, fp, indent=2, ensure_ascii=False, default=str)

    def _parse_vector(s: str) -> List[float]:
        if not s:
            return []
        parts = [p.strip() for p in s.split(',') if p.strip()]
        return [float(x) for x in parts]

    parser = argparse.ArgumentParser(description='Tooling for inspecting hyperdimensional connections')
    parser.add_argument('-i', '--input', default='connections_linked.json', help='Input connections JSON (default: connections_linked.json)')
    parser.add_argument('-m', '--mode', choices=['entity_search', 'directional', 'energy_flow', 'reciprocal', 'discover', 'all'], default='all', help='Operation to run')
    parser.add_argument('-o', '--output', default=None, help='Output file (defaults to <mode>.json or multiple files for "all")')

    # Common options
    parser.add_argument('--node', help='Node id or label for node-based searches')
    parser.add_argument('-k', type=int, default=5, help='Top-k for entity_search or other top-k defaults')

    # Directional (only --target is needed; --node is common)
    parser.add_argument('--target', required=True, help='Target node id or label for directional search')

    # Energy flow
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic choice for energy flow')

    # Reciprocal
    parser.add_argument('--angle_threshold', type=float, default=0.5, help='Angle threshold for reciprocal search')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.warning('Input file %s not found; proceeding with empty payload', args.input)
        payload = {}
    else:
        try:
            payload = load_precomputed_connections(str(input_path))
        except Exception:
            logger.exception('Failed to load input payload; using empty payload')
            payload = {}

    outputs = {}
    def write_result(name: str, data):
        out_path = args.output
        if out_path is None:
            out_path = f'{name}.json'
        # If running multiple modes, append mode name
        if args.mode == 'all' and args.output:
            base, ext = os.path.splitext(args.output)
            out_path = f"{base}_{name}{ext or '.json'}"
        with open(out_path, 'w', encoding='utf-8') as fp:
            _dump_json(data, fp)
        print(f'Wrote {out_path}')

    # Helper to run discover on a precomputed payload by creating a fake MT
    def run_discover_from_payload(payload_map: dict):
        try:
            class _FakeTraverse:
                def __init__(self, payload):
                    self.payload = payload
                def adjacency(self):
                    adj = {}
                    conns = (payload.get('connections') if isinstance(payload, dict) else {}) or {}
                    for src, lst in conns.items():
                        adj[src] = []
                        for e in (lst or []):
                            tgt = e.get('target_idx') or e.get('target_element_index') or e.get('target')
                            adj[src].append((tgt, e))
                    return adj

            class _FakeMT:
                def __init__(self, payload):
                    self._traverse_graph = _FakeTraverse(payload)

            fake_mt = _FakeMT(payload_map)
            return discover_connections(fake_mt)
        except Exception:
            logger.exception('run_discover_from_payload failed')
            return {}

    modes_to_run = [args.mode] if args.mode != 'all' else ['entity_search', 'directional', 'energy_flow', 'reciprocal', 'discover']

    for mode in modes_to_run:
        if mode == 'entity_search':
            if not args.node:
                print('entity_search requires --node', file=sys.stderr); sys.exit(2)
            res = entity_search(payload, node_id=args.node, k=args.k)
            write_result('entity_search', res)
            outputs['entity_search'] = res

        if mode == 'directional':
            # directional mode now only requires a start node and a target node.
            if not args.node or not args.target:
                print('directional requires --node and --target', file=sys.stderr); sys.exit(2)
            res = search_directional(
                payload.get('connections', {}),
                start_node=args.node,
                target_node=args.target,
                payload=payload
            )
            write_result('search_directional', res)
            outputs['directional'] = res

        elif mode == 'energy_flow':
            if not args.node:
                print('energy_flow requires --node', file=sys.stderr); sys.exit(2)
            res = search_energy_flow(payload.get('connections', {}), start_node=args.node, steps=args.steps, deterministic=args.deterministic, payload=payload, top_k=(None if args.top_k < 0 else args.top_k))
            write_result('search_energy_flow', res)
            outputs['energy_flow'] = res

        elif mode == 'reciprocal':
            res = search_reciprocal(payload.get('connections', {}), angle_threshold=args.angle_threshold, payload=payload)
            write_result('search_reciprocal', res)
            outputs['reciprocal'] = res

        elif mode == 'discover':
            # If a precomputed payload was loaded, run discover via fake MT
            if payload:
                res = run_discover_from_payload(payload)
            else:
                # else try to call discover_connections with MatrixTransformer if available
                if MatrixTransformer is None:
                    print('No payload and MatrixTransformer not available; cannot run discover', file=sys.stderr); res = {}
                else:
                    try:
                        mt = MatrixTransformer()
                        res = discover_connections(mt)
                    except Exception:
                        logger.exception('discover using MatrixTransformer failed')
                        res = {}
            write_result('search_discover_connections', res)
            outputs['discover'] = res

    # Summary
    print('Completed modes:', ', '.join(modes_to_run))
