import math
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Any
import numpy as np

def build_adjacency(connections: Dict[Any, List[dict]]) -> Dict[int, List[Tuple[int, dict]]]:
    """Convert connections mapping to adjacency dict with integer node ids."""
    out = {}
    for src, edges in connections.items():
        try:
            src_i = int(src)
        except Exception:
            src_i = src
        out[src_i] = []
        for e in edges:
            tgt = e.get("target_idx")
            try:
                tgt_i = int(tgt)
            except Exception:
                tgt_i = tgt
            out[src_i].append((tgt_i, e))
    return out


def _bfs_components(adj: Dict[int, List[Tuple[int, dict]]]) -> List[List[int]]:
    visited = set()
    comps = []
    for node in adj:
        if node in visited:
            continue
        dq = deque([node])
        comp = []
        visited.add(node)
        while dq:
            u = dq.popleft()
            comp.append(u)
            for v, _ in adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    dq.append(v)
        comps.append(comp)
    return comps


def _histogram(values: List[float], bins: int = 10) -> dict:
    try:
        a = np.asarray(values, dtype=float)
        if a.size == 0:
            return {"bins": [], "counts": []}
        counts, edges = np.histogram(a[~np.isnan(a)], bins=bins)
        return {"bins": edges.tolist(), "counts": counts.tolist()}
    except Exception:
        return {"bins": [], "counts": []}


def get_top_k(adj: Dict[int, List[Tuple[int, dict]]], top_k: int = 5,
              alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1) -> Dict[int, List[dict]]:
    """Compute top-k neighbors per node using combined score (strength, ratio, geodesic_error).

    Returns a mapping node -> list of enriched dicts with combined_score and novelty.
    """
    # gather global arrays for normalization
    strengths, ratios, geods = [], [], []
    for src, lst in adj.items():
        for tgt, e in lst:
            try:
                strengths.append(float(e.get("strength", 0.0)))
            except Exception:
                strengths.append(0.0)
            try:
                ratios.append(float(e.get("ratio", 0.0)))
            except Exception:
                ratios.append(0.0)
            try:
                geods.append(float(e.get("geodesic_error", 0.0)))
            except Exception:
                geods.append(0.0)

    def _minmax(arr):
        a = np.asarray(arr, dtype=float) if len(arr) else np.array([0.0])
        lo = float(np.nanmin(a))
        hi = float(np.nanmax(a))
        if np.isclose(hi, lo):
            hi = lo + 1.0
        return lo, hi

    s_lo, s_hi = _minmax(strengths)
    r_lo, r_hi = _minmax(np.log1p(ratios) if len(ratios) else [0.0])
    g_lo, g_hi = _minmax(geods)

    def _norm(v, lo, hi):
        try:
            val = float(v)
            return max(0.0, min(1.0, (val - lo) / (hi - lo)))
        except Exception:
            return 0.0

    topk_map = {}
    for src, lst in adj.items():
        enriched = []
        for tgt, e in lst:
            strength = float(e.get("strength", 0.0)) if e.get("strength") is not None else 0.0
            ratio = float(e.get("ratio", 0.0)) if e.get("ratio") is not None else 0.0
            geod = float(e.get("geodesic_error", 0.0)) if e.get("geodesic_error") is not None else 0.0

            sn = _norm(strength, s_lo, s_hi)
            rn = _norm(np.log1p(ratio), r_lo, r_hi)
            gn = _norm(geod, g_lo, g_hi)

            combined = alpha * sn + beta * rn - gamma * gn
            combined = max(0.0, min(1.0, combined))

            # novelty: high ratio, high geodesic, low strength
            novelty = 0.6 * rn + 0.3 * gn + 0.1 * (1.0 - sn)

            # Preserve original edge fields (including metadata) while adding
            # our computed scores. This prevents losing `source_metadata`/
            # `target_metadata` when consumers call get_top_k.
            extra = {k: v for k, v in (e.items() if isinstance(e, dict) else []) if k not in (
                'strength', 'ratio', 'geodesic_error', 'target_idx')}
            enriched.append({
                "target_idx": int(tgt) if isinstance(tgt, (int, np.integer)) else tgt,
                "strength": float(strength),
                "ratio": float(ratio),
                "geodesic_error": float(geod),
                "combined_score": float(combined),
                "novelty": float(novelty),
                **extra,
            })

        sorted_e = sorted(enriched, key=lambda x: x.get("combined_score", 0.0), reverse=True)
        topk_map[int(src)] = sorted_e[:top_k]

    return topk_map


def compute_hyperconnection_metrics(hd_payload: dict, top_k: int = 5) -> dict:
    """High level metrics collector returning a JSON-serializable dict.

    Contains: num_nodes, num_edges, avg_degree, degree_hist, top_k, reciprocity_rate,
    histograms (ratio/geodesic), components, maybe modularity and centralities if networkx installed.
    """
    connections = hd_payload.get("connections", {})
    adj = build_adjacency(connections)

    # basic counts
    num_nodes = len(adj)
    num_edges = sum(len(lst) for lst in adj.values())
    degree_hist = {n: len(lst) for n, lst in adj.items()}
    avg_degree = float(np.mean(list(degree_hist.values()))) if degree_hist else 0.0

    # top_k map
    top_k_map = get_top_k(adj, top_k=top_k)

    # reciprocity
    rev_index = defaultdict(list)
    ratios = []
    geods = []
    curvature = []
    local_energy = []
    target_energy = []
    for src, lst in adj.items():
        for tgt, e in lst:
            rev_index[(src, tgt)].append(e)
            try:
                ratios.append(float(e.get("ratio", 0.0)))
            except Exception:
                pass
            try:
                geods.append(float(e.get("geodesic_error", 0.0)))
            except Exception:
                pass
            try:
                curvature.append(float(e.get("local_curvature", 0.0)))
            except Exception:
                pass
            try:
                local_energy.append(float(e.get("local_energy", 0.0)))
            except Exception:
                pass
            try:
                target_energy.append(float(e.get("target_energy", 0.0)))
            except Exception:
                pass

    mutual = 0
    for (s, t) in list(rev_index.keys()):
        if (t, s) in rev_index:
            mutual += 1
    reciprocity_rate = float(mutual) / float(num_edges) if num_edges > 0 else 0.0

    ratio_hist = _histogram(ratios)
    geodesic_hist = _histogram(geods)
    curvature_hist = _histogram(curvature)
    local_energy_hist = _histogram(local_energy)
    target_energy_hist = _histogram(target_energy)

    # components
    # convert to adjacency lists of ints for BFS
    comp_adj = {k: [(t, e) for (t, e) in v] for k, v in adj.items()}
    components = _bfs_components(comp_adj)
    component_sizes = [len(c) for c in components]

    # try networkx extras
    modularity = None
    communities = []
    clustering_coef = None
    centralities = {}
    graph_density = None
    avg_shortest_path = None
    assortativity = None
    try:
        import networkx as nx
        G = nx.DiGraph()
        for s, lst in adj.items():
            for t, e in lst:
                w = float(e.get("strength", 0.0)) if e.get("strength") is not None else 0.0
                G.add_edge(int(s), int(t), weight=w)

        N = G.number_of_nodes(); M = G.number_of_edges()
        graph_density = float(M) / float(max(1, N * (N - 1)))
        UG = G.to_undirected()
        clustering_coef = float(nx.average_clustering(UG, weight="weight")) if UG.number_of_nodes() > 0 else None

        try:
            pr = nx.pagerank_numpy(G, weight="weight") if G.number_of_nodes() > 0 else {}
        except Exception:
            pr = nx.pagerank(G, weight="weight") if G.number_of_nodes() > 0 else {}
        try:
            bet = nx.betweenness_centrality(G)
        except Exception:
            bet = {}
        try:
            clo = nx.closeness_centrality(G)
        except Exception:
            clo = {}

        centralities = {"pagerank": {int(k): float(v) for k, v in pr.items()},
                       "betweenness": {int(k): float(v) for k, v in bet.items()},
                       "closeness": {int(k): float(v) for k, v in clo.items()}}

        from networkx.algorithms import community as nx_comm
        comms = list(nx_comm.greedy_modularity_communities(UG, weight="weight"))
        communities = [list(c) for c in comms]
        try:
            modularity = float(nx_comm.modularity(UG, comms, weight="weight"))
        except Exception:
            modularity = None

        # average shortest path on largest component
        try:
            if UG.number_of_nodes() > 0:
                largest = max(nx.connected_components(UG), key=len)
                sub = UG.subgraph(largest)
                length = dict(nx.all_pairs_dijkstra_path_length(sub, weight=lambda u, v, d: 1.0 / (d.get("weight", 1e-6) + 1e-6)))
                vals = []
                for u, m in length.items():
                    for v, l in m.items():
                        if u != v:
                            vals.append(l)
                avg_shortest_path = float(np.mean(vals)) if vals else None
        except Exception:
            avg_shortest_path = None

        # assortativity by label if matrix_metadata provided
        try:
            md = hd_payload.get("matrix_metadata", {})
            node_dataset = {int(k): v.get("dataset_id") for k, v in md.items()}
            if node_dataset:
                label_map = {}
                for n, lab in node_dataset.items():
                    if lab not in label_map:
                        label_map[lab] = len(label_map)
                for n in list(G.nodes()):
                    G.nodes[n]["label"] = label_map.get(node_dataset.get(int(n)))
                assortativity = float(nx.attribute_assortativity_coefficient(UG, "label"))
        except Exception:
            assortativity = None

    except Exception:
        pass

    metrics = {
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "avg_degree": float(avg_degree),
        "degree_hist": degree_hist,
        "top_k": top_k_map,
        "reciprocity_rate": float(reciprocity_rate),
        "ratio_hist": ratio_hist,
        "geodesic_error_hist": geodesic_hist,
        "curvature_hist": curvature_hist,
        "local_energy_hist": local_energy_hist,
        "target_energy_hist": target_energy_hist,
        "num_components": int(len(components)),
        "component_sizes": component_sizes,
        "largest_component": int(max(component_sizes) if component_sizes else 0),
        "graph_density": graph_density,
        "clustering_coefficient": clustering_coef,
        "modularity": (float(modularity) if modularity is not None else None),
        "num_communities": int(len(communities)) if communities else 0,
        "communities": communities[:50],
        "centralities": centralities,
        "avg_shortest_path": (float(avg_shortest_path) if avg_shortest_path is not None else None),
        "assortativity": (float(assortativity) if assortativity is not None else None),
    }

    # Content preview statistics: summarize presence and lengths of previews
    try:
        preview_stats = {
            'total_with_source_preview': 0,
            'total_with_target_preview': 0,
            'avg_source_preview_length': 0.0,
            'avg_target_preview_length': 0.0,
            'sample_source_previews': [],
            'sample_target_previews': [],
        }
        src_lens = []
        tgt_lens = []
        for src, lst in adj.items():
            for tgt, e in lst:
                if isinstance(e, dict):
                    src_meta = e.get('source_metadata')
                    if isinstance(src_meta, dict) and src_meta.get('content_preview') is not None:
                        p = str(src_meta.get('content_preview'))
                        preview_stats['total_with_source_preview'] += 1
                        src_lens.append(len(p))
                        if len(preview_stats['sample_source_previews']) < 5:
                            preview_stats['sample_source_previews'].append({
                                'chunk_index': src_meta.get('chunk_index'),
                                'preview_snippet': p,
                                'byte_count': src_meta.get('byte_count'),
                                'element_count': src_meta.get('element_count'),
                            })
                    tgt_meta = e.get('target_metadata')
                    if isinstance(tgt_meta, dict) and tgt_meta.get('content_preview') is not None:
                        p = str(tgt_meta.get('content_preview'))
                        preview_stats['total_with_target_preview'] += 1
                        tgt_lens.append(len(p))
                        if len(preview_stats['sample_target_previews']) < 5:
                            preview_stats['sample_target_previews'].append({
                                'chunk_index': tgt_meta.get('chunk_index'),
                                'preview_snippet': p,
                                'byte_count': tgt_meta.get('byte_count'),
                                'element_count': tgt_meta.get('element_count'),
                            })
        if src_lens:
            preview_stats['avg_source_preview_length'] = float(np.mean(src_lens))
        if tgt_lens:
            preview_stats['avg_target_preview_length'] = float(np.mean(tgt_lens))
        metrics['content_preview_stats'] = preview_stats
    except Exception:
        # non-fatal; metrics should still return
        pass

    novelty_edges = []
    # collect top novelty from top_k map
    for s, lst in top_k_map.items():
        for e in lst:
            novelty_edges.append((s, e.get("target_idx"), float(e.get("novelty", 0.0))))
    novelty_edges_sorted = sorted(novelty_edges, key=lambda x: x[2], reverse=True)[:50]
    metrics["top_novelty_edges"] = [{"src": int(s), "tgt": int(t), "novelty": float(n)} for s, t, n in novelty_edges_sorted]

    return metrics
