import pandas as pd
import json
import sys
import os
from pathlib import Path
import uuid
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('convertto')


# Helper function for cleaning and sanitizing strings
def sanitize_string(s):
    """Sanitize a string to ensure it can be properly encoded/decoded"""
    if s is None:
        return None
    
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return repr(s)
            
    # Replace problematic characters that might cause encoding issues
    try:
        # Test if the string can be encoded/decoded properly
        s.encode('utf-8').decode('utf-8')
        return s
    except UnicodeError:
        # If there's an encoding issue, replace problematic characters
        return s.encode('utf-8', errors='replace').decode('utf-8', errors='replace')


# Local safe JSON helpers (replace external `json_handler` dependency)
def safe_json_dumps(obj):
    """Serialize an object to a JSON string using safe defaults.
    Falls back to string conversion for non-serializable objects.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        try:
            # As a last resort, coerce to string representation
            return json.dumps(str(obj), ensure_ascii=False)
        except Exception:
            # Give up and return an empty JSON array/object depending on type
            return 'null'


def safe_json_dump(obj, file_path):
    """Write JSON to a file using UTF-8 and safe defaults."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, default=str)
        return True
    except Exception:
        try:
            with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
                json.dump(obj, f, ensure_ascii=False, default=str)
            return True
        except Exception:
            return False


def safe_json_load(file_path):
    """Load JSON from file using the robust loader defined below."""
    return load_json_safe(file_path)


# Markdown / PDF helpers
def df_to_markdown(df, outdir="exports", filename="results.md", title="Connection Exploration Report", row_preview=100):
    Path(outdir).mkdir(exist_ok=True)
    md_file = Path(outdir) / filename
    try:
        with open(md_file, "w", encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            if df is None or df.empty:
                f.write("_No rows available_\n")
                return str(md_file)

            # If standard to_markdown exists, prefer it
            try:
                table_md = df.head(row_preview).to_markdown(index=False)
                f.write(table_md)
                f.write("\n")
            except Exception:
                # Fallback: simple pipe table
                cols = list(df.columns)
                f.write("| " + " | ".join(cols) + " |\n")
                f.write("| " + " | ".join(['---'] * len(cols)) + " |\n")
                for _, r in df.head(row_preview).iterrows():
                    vals = [str(r.get(c, '')) for c in cols]
                    f.write("| " + " | ".join(vals) + " |\n")
        print(f"Markdown written: {md_file}")
        return str(md_file)
    except Exception as e:
        print("Failed to write markdown:", e)
        return None



# Small helper to create filesystem-safe filenames from titles
def _sanitize_filename(s: str) -> str:
    """
    Create a filesystem-safe filename from any input string.
    Handles encoding issues and special characters.
    """
    import re
    
    # Handle None or non-string inputs
    if s is None:
        return "output"
    
    try:
        # Try to convert to string (might fail for complex objects)
        s = str(s)
    except Exception:
        return "output"
    
    try:
        # Try to fix potential encoding issues
        s = s.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        
        # replace common arrows and separators
        s = s.replace('→', '-').replace(' ', '_')
        
        # remove any character that's not alnum, underscore, dash, or dot
        s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
        
        # trim length
        return s[:200]
    except Exception as e:
        # If all else fails, return a safe default
        logger.warning(f"Failed to sanitize filename '{s[:20]}...': {e}")
        return f"output_{uuid.uuid4().hex[:8]}"


def safe_write_parquet(df, path):
    """Attempt to write a DataFrame to Parquet. If direct write fails due to complex/object columns,
    coerce object/list/dict columns to JSON strings and retry. Prints helpful messages on failure.
    """
    try:
        df.to_parquet(path)
        print(f"Parquet file created: {path}")
        return True
    except Exception as e:
        # try to coerce problematic object columns to JSON strings
        try:
            df2 = df.copy()
            
            # Special handling for "Row Index" column that's causing problems
            if "Row Index" in df2.columns:
                # Convert "Row Index" to string type to avoid integer conversion issues
                df2["Row Index"] = df2["Row Index"].astype(str)
            
            for col in df2.columns:
                if df2[col].dtype == object:
                    def _coerce(v):
                        try:
                            # keep None/NaN as-is
                            if v is None or (isinstance(v, float) and pd.isna(v)):
                                return None
                            if isinstance(v, (dict, list)):
                                try:
                                    return safe_json_dumps(v)
                                except Exception:
                                    return json.dumps(v, ensure_ascii=False, default=str)
                            if isinstance(v, (int, float, bool, str)):
                                return v
                            # For complex objects, try to convert safely
                            try:
                                # Try direct string conversion
                                return str(v)
                            except Exception:
                                # Fall back to repr if str() fails
                                return repr(v)
                        except Exception:
                            return str(v)
                    df2[col] = df2[col].apply(_coerce)

            # Check for any empty strings in numeric columns that might cause issues
            for col in df2.columns:
                if df2[col].dtype == object:
                    # Safely check for numeric strings and replace empty strings with None
                    # This avoids using .str accessor on non-string data
                    is_numeric_col = pd.to_numeric(df2[col], errors='coerce').notna().any()
                    if is_numeric_col:
                        df2[col] = df2[col].replace('', None)
            
            df2.to_parquet(path)
            print(f"Parquet file created after coercion: {path}")
            return True
        except Exception as e2:
            print(f"Could not write parquet to {path} - Error: {str(e2)}")
            print("Problematic columns might include:")
            for col in df.columns:
                try:
                    # Try to identify problematic columns by checking for mixed types
                    if df[col].dtype == object:
                        types = set(type(x) for x in df[col].dropna().values)
                        if len(types) > 1:
                            print(f"  - Column '{col}' has mixed types: {', '.join(str(t) for t in types)}")
                except Exception:
                    pass
            return False


# Load the JSON data with robust error handling
def load_json_safe(file_path):
    # First try with UTF-8 encoding (most common)
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Fall back to different encodings
        for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
                
        # Last resort: read as binary and replace invalid chars
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                text = content.decode('utf-8', errors='replace')
                return json.loads(text)
        except Exception as e:
            print(f"Failed to load JSON with all encodings: {e}")
            raise

def main():
    """Main entry point of the script with robust error handling."""
    # Create the exports directory if it doesn't exist
    os.makedirs("exports", exist_ok=True)
    
    # Determine the path to results.json - look in current directory or use first argument
    results_path = "results.json"
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        results_path = sys.argv[1]
    elif not os.path.exists(results_path):
        # Look in common subdirectories
        for path in ["./connection_results/results.json", "../connection_results/results.json"]:
            if os.path.exists(path):
                results_path = path
                break

    logger.info(f"Loading data from: {results_path}")
    
    if not os.path.exists(results_path):
        logger.error(f"Error: Could not find {results_path}")
        print(f"Error: Could not find {results_path}")
        print("Please specify the path to results.json as the first argument or ensure it exists in the current directory.")
        return 1

    try:
        # Load JSON using the local safe loader
        data = safe_json_load(results_path)
            
        logger.info(f"Successfully loaded JSON data ({len(str(data))} characters)")
        # Call run_conversion with the loaded data
        run_conversion(data)
        return data
        
    except Exception as e:
        logger.error(f"Error loading {results_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        print(f"Error loading {results_path}: {e}")
        return None

def run_conversion(data):
    """
    Takes the loaded JSON data and runs all the conversion and export steps.
    """
    if data is None:
        logger.error("No data provided for conversion. Aborting.")
        print("Error: No data provided for conversion. Aborting.")
        return

    # Compute a descriptive base name for exported files based on the JSON structure
    def _compute_base_name(data):
        if not isinstance(data, dict):
            return 'results'
        if "results" in data:
            return _sanitize_filename(data.get('search_query') or 'search_results')
        if "dataset_info" in data:
            return _sanitize_filename(data.get('analysis_type') or 'dataset_connections')
        if "source_dataset" in data and "target_dataset" in data:
            s = (data.get('source_dataset') or {}).get('file_name') or (data.get('source_dataset') or {}).get('file_path') or 'source'
            t = (data.get('target_dataset') or {}).get('file_name') or (data.get('target_dataset') or {}).get('file_path') or 'target'
            return _sanitize_filename(f"{s}_to_{t}")
        if "bridge_datasets" in data:
            return _sanitize_filename(data.get('exploration_type') or 'bridge_analysis')
        return 'results'


    base_name = _compute_base_name(data)

    # Convert the data to a DataFrame with sensible flattening and summaries for nested fields
    def _coerce_key_lookup(mapping, key):
        if mapping is None:
            return None
        if key in mapping:
            return mapping[key]
        ks = str(key)
        if ks in mapping:
            return mapping[ks]
        try:
            ki = int(key)
            if ki in mapping:
                return mapping[ki]
        except Exception:
            pass
        return None

    records = []
    if "results" in data:
        # Handle graph_roadmap.json structure: flatten each result and add small summaries
        for r in data.get("results", []):
            # Flatten nested dicts (lists remain lists)
            flat_df = pd.json_normalize(r, sep='.')
            rec = flat_df.to_dict(orient='records')[0] if not flat_df.empty else {}

            # Summarize common nested lists
            if isinstance(r.get('entity_matches'), list):
                rec['entity_matches_count'] = len(r['entity_matches'])
            if isinstance(r.get('contextual_info'), dict):
                rec['file_size_bytes'] = r['contextual_info'].get('file_size_bytes')
                rec['total_elements'] = r['contextual_info'].get('total_elements')

            records.append(rec)
    elif "dataset_info" in data and "semantic_connections" in data:
        # Handle connections.json structure: flatten each dataset_info entry and summarize semantic connections
        dataset_info = data.get('dataset_info', {})
        semantic_map = data.get('semantic_connections', {})
        for key, dataset in dataset_info.items():
            flat_df = pd.json_normalize(dataset, sep='.')
            rec = flat_df.to_dict(orient='records')[0] if not flat_df.empty else {}
            rec['dataset_index'] = key

            # Look up semantic connections for this dataset index (keys may be str or int)
            connections = _coerce_key_lookup(semantic_map, key) or []
            rec['semantic_connections_count'] = len(connections)

            # Add a lightweight summary for CSV: target names and top relevance
            try:
                rec['semantic_connection_targets'] = [c.get('target_name') for c in connections if isinstance(c, dict)]
            except Exception:
                rec['semantic_connection_targets'] = None

            try:
                rec['semantic_connection_relevance'] = [c.get('relevance_score') for c in connections if isinstance(c, dict)]
                if rec['semantic_connection_relevance']:
                    rec['top_relevance_score'] = max(rec['semantic_connection_relevance'])
                else:
                    rec['top_relevance_score'] = None
            except Exception:
                rec['top_relevance_score'] = None
                rec['semantic_connection_relevance'] = None

            # compute averages / summaries across connections
            relevance_scores = [c.get('relevance_score') for c in connections if isinstance(c, dict) and c.get('relevance_score') is not None]
            if relevance_scores:
                rec['avg_relevance_score'] = sum(relevance_scores) / len(relevance_scores)
            else:
                rec['avg_relevance_score'] = None

            # average contextual_bridges length
            bridge_counts = []
            semantic_coord_count = 0
            semantic_compatibilities = []
            for c in connections:
                if isinstance(c, dict):
                    bridge_counts.append(len(c.get('contextual_bridges') or []))
                    if c.get('semantic_coordinates'):
                        semantic_coord_count += 1
                    if c.get('contextual_semantic_compatibility') is not None:
                        semantic_compatibilities.append(c['contextual_semantic_compatibility'])

            rec['avg_contextual_bridges_per_connection'] = (sum(bridge_counts) / len(bridge_counts)) if bridge_counts else 0
            rec['connections_with_semantic_coordinates'] = semantic_coord_count
            rec['avg_contextual_semantic_compatibility'] = (sum(semantic_compatibilities) / len(semantic_compatibilities)) if semantic_compatibilities else None

            # expose some contextual_info quick fields if present
            ci = dataset.get('contextual_info', {}) or {}
            rec['file_size_bytes'] = ci.get('file_size_bytes')
            rec['total_elements'] = ci.get('total_elements')
            rec['total_columns'] = ci.get('total_columns')

            # store a compact JSON representation of the full connections for later export if needed
            try:
                # Use the safe JSON serializer
                rec['semantic_connections_json'] = safe_json_dumps(connections)
            except Exception as e:
                rec['semantic_connections_json'] = f"Serialization error: {e}"
                logger.warning(f"Could not serialize semantic_connections for dataset {key}: {e}")

            records.append(rec)
    # New branch: single-pathway results (e.g., pathway_results.json)
    elif "source_dataset" in data and "target_dataset" in data:
        # Flatten a single pathway result into one record
        p = data
        rec = {}

        # Pathway-level fields
        rec['pathway_type'] = p.get('pathway_type')
        rec['confidence'] = p.get('confidence')
        
        # Add a more readable description
        rec['connection_summary'] = f"Connection from {p.get('source_dataset',{}).get('file_name', 'unknown')} to {p.get('target_dataset',{}).get('file_name', 'unknown')} with {len(p.get('contextual_connections') or [])} contextual connections and {len(p.get('shared_entities') or [])} shared entities"

        # Flatten attention scores
        attention = p.get('attention_scores', {}) or {}
        for k, v in attention.items():
            rec[f'attention_{k}'] = v

        # Counts and high-level lists
        rec['n_contextual_bridges'] = len(p.get('contextual_bridges') or [])
        # Count contextual connections (they may have 'entity_name' or 'description' fields)
        ctx_conns = [c for c in (p.get('contextual_connections') or []) if c and (c.get('entity_name') or c.get('description'))]
        rec['n_contextual_connections'] = len(ctx_conns)
        rec['n_shared_entities'] = len(p.get('shared_entities') or [])

        # Extract contextual connections details
        contextual_connections = p.get('contextual_connections') or []
        if contextual_connections:
            # Add first 5 connection descriptions as separate fields for better readability in CSV/markdown
            for i, conn in enumerate(contextual_connections[:5]):
                rec[f'contextual_connection_{i+1}'] = conn.get('description') or conn.get('entity_name')

        # Extract shared entities details
        shared_entities = p.get('shared_entities') or []
        if shared_entities:
            # Add first 5 shared entities as separate fields
            for i, entity in enumerate(shared_entities[:5]):
                rec[f'shared_entity_{i+1}'] = entity.get('entity_name')
        
        # Source dataset flattening
        src = p.get('source_dataset', {}) or {}
        rec['source_file_name'] = src.get('file_name')
        rec['source_index'] = src.get('index')
        s_shape = src.get('shape') if isinstance(src.get('shape'), list) else []
        rec['source_rows'] = s_shape[0] if len(s_shape) > 0 else None
        rec['source_cols'] = s_shape[1] if len(s_shape) > 1 else None
        src_ci = src.get('contextual_info', {}) or {}
        rec['source_file_size_bytes'] = src_ci.get('file_size_bytes')
        rec['source_total_elements'] = src_ci.get('total_elements')
        rec['source_total_columns'] = src_ci.get('total_columns')
        rec['source_key_entities_count'] = len(src.get('key_entities') or [])
        rec['source_entity_matches_count'] = len(src.get('entity_matches') or [])
        rec['source_cross_matches_count'] = len(src.get('cross_matches') or [])
        rec['source_local_relationships_count'] = len((src.get('local_relationships') or {}).get('significant_elements') or [])
        # semantic coords mean
        sc_nums = (src.get('semantic_coordinates') or {}).get('numerical') or []
        try:
            rec['source_semantic_coords_mean'] = (sum(sc_nums) / len(sc_nums)) if sc_nums else None
        except Exception:
            rec['source_semantic_coords_mean'] = None

        # Target dataset flattening
        tgt = p.get('target_dataset', {}) or {}
        rec['target_file_name'] = tgt.get('file_name')
        rec['target_index'] = tgt.get('index')
        t_shape = tgt.get('shape') if isinstance(tgt.get('shape'), list) else []
        rec['target_rows'] = t_shape[0] if len(t_shape) > 0 else None
        rec['target_cols'] = t_shape[1] if len(t_shape) > 1 else None
        tgt_ci = tgt.get('contextual_info', {}) or {}
        rec['target_file_size_bytes'] = tgt_ci.get('file_size_bytes')
        rec['target_total_elements'] = tgt_ci.get('total_elements')
        rec['target_total_columns'] = tgt_ci.get('total_columns')
        rec['target_key_entities_count'] = len(tgt.get('key_entities') or [])
        rec['target_entity_matches_count'] = len(tgt.get('entity_matches') or [])
        rec['target_cross_matches_count'] = len(tgt.get('cross_matches') or [])
        rec['target_local_relationships_count'] = len((tgt.get('local_relationships') or {}).get('significant_elements') or [])
        tc_nums = (tgt.get('semantic_coordinates') or {}).get('numerical') or []
        try:
            rec['target_semantic_coords_mean'] = (sum(tc_nums) / len(tc_nums)) if tc_nums else None
        except Exception:
            rec['target_semantic_coords_mean'] = None

        # JSON drilldown fields
        # Safe JSON serialization with encoding handling
        try:
            rec['contextual_bridges_json'] = safe_json_dumps(p.get('contextual_bridges') or [])
        except Exception as e:
            print(f"Warning: Failed to serialize contextual_bridges: {e}")
            rec['contextual_bridges_json'] = None

        try:
            rec['contextual_connections_json'] = safe_json_dumps(p.get('contextual_connections') or [])
        except Exception as e:
            print(f"Warning: Failed to serialize contextual_connections: {e}")
            rec['contextual_connections_json'] = None

        try:
            rec['pathway_metadata_json'] = safe_json_dumps(p.get('pathway_metadata') or {})
        except Exception as e:
            print(f"Warning: Failed to serialize pathway_metadata: {e}")
            rec['pathway_metadata_json'] = None

        records.append(rec)
    elif "bridge_datasets" in data:
        # Handle results.json (bridge_datasets) structure: one row per bridge_dataset with summaries
        for b in data.get('bridge_datasets', []):
            bridge = b.get('bridge_dataset', {}) or {}
            connections = b.get('connections', []) or []
            flat_df = pd.json_normalize(bridge, sep='.')
            rec = flat_df.to_dict(orient='records')[0] if not flat_df.empty else {}

            rec['bridge_score'] = b.get('bridge_score')
            rec['connectivity_count'] = b.get('connectivity_count', len(connections))

            # shape -> rows / cols
            shape = bridge.get('shape') if isinstance(bridge.get('shape'), list) else []
            rec['bridge_rows'] = shape[0] if len(shape) > 0 else None
            rec['bridge_cols'] = shape[1] if len(shape) > 1 else None

            # connection-level summaries
            strengths = [c.get('connection_strength') for c in connections if isinstance(c, dict) and c.get('connection_strength') is not None]
            rec['total_connections'] = len(connections)
            rec['avg_connection_strength'] = (sum(strengths) / len(strengths)) if strengths else None

            comps = []
            for c in connections:
                if isinstance(c, dict) and c.get('semantic_compatibility') is not None:
                    comps.append(c['semantic_compatibility'])
            rec['avg_semantic_compatibility'] = (sum(comps) / len(comps)) if comps else None

            rec['n_contextual_bridges'] = sum(len(c.get('contextual_bridges') or []) for c in connections if isinstance(c, dict))
            rec['n_semantic_connections'] = sum(len(c.get('semantic_connections') or []) for c in connections if isinstance(c, dict))

            # expose some contextual_info quick fields if present in bridge_analysis
            ci = b.get('bridge_analysis', {}).get('contextual_info', {}) or {}
            rec['file_size_bytes'] = ci.get('file_size_bytes')
            rec['total_elements'] = ci.get('total_elements')
            rec['total_columns'] = ci.get('total_columns')

            # store compact JSON for connections for drill-down
            try:
                rec['connections_json'] = safe_json_dumps(connections)
            except Exception as e:
                rec['connections_json'] = f"Serialization error: {e}"
                logger.warning(f"Could not serialize connections for bridge: {e}")

            records.append(rec)


    # Produce DataFrame
    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame()

    # Save as 
    Path("exports").mkdir(exist_ok=True)
    df.to_csv("exports/results.csv", index=False)
    print("CSV file created: exports/results.csv")

    # Save as Excel (if openpyxl is available)
    try:
        df.to_excel("exports/results.xlsx", index=False)
        print("Excel file created: exports/results.xlsx")
    except ImportError:
        print("openpyxl not available. Install with: pip install openpyxl")

    # Also write parquet (fast, columnar) into exports/ if possible
    Path("exports").mkdir(exist_ok=True)
    try:
        if not df.empty:
            safe_write_parquet(df, "exports/results.parquet")
        else:
            print("DataFrame empty - skipping results.parquet write")
    except Exception as e:
        print("Could not write parquet (install pyarrow or fastparquet).", str(e))

    # Graph export helpers: build a NetworkX graph from an edges DataFrame and write multiple formats
    def _build_graph_from_edges_df(edges_df, node_attrs=None):
        try:
            import networkx as nx
        except Exception:
            raise
        # Use MultiGraph to preserve parallel edges
        G = nx.MultiGraph()
        # helper: sanitize attribute values for graph writers
        def _sanitize_value(v):
            # skip missing values
            if pd.isna(v):
                return None
            try:
                # handle numpy types
                if 'numpy' in str(type(v)):
                    return v.item()
            except Exception:
                return None

            # primitive types are safe
            if isinstance(v, (int, float, bool, str)):
                return v

            # lists/dicts -> JSON string
            try:
                return json.dumps(v, default=str)
            except Exception:
                return str(v)

            # fallback to string representation
            try:
                return str(v)
            except Exception:
                return None
        for _, row in edges_df.iterrows():
            src = row.get('source')
            tgt = row.get('target')
            if src is None or tgt is None:
                continue
            # prepare attributes excluding source/target
            attrs = row.drop(labels=['source', 'target'], errors='ignore').to_dict()
            safe_attrs = {}
            for k, v in attrs.items():
                sv = _sanitize_value(v)
                if sv is not None:
                    # GEXF requires attribute keys to be strings
                    safe_attrs[str(k)] = sv

            # add nodes with attrs if available
            s = str(src)
            t = str(tgt)
            # add nodes with sanitized attributes if available
            def _sanitize_attrs_dict(attrs_dict):
                safe_dict = {}
                for k, v in attrs_dict.items():
                    sv = _sanitize_value(v)
                    if sv is not None:
                        safe_dict[str(k)] = sv
                return safe_dict

            if node_attrs and s in node_attrs:
                G.add_node(s, **_sanitize_attrs_dict(node_attrs[s]))
            else:
                G.add_node(s)
            if node_attrs and t in node_attrs:
                G.add_node(t, **_sanitize_attrs_dict(node_attrs[t]))
            else:
                G.add_node(t)

            G.add_edge(s, t, **safe_attrs)
        return G


    # Graph file writers removed. Use SQLite export (_write_sqlite) for structured graph output.


    # Build an edges DataFrame from any available connection-like fields across supported JSON shapes
    def _collect_edges_from_data(data):
        edges = []

        # results entries may contain contextual_connections / semantic_connections / connections
        if isinstance(data, dict) and 'results' in data:
            for r in data.get('results', []):
                src = r.get('dataset_name') or r.get('dataset_path') or None
                # merge possible connection lists
                for field in ('contextual_connections', 'semantic_connections', 'connections'):
                    for c in r.get(field, []) or []:
                        if not isinstance(c, dict):
                            continue
                        tgt = c.get('target_name') or c.get('dataset_name') or c.get('target') or c.get('target_dataset')
                        if not tgt:
                            continue
                        edges.append({
                            'source': src,
                            'target': tgt,
                            'source_index': r.get('dataset_index') if isinstance(r, dict) else None,
                            'relevance_score': c.get('relevance_score') or r.get('relevance_score'),
                            'connection_strength': c.get('connection_strength'),
                            'contextual_bridges_count': len(c.get('contextual_bridges') or []) if isinstance(c.get('contextual_bridges'), list) else None
                        })

        # dataset_info + semantic_connections (connections between datasets)
        if isinstance(data, dict) and 'dataset_info' in data and 'semantic_connections' in data:
            dataset_info = data.get('dataset_info', {})
            semantic_map = data.get('semantic_connections', {})
            for key, dataset in dataset_info.items():
                src = (dataset.get('file_info') or {}).get('file_name') or key
                connections = _coerce_key_lookup(semantic_map, key) or []
                for c in connections:
                    if not isinstance(c, dict):
                        continue
                    tgt = c.get('target_name') or c.get('target') or c.get('target_dataset') or c.get('target_index')
                    edges.append({
                        'source': src,
                        'target': tgt,
                        'source_index': key,
                        'target_index': c.get('target_index'),
                        'relevance_score': c.get('relevance_score'),
                        'connection_strength': c.get('connection_strength'),
                        'semantic_compatibility': (c.get('contextual_analysis') or {}).get('semantic_compatibility')
                    })

        # bridge_datasets contain a bridge_dataset and connections
        if isinstance(data, dict) and 'bridge_datasets' in data:
            for i, b in enumerate(data.get('bridge_datasets', []) or []):
                bridge = b.get('bridge_dataset', {}) or {}
                src = bridge.get('file_name') or bridge.get('file_path') or f'bridge_{i}'
                for c in b.get('connections', []) or []:
                    if not isinstance(c, dict):
                        continue
                    tgt = c.get('target_name') or c.get('dataset_name') or c.get('target')
                    edges.append({
                        'source': src,
                        'target': tgt,
                        'bridge_score': b.get('bridge_score'),
                        'connection_strength': c.get('connection_strength'),
                        'contextual_bridges_count': len(c.get('contextual_bridges') or []) if isinstance(c.get('contextual_bridges'), list) else None
                    })

        # single-pathway: create one edge between source and target datasets
        if isinstance(data, dict) and 'source_dataset' in data and 'target_dataset' in data:
            src = (data.get('source_dataset') or {}).get('file_name') or (data.get('source_dataset') or {}).get('file_path') or 'source'
            tgt = (data.get('target_dataset') or {}).get('file_name') or (data.get('target_dataset') or {}).get('file_path') or 'target'
            edges.append({
                'source': src,
                'target': tgt,
                'pathway_type': data.get('pathway_type'),
                'confidence': data.get('confidence'),
                'n_contextual_bridges': len(data.get('contextual_bridges') or []),
                'n_shared_entities': len(data.get('shared_entities') or [])
            })

        # coerce to DataFrame and drop incomplete rows
        if edges:
            edf = pd.DataFrame(edges)
            # drop rows missing source/target
            edf = edf.dropna(subset=['source', 'target'])
            return edf
        return pd.DataFrame()


    # Collect node-level attributes from the JSON so nodes can be annotated in the graph
    def _collect_node_attrs_from_data(data):
        attrs = {}
        if not isinstance(data, dict):
            return attrs

        # from results list
        if 'results' in data:
            for r in data.get('results', []):
                name = r.get('dataset_name') or r.get('dataset_path')
                if not name:
                    continue
                ci = r.get('contextual_info', {}) or {}
                attrs[str(name)] = {
                    'file_size_bytes': ci.get('file_size_bytes'),
                    'total_elements': ci.get('total_elements'),
                    'total_columns': ci.get('total_columns'),
                    'data_type': r.get('data_type')
                }

        # from dataset_info
        if 'dataset_info' in data and isinstance(data.get('dataset_info'), dict):
            for key, ds in data.get('dataset_info', {}).items():
                fi = (ds.get('file_info') or {})
                name = fi.get('file_name') or key
                ci = ds.get('contextual_info', {}) or {}
                attrs[str(name)] = {
                    'file_size_bytes': ci.get('file_size_bytes'),
                    'total_elements': ci.get('total_elements'),
                    'total_columns': ci.get('total_columns'),
                    'data_type': fi.get('data_type')
                }

        # bridge datasets
        if 'bridge_datasets' in data:
            for i, b in enumerate(data.get('bridge_datasets', []) or []):
                bridge = b.get('bridge_dataset', {}) or {}
                name = bridge.get('file_name') or bridge.get('file_path') or f'bridge_{i}'
                ci = (b.get('bridge_analysis') or {}).get('contextual_info', {}) or {}
                attrs[str(name)] = {
                    'file_size_bytes': ci.get('file_size_bytes'),
                    'total_elements': ci.get('total_elements'),
                    'total_columns': ci.get('total_columns'),
                    'data_type': bridge.get('data_type')
                }

        # single pathway
        if 'source_dataset' in data and 'target_dataset' in data:
            s = (data.get('source_dataset') or {}).get('file_name') or (data.get('source_dataset') or {}).get('file_path') or 'source'
            t = (data.get('target_dataset') or {}).get('file_name') or (data.get('target_dataset') or {}).get('file_path') or 'target'
            s_ci = (data.get('source_dataset') or {}).get('contextual_info', {}) or {}
            t_ci = (data.get('target_dataset') or {}).get('contextual_info', {}) or {}
            attrs[str(s)] = {
                'file_size_bytes': s_ci.get('file_size_bytes'),
                'total_elements': s_ci.get('total_elements'),
                'total_columns': s_ci.get('total_columns'),
                'data_type': (data.get('source_dataset') or {}).get('data_type')
            }
            attrs[str(t)] = {
                'file_size_bytes': t_ci.get('file_size_bytes'),
                'total_elements': t_ci.get('total_elements'),
                'total_columns': t_ci.get('total_columns'),
                'data_type': (data.get('target_dataset') or {}).get('data_type')
            }

        return attrs


    # Attempt to collect edges and write graph formats (graphml/gexf/gml)
    try:
        edges_df = _collect_edges_from_data(data)
        # also prepare node table for export
        node_attrs = _collect_node_attrs_from_data(data)
        if node_attrs:
            nodes_df = pd.DataFrame.from_dict(node_attrs, orient='index')
            nodes_df = nodes_df.reset_index().rename(columns={'index': 'node'})
        else:
            nodes_df = pd.DataFrame()

        if not edges_df.empty:
            print("RDF export functionality has been removed. Use SQLite or other formats for structured data export.")
    except Exception as e:
        print("Failed to export graphs:", e)


    def _write_sqlite(outdir='exports', edges_df=None, nodes_df=None, matches_df=None):
        Path(outdir).mkdir(exist_ok=True)
        try:
            import sqlite3
        except Exception:
            print("sqlite3 not available in this Python build; skipping sqlite export")
            return
        db_path = Path(outdir) / "results.db"
        try:
            conn = sqlite3.connect(str(db_path))
            if nodes_df is not None and not nodes_df.empty:
                # Coerce all columns to string to avoid DB type errors with mixed/complex data
                nodes_df.astype(str).to_sql('nodes', conn, if_exists='replace', index=False)
            if edges_df is not None and not edges_df.empty:
                # Coerce all columns to string to avoid DB type errors with mixed/complex data
                edges_df.astype(str).to_sql('edges', conn, if_exists='replace', index=False)
            if matches_df is not None and not matches_df.empty:
                matches_df.astype(str).to_sql('entity_matches', conn, if_exists='replace', index=False)
        except Exception as e:
            print("Failed to create sqlite DB:", e)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # Export main results to markdown and pdf (if possible)
    try:
        if not df.empty:
            # choose a descriptive base name for exports depending on input type
            if "results" in data:
                base_name = data.get('search_query') or 'search_results'
            elif "dataset_info" in data:
                base_name = data.get('analysis_type') or 'dataset_connections'
            elif "source_dataset" in data and "target_dataset" in data:
                s = (data.get('source_dataset') or {}).get('file_name') or 's'
                t = (data.get('target_dataset') or {}).get('file_name') or 't'
                base_name = f"{s}_to_{t}"
            elif "bridge_datasets" in data:
                base_name = data.get('exploration_type') or 'bridge_analysis'
            else:
                base_name = 'results'

            base_name = _sanitize_filename(base_name)

            md_filename = f"{base_name}_results.md"
            pdf_filename = f"{base_name}_results.pdf"
        md = df_to_markdown(df, outdir='exports', filename=md_filename, title=f"Results - {base_name}")
    except Exception as e:
        print("Failed to export main results to markdown/pdf:", e)

    # Save as HTML with comprehensive formatting
    def dict_to_html(obj, indent=0):
        """Convert a dictionary/list to formatted HTML with enhanced handling for nested fields."""
        html = ""
        margin = "  " * indent
        special_keys = {"semantic_connections", "contextual_bridges", "semantic_coordinates", "contextual_analysis"}

        if isinstance(obj, dict):
            for key, value in obj.items():
                # Handle numeric keys by converting them to strings
                if isinstance(key, (int, float)):
                    key = str(key)

                if isinstance(value, (dict, list)) and value:
                    # Special-case very large or nested semantic/contextual fields: show a summary and a toggle
                    if isinstance(key, str) and key in special_keys:
                        uid = uuid.uuid4().hex
                        summary = f"{len(value)} items" if isinstance(value, list) else f"{len(value)} keys"
                        html += f"{margin}<div class=\"json-key\">{key}: <span class=\"value-summary\">({summary})</span> " \
                                f"<button class=\"toggle-button\" onclick=\"toggleSection('detail-{uid}')\">Show</button></div>\n"
                        html += f"{margin}<div id=\"detail-{uid}\" class=\"json-value collapsible\">\n"
                        html += dict_to_html(value, indent + 1)
                        html += f"{margin}</div>\n"
                    else:
                        html += f"{margin}<div class=\"json-key\">{key}:</div>\n"
                        html += f"{margin}<div class=\"json-value\">\n"
                        html += dict_to_html(value, indent + 1)
                        html += f"{margin}</div>\n"
                elif not value:
                    # Explicitly handle empty lists or dictionaries
                    html += f"{margin}<div class=\"json-item\"><span class=\"key\">{key}:</span> <span class=\"value\">(empty)</span></div>\n"
                else:
                    if isinstance(value, str) and len(value) > 100:
                        # Truncate very long strings for readability
                        display_value = value[:100] + "..." if len(value) > 100 else value
                    else:
                        display_value = value
                    # Sanitize the display value to avoid encoding issues
                try:
                    safe_value = sanitize_string(display_value)
                    html += f"{margin}<div class=\"json-item\"><span class=\"key\">{sanitize_string(key)}:</span> <span class=\"value\">{safe_value}</span></div>\n"
                except Exception:
                    # If string interpolation fails due to encoding issues, use a safer approach
                    html += f"{margin}<div class=\"json-item\"><span class=\"key\">{sanitize_string(key)}:</span> <span class=\"value\">[Complex Value]</span></div>\n"
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    html += f"{margin}<div class=\"json-array-item\">Item {i + 1}:</div>\n"
                    html += f"{margin}<div class=\"json-value\">\n"
                    html += dict_to_html(item, indent + 1)
                    html += f"{margin}</div>\n"
                elif not item:
                    # Explicitly handle empty items in lists
                    html += f"{margin}<div class=\"json-item\">• (empty)</div>\n"
                else:
                    html += f"{margin}<div class=\"json-item\">• {item}</div>\n"
        else:
            html += f"{margin}<div class=\"json-value\">{obj}</div>\n"

        return html

    # Create a summary table for quick overview
    summary_data = []
    if "results" in data:
        # Handle graph_roadmap.json structure
        for result in data['results']:
            # Format total_elements safely
            total_elements = result.get('contextual_info', {}).get('total_elements', 0)
            total_elements_str = str(total_elements)
            # Add commas as thousand separators manually
            if len(total_elements_str) > 3:
                total_elements_formatted = ""
                for i, char in enumerate(reversed(total_elements_str)):
                    if i > 0 and i % 3 == 0:
                        total_elements_formatted = "," + total_elements_formatted
                    total_elements_formatted = char + total_elements_formatted
            else:
                total_elements_formatted = total_elements_str
                
            summary_data.append({
                'Dataset': result['dataset_name'],
                'Data Type': result['data_type'],
                'Relevance Score': f"{result['relevance_score']:.3f}",
                'Entity Matches': len(result.get('entity_matches', [])),
                'File Size (MB)': f"{result.get('contextual_info', {}).get('file_size_bytes', 0) / (1024*1024):.1f}",
                'Total Elements': total_elements_formatted,
                'Columns': result.get('contextual_info', {}).get('total_columns', 'N/A')
            })
    elif "dataset_info" in data:
        # Handle connections.json structure
        for key, dataset in data['dataset_info'].items():
            # Format total_elements safely
            total_elements = dataset['contextual_info'].get('total_elements', 0)
            total_elements_str = str(total_elements)
            # Add commas as thousand separators manually
            if len(total_elements_str) > 3:
                total_elements_formatted = ""
                for i, char in enumerate(reversed(total_elements_str)):
                    if i > 0 and i % 3 == 0:
                        total_elements_formatted = "," + total_elements_formatted
                    total_elements_formatted = char + total_elements_formatted
            else:
                total_elements_formatted = total_elements_str
                
            summary_data.append({
                'Dataset': dataset['file_info']['file_name'],
                'Data Type': dataset['file_info']['data_type'],
                'Relevance Score': "N/A",  # Not available at this level
                'Entity Matches': "N/A",  # Not available at this level
                'File Size (MB)': f"{dataset['contextual_info'].get('file_size_bytes', 0) / (1024*1024):.1f}",
                'Total Elements': total_elements_formatted,
                'Columns': dataset['contextual_info'].get('total_columns', 'N/A')
            })
    # New: single-pathway summary (e.g., pathway_results.json)
    elif "source_dataset" in data and "target_dataset" in data:
        src = data.get('source_dataset', {}) or {}
        tgt = data.get('target_dataset', {}) or {}
        s_ci = src.get('contextual_info', {}) or {}
        t_ci = tgt.get('contextual_info', {}) or {}

        s_name = src.get('file_name') or src.get('file_path') or ''
        t_name = tgt.get('file_name') or tgt.get('file_path') or ''

        s_bytes = s_ci.get('file_size_bytes') or 0
        t_bytes = t_ci.get('file_size_bytes') or 0

        s_total = s_ci.get('total_elements') or 0
        t_total = t_ci.get('total_elements') or 0

        s_cols = s_ci.get('total_columns') if s_ci.get('total_columns') is not None else 'N/A'
        t_cols = t_ci.get('total_columns') if t_ci.get('total_columns') is not None else 'N/A'

        # total entity matches (source + target) when available
        s_em = len(src.get('entity_matches') or [])
        t_em = len(tgt.get('entity_matches') or [])

        # Format numbers safely without potential format specifier conflicts
        total_elements_str = str(s_total + t_total)
        # Add commas as thousand separators manually if needed
        if len(total_elements_str) > 3:
            total_elements_formatted = ""
            for i, char in enumerate(reversed(total_elements_str)):
                if i > 0 and i % 3 == 0:
                    total_elements_formatted = "," + total_elements_formatted
                total_elements_formatted = char + total_elements_formatted
        else:
            total_elements_formatted = total_elements_str

        summary_data.append({
            'Dataset': f"{s_name} → {t_name}",
            'Data Type': f"{src.get('data_type', '')}/{tgt.get('data_type', '')}",
            'Relevance Score': f"{data.get('confidence'):.3f}" if data.get('confidence') is not None else "N/A",
            'Entity Matches': s_em + t_em,
            'File Size (MB)': f"{(s_bytes / (1024*1024)):.1f} / {(t_bytes / (1024*1024)):.1f}",
            'Total Elements': total_elements_formatted,
            'Columns': f"{s_cols} / {t_cols}"
        })
    elif "bridge_datasets" in data:
        # Handle results.json bridge_datasets structure
        for b in data.get('bridge_datasets', []):
            bridge = b.get('bridge_dataset', {}) or {}
            ba = b.get('bridge_analysis', {}) or {}
            ci = ba.get('contextual_info', {}) or {}
            shape = bridge.get('shape') if isinstance(bridge.get('shape'), list) else []
            rows = shape[0] if len(shape) > 0 else ''
            cols = shape[1] if len(shape) > 1 else ''
            
            # Format total_elements safely
            total_elements = ci.get('total_elements', 0)
            total_elements_str = str(total_elements)
            # Add commas as thousand separators manually
            if len(total_elements_str) > 3:
                total_elements_formatted = ""
                for i, char in enumerate(reversed(total_elements_str)):
                    if i > 0 and i % 3 == 0:
                        total_elements_formatted = "," + total_elements_formatted
                    total_elements_formatted = char + total_elements_formatted
            else:
                total_elements_formatted = total_elements_str
                
            summary_data.append({
                'Dataset': bridge.get('file_name') or bridge.get('file_path') or '',
                'Data Type': bridge.get('data_type') or '',
                'Relevance Score': f"{b.get('bridge_score'):.3f}" if b.get('bridge_score') is not None else "N/A",
                'Entity Matches': b.get('connectivity_count', b.get('connectivity_count', 'N/A')),
                'File Size (MB)': f"{ci.get('file_size_bytes', 0) / (1024*1024):.1f}",
                'Total Elements': total_elements_formatted,
                'Columns': ci.get('total_columns', cols or 'N/A')
            })
    else:
        raise ValueError("Unsupported JSON structure. Ensure the file contains 'results', 'dataset_info', or 'bridge_datasets'.")

    summary_df = pd.DataFrame(summary_data)
    summary_table = summary_df.to_html(classes='table table-striped', 
                                      table_id='summary-table', 
                                      escape=False, 
                                      index=False,
                                      border=0)

    # Generate detailed HTML for each dataset
    detailed_html = ""
    if "results" in data:
        # Handle graph_roadmap.json structure
        for i, result in enumerate(data['results']):
            detailed_html += f"""
            <div class="dataset-section" id="dataset-{i}">
            <h2>{result['dataset_name']} (Relevance: {result['relevance_score']:.3f})</h2>
                <div class="dataset-content">
                    {dict_to_html(result)}
                </div>
            </div>
            """
    elif "dataset_info" in data:
        # Handle connections.json structure
        for key, dataset in data['dataset_info'].items():
            # attach semantic connections for this dataset if available
            conns = _coerce_key_lookup(data.get('semantic_connections', {}), key) or []
            # embed connection summary into the dataset dict copy so dict_to_html will render it
            dataset_with_conns = dict(dataset)
            if conns:
                dataset_with_conns['semantic_connections_summary'] = f"{len(conns)} connections found"
            detailed_html += f"""
            <div class="dataset-section" id="dataset-{key}">
                <h2>{dataset['file_info']['file_name']}</h2>
                <div class="dataset-content">
                    {dict_to_html(dataset_with_conns)}
                </div>
            </div>
            """
    elif "bridge_datasets" in data:
        # Handle results.json detailed bridge rendering
        for i, b in enumerate(data.get('bridge_datasets', [])):
            bridge = b.get('bridge_dataset', {}) or {}
            title = bridge.get('file_name') or bridge.get('file_path') or f'bridge-{i}'
            # ensure connections are visible in the rendered dict
            bcopy = dict(b)
            if bcopy.get('connections'):
                bcopy['connections_summary'] = f"{len(bcopy['connections'])} connections"
            detailed_html += f"""
            <div class="dataset-section" id="dataset-bridge-{i}">
                <h2>Bridge: {title} (Score: {b.get('bridge_score')})</h2>
                <div class="dataset-content">
                    {dict_to_html(bcopy)}
                </div>
            </div>
            """
    elif "source_dataset" in data and "target_dataset" in data:
        # Handle pathway analysis results (e.g., graph_roadmap.json)
        src = data.get('source_dataset', {}) or {}
        tgt = data.get('target_dataset', {}) or {}
        
        src_name = src.get('file_name') or src.get('file_path') or 'Source'
        tgt_name = tgt.get('file_name') or tgt.get('file_path') or 'Target'
        
        detailed_html += f"""
        <div class="dataset-section" id="pathway-source">
            <h2>Source Dataset: {src_name}</h2>
            <div class="dataset-content">
                {dict_to_html(src)}
            </div>
        </div>
        <div class="dataset-section" id="pathway-target">
            <h2>Target Dataset: {tgt_name}</h2>
            <div class="dataset-content">
                {dict_to_html(tgt)}
            </div>
        </div>
        """
        
        # Add pathway analysis details if available
        if 'contextual_connections' in data or 'semantic_connections' in data or 'shared_entities' in data:
            connection_data = {}
            if 'contextual_connections' in data:
                connection_data['contextual_connections'] = data['contextual_connections']
            if 'semantic_connections' in data:
                connection_data['semantic_connections'] = data['semantic_connections']
            if 'shared_entities' in data:
                connection_data['shared_entities'] = data['shared_entities']
            if 'confidence' in data:
                connection_data['confidence'] = data['confidence']
            
            detailed_html += f"""
            <div class="dataset-section" id="connection-analysis">
                <h2>Connection Analysis Results</h2>
                <div class="dataset-content">
                    {dict_to_html(connection_data)}
                </div>
            </div>
            """
    else:
        raise ValueError("Unsupported JSON structure. Ensure the file contains 'results', 'dataset_info', 'bridge_datasets', or pathway analysis data.")

    with open("exports/results.html", "w", encoding='utf-8') as f:
        # Precompute summary and table-of-contents HTML to avoid complex inline f-string parsing
        if "search_query" in data and "exploration_type" in data:
            summary_html = f"""
            <div class=\"summary\">
                <h3>Search Summary</h3>
                <div class=\"stat\">
                    <span class=\"stat-value\">{data.get('search_query', '')}</span>
                    <span class=\"stat-label\">Search Query</span>
                </div>
                <div class=\"stat\">
                    <span class=\"stat-value\">{data.get('exploration_type', '').replace('_', ' ').title()}</span>
                    <span class=\"stat-label\">Exploration Type</span>
                </div>
                <div class=\"stat\">
                    <span class=\"stat-value\">{data.get('matches_found', '')}</span>
                    <span class=\"stat-label\">Matches Found</span>
                </div>
                <div class=\"stat\">
                    <span class=\"stat-value\">{len(data.get('results', []))}</span>
                    <span class=\"stat-label\">Total Datasets</span>
                </div>
            </div>
            """
        elif "dataset_info" in data:
            summary_html = f"""
            <div class=\"summary\">
                <h3>Dataset Summary</h3>
                <div class=\"stat\">
                    <span class=\"stat-value\">{data.get('analysis_type', '').replace('_', ' ').title()}</span>
                    <span class=\"stat-label\">Analysis Type</span>
                </div>
                <div class=\"stat\">
                    <span class=\"stat-value\">{data.get('total_matrices', '')}</span>
                    <span class=\"stat-label\">Total Matrices</span>
                </div>
                <div class=\"stat\">
                    <span class=\"stat-value\">{data.get('connected_matrices', '')}</span>
                    <span class=\"stat-label\">Connected Matrices</span>
                </div>
            </div>
            """
        elif "bridge_datasets" in data:
            summary_html = f"""
            <div class=\"summary\">
                <h3>Bridge Analysis Summary</h3>
                <div class=\"stat\">
                    <span class=\"stat-value\">{data.get('exploration_type', '').replace('_', ' ').title()}</span>
                    <span class=\"stat-label\">Exploration Type</span>
                </div>
                <div class=\"stat\">
                    <span class=\"stat-value\">{data.get('bridges_found', '')}</span>
                    <span class=\"stat-label\">Bridges Found</span>
                </div>
                <div class=\"stat\">
                    <span class=\"stat-value\">{len(data.get('bridge_datasets', []))}</span>
                    <span class=\"stat-label\">Bridge Datasets</span>
                </div>
            </div>
            """
        else:
            summary_html = ""

        # Precompute table-of-contents links
        if "results" in data:
            toc_links = "".join([f'<a href="#dataset-{i}">{result.get("dataset_name", "")}</a>' for i, result in enumerate(data.get('results', []))])
        elif "dataset_info" in data:
            toc_links = "".join([f'<a href="#dataset-{k}">{v.get("file_info", {}).get("file_name", k)}</a>' for k, v in data.get('dataset_info', {}).items()])
        elif "bridge_datasets" in data:
            toc_links = "".join([f'<a href="#dataset-bridge-{i}">{ (b.get("bridge_dataset") or {}).get("file_name") or (b.get("bridge_dataset") or {}).get("file_path") or i }</a>' for i, b in enumerate(data.get('bridge_datasets', []))])
        else:
            toc_links = ""

        # Choose page title and header based on detected JSON structure
        if "results" in data:
            page_title = f"{data.get('search_query', 'Search')} - Search Results" if data.get('search_query') else "Search Results - Complete Data"
            header_title = f"{data.get('search_query', 'Entity Search')} Entity Search Results - Complete Data" if data.get('search_query') else "Entity Search Results - Complete Data"
        elif "dataset_info" in data:
            page_title = "Dataset Connections - Complete Data"
            header_title = "Dataset Connections - Complete Data"
        elif "source_dataset" in data and "target_dataset" in data:
            s_name = (data.get('source_dataset') or {}).get('file_name') or (data.get('source_dataset') or {}).get('file_path') or 'Source'
            t_name = (data.get('target_dataset') or {}).get('file_name') or (data.get('target_dataset') or {}).get('file_path') or 'Target'
            page_title = f"Connection: {s_name} → {t_name}"
            header_title = f"Connection Result: {s_name} -> {t_name}"
        elif "bridge_datasets" in data:
            page_title = "Bridge Analysis - Complete Data"
            header_title = "Bridge Analysis - Complete Data"
        else:
            page_title = "Search Results - Complete Data"
            header_title = "Search Results - Complete Data"

        f.write(f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{page_title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 40px;
            }}
            .summary {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            .summary h3 {{
                margin: 0 0 15px 0;
                font-size: 1.3em;
                grid-column: 1 / -1;
            }}
            .stat {{
                background: rgba(255,255,255,0.1);
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 1.5em;
                font-weight: bold;
                display: block;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }}
            .table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                border: none;
            }}
            .table td {{
                padding: 12px;
                border-bottom: 1px solid #e0e0e0;
                vertical-align: top;
            }}
            .table tbody tr:hover {{
                background-color: #f8f9fa;
            }}
            .table tbody tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .dataset-section {{
                margin: 30px 0;
                border: 2px solid #ecf0f1;
                border-radius: 10px;
                padding: 20px;
                background: #fafbfc;
            }}
            .dataset-content {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }}
            .json-key {{
                font-weight: bold;
                color: #2c3e50;
                margin: 10px 0 5px 0;
                font-size: 1.1em;
            }}
            .json-value {{
                margin-left: 20px;
                margin-bottom: 10px;
            }}
            .json-item {{
                margin: 5px 0;
                padding: 5px;
                background: #f8f9fa;
                border-radius: 4px;
            }}
            .json-item .key {{
                font-weight: bold;
                color: #2980b9;
            }}
            .json-item .value {{
                color: #27ae60;
                margin-left: 10px;
            }}
            .json-array-item {{
                font-weight: bold;
                color: #8e44ad;
                margin: 10px 0 5px 0;
            }}
            .toc {{
                background: #ecf0f1;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .toc h3 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .toc a {{
                color: #3498db;
                text-decoration: none;
                display: block;
                padding: 5px 0;
            }}
            .toc a:hover {{
                text-decoration: underline;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
                border-top: 1px solid #ecf0f1;
                padding-top: 20px;
            }}
            .toggle-button {{
                background: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
                margin: 10px 0;
            }}
            .toggle-button:hover {{
                background: #2980b9;
            }}
            .collapsible {{
                display: none;
            }}
            .collapsible.show {{
                display: block;
            }}
        </style>
        <script>
            function toggleSection(id) {{
                const element = document.getElementById(id);
                const button = document.querySelector("button[onclick*='" + id + "']");
                if (element && button) {{
                    if (element.classList.contains('show')) {{
                        element.classList.remove('show');
                        button.textContent = button.textContent.replace('Hide', 'Show');
                    }} else {{
                        element.classList.add('show');
                        button.textContent = button.textContent.replace('Show', 'Hide');
                    }}
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>{header_title}</h1>

        <!-- Summary Section -->
        {summary_html}

            <div class="toc">
                <h3>Table of Contents</h3>
                <a href="#summary-table">Summary Table</a>
                {toc_links}
            </div>

            <div id="summary-table">
                <h2>Summary Table</h2>
                {summary_table}
            </div>

            <h2>Complete Dataset Details</h2>
            <button class="toggle-button" onclick="toggleSection('all-details')">Show Complete JSON Data</button>

            <div id="all-details" class="collapsible">
                {detailed_html}
            </div>

            <div class="footer">
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | TensorPack Entity Search</p>
                <p>Total JSON data size: {len(str(data)):,} characters</p>
            </div>
        </div>
    </body>
    </html>""")
        print("HTML file created: exports/results.html")

    # Create a summary report only when the JSON contains 'results'
    if 'results' in data and isinstance(data.get('results'), list):
        try:
            summary_data = {
                'Dataset': [result.get('dataset_name', '') for result in data['results']],
                'Relevance Score': [result.get('relevance_score', None) for result in data['results']],
                'Data Type': [result.get('data_type', '') for result in data['results']],
                'Entity Matches': [len(result.get('entity_matches', [])) for result in data['results']],
                'File Path': [result.get('dataset_path', '') for result in data['results']]
            }

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv("exports/summary_report.csv", index=False)
            print("Summary report created: exports/summary_report.csv")

            # also write parquet for summary
            try:
                safe_write_parquet(summary_df, "exports/summary_report.parquet")
            except Exception:
                print("Could not write summary parquet.")

            # also write Excel and HTML versions to exports/
            try:
                summary_df.to_excel("exports/summary_report.xlsx", index=False)
            except Exception:
                print("Could not write summary excel.")

            try:
                summary_html = summary_df.to_html(classes='table table-striped', escape=False, index=False)
                with open("exports/summary_report.html", "w", encoding='utf-8') as f:
                    f.write(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Summary Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 40px;
            }}
            .summary {{
                background: linear-gradient(135deg, #e74c3c, #c0392b);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .summary h3 {{
                margin: 0 0 15px 0;
                font-size: 1.3em;
            }}
            .stat {{
                background: rgba(255,255,255,0.1);
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 1.5em;
                font-weight: bold;
                display: block;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
            }}
            .table th {{
                background: #34495e;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                border: none;
            }}
            .table td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            .table tr:hover {{
                background: #f8f9fa;
            }}
            .table tbody tr:nth-child(even) {{
                background: #f9f9f9;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
                border-top: 1px solid #ecf0f1;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Summary Report</h1>
            <div class="summary">
                <h3>Found {len(data.get('results', []))} datasets</h3>
            </div>
            {summary_html}
        </div>
    </body>
    </html>
                    """)
            except Exception as e:
                print("Failed to write summary HTML:", e)

        except Exception as e:
            print("Failed to create summary report:", e)

        # write summary markdown and also store summary table in sqlite
        try:
            if 'summary_df' in locals() and not summary_df.empty:
                summary_md = f"{base_name}_summary.md"
                md = df_to_markdown(summary_df, outdir='exports', filename=summary_md, title=f"Summary Report - {base_name}")
                # also write summary table to sqlite as 'summary'
                try:
                    import sqlite3
                    dbp = Path('exports') / 'results.db'
                    conn = sqlite3.connect(str(dbp))
                    summary_df.to_sql('summary', conn, if_exists='replace', index=False)
                    conn.close()
                    print(f"SQLite: 'summary' table written to {dbp}")
                except Exception as e:
                    print("Could not write summary to sqlite:", e)
        except Exception as e:
            print("Failed to export summary to markdown/pdf:", e)

    # Create a detailed entity matches file for the primary dataset if available
    if 'results' in data and isinstance(data.get('results'), list) and len(data['results']) > 0:
        primary_result = data['results'][0]
        if primary_result and primary_result.get('entity_matches'):
            entity_matches = []
            for match in primary_result['entity_matches']:
                # Basic match info
                em = {'entity_name': match.get('entity_name'), 'relevance_score': match.get('relevance_score')}
                # Add context snippets if available
                if match.get('context_snippets'):
                    em['context_snippet_1'] = match['context_snippets'][0] if len(match['context_snippets']) > 0 else ''
                entity_matches.append(em)

            if entity_matches:
                try:
                    # Create DataFrame with explicit dtypes to avoid conversion issues
                    entity_df = pd.DataFrame(entity_matches)
                    
                    # Ensure Row Index is string type
                    if 'Row Index' in entity_df.columns:
                        entity_df['Row Index'] = entity_df['Row Index'].astype(str)
                    
                    # Handle Confidence column type (could be float or string)
                    if 'Confidence' in entity_df.columns:
                        try:
                            entity_df['Confidence'] = pd.to_numeric(entity_df['Confidence'], errors='coerce')
                        except:
                            # If conversion fails, keep as strings
                            pass
                    
                    entity_df.to_csv("exports/entity_matches_detail.csv", index=False)
                    print(f"Detailed entity matches created: exports/entity_matches_detail.csv ({len(entity_matches)} matches)")

                    # write parquet for entity matches
                    safe_write_parquet(entity_df, "exports/entity_matches_detail.parquet")
                except Exception as e:
                    print("Failed to create entity matches CSV/parquet:", e)

                # Also create a nice HTML report for entity matches
                try:
                    entity_html = entity_df.to_html(classes='table', escape=False, index=False)

                    # Dynamic title/header based on primary_result dataset name
                    dataset_name = primary_result.get('dataset_name') if isinstance(primary_result, dict) else None
                    page_title_entity = f"{dataset_name} - Entity Matches" if dataset_name else "Entity Matches Detail"
                    header_entity = f"{dataset_name} Entity Matches" if dataset_name else "Entity Matches"

                    with open("exports/entity_matches.html", "w", encoding='utf-8') as f:
                        f.write(f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{page_title_entity}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            .summary {{ background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
            .table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
            .table th {{ background: #34495e; color: white; padding: 12px; text-align: left; }}
            .table td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            .table tr:hover {{ background: #f8f9fa; }}
            .table tbody tr:nth-child(even) {{ background: #f9f9f9; }}
            .footer {{ margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 0.9em; border-top: 1px solid #ecf0f1; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{header_entity}</h1>
            <div class="summary">
                <h3>Found {len(entity_matches)} exact matches in {primary_result.get('dataset_name', '')}</h3>
                <p>Dataset: {primary_result.get('dataset_name', '')} | Relevance Score: {primary_result.get('relevance_score', 0):.3f}</p>
            </div>
            {entity_html}
        </div>
    </body>
    </html>""")
                    print(f"Entity matches HTML created: exports/entity_matches.html")
                    # also create markdown/excel/html for entity matches and store in sqlite
                    try:
                        em_base = _sanitize_filename(primary_result.get('dataset_name') or base_name or 'entity_matches')
                        em_md = f"{em_base}_entity_matches.md"
                        md = df_to_markdown(entity_df, outdir='exports', filename=em_md, title=f"Entity Matches: {primary_result.get('dataset_name','')}")
                    except Exception as e:
                        print("Failed to export entity matches to markdown:", e)
                    try:
                        entity_df.to_excel(f"exports/{em_base}_entity_matches.xlsx", index=False)
                        print(f"Excel file created: exports/{em_base}_entity_matches.xlsx")
                    except Exception:
                        print("Could not write entity matches Excel (install openpyxl)")
                    try:
                        with open(f"exports/{em_base}_entity_matches.html", "w", encoding='utf-8') as ef:
                            ef.write(entity_df.to_html(classes='table', escape=False, index=False))
                        print(f"HTML file created: exports/{em_base}_entity_matches.html")
                    except Exception as e:
                        print("Could not write entity matches HTML:", e)

                    # write matches to sqlite (append to existing DB if present)
                    try:
                        _write_sqlite(outdir='exports', edges_df=edges_df if 'edges_df' in locals() else None, nodes_df=nodes_df if 'nodes_df' in locals() else None, matches_df=entity_df)
                    except Exception as e:
                        print("Failed to write entity matches to sqlite:", e)
                except Exception as e:
                    print("Failed to create entity matches HTML:", e)

