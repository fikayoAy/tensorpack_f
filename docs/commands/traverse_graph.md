# traverse_graph_command
This explores semantic pathways and connections

**between datasets** (matrices, JSON, images, video, text, etc.). Supports point-to-point traversal, bridge discovery, and entity search — with outputs for further network analysis and visualization.
## Table of Contents

- [What it does](#what-it-does)
- [Saving behavior & output extensions](#saving-behavior--output-extensions)
- [CLI Reference](#cli-reference)
- [Quick Notes](#quick-notes)
- [Programmatic usage](#programmatic-usage)
- [Demo videos](#demo-videos)

## What it does

1. Configures OCR heuristics, logging, and license checks.
2. Expands file patterns and loads inputs via `load_and_apply_transform`.
3. Extracts semantic + metadata info for each dataset (detected type, matrix type, coordinates).
4. Supports three modes:
    - **Point-to-point:** find pathways between source and target datasets.
    - **Bridge discovery:** detect datasets acting as semantic bridges.
    - **Entity search:** locate entities across datasets and report matches.
5. Uses transformer routines + semantic/entity matchers to enrich results.
6. Always writes a JSON output with provenance, pathways, and metadata.
7. Optionally:
    - Saves `.npy/.npz` for numeric arrays.
    - Generates edge lists / graph exports.
    - Creates visualizations (heatmaps, t-SNE, dendrograms).
8. Returns `0` on success, `1` on failure.

---

## Saving behavior & output extensions

- `.json` → primary output (metadata, provenance, pathway details).
- `.npy` / `.npz` → lossless numeric exports.
- `.csv` / `.tsv` → 2D arrays only (lossy otherwise).
- `.xlsx`, `.parquet`, `.sqlite`, `.html`, `.md`, etc. → optional, via `convertto.py`.

**Details:**

- JSON is always written; companion binaries store large arrays.
- Unsupported/missing export options → fallback to `.npy` or warnings.
- With `-export-formats`, a temporary `results.json` is created, then passed to `convertto.py`.

---

## CLI Reference

**General syntax:**

```bash
tensorpack traverse-graph --inputs <patterns...> [options]

```

**Bridge discovery:**

```bash
tensorpack traverse-graph --inputs "data/*.csv" --find-bridges --output bridges --export-formats all --visualize viz

```

**Point-to-point pathway:**

```bash
tensorpack traverse-graph --inputs a.csv b.csv --source-dataset a.csv  --target-dataset b.csv --output pathway_results

```

**Entity search:**

```bash
tensorpack traverse-graph --inputs *.json --search-entity "BRCA1" --output brca_search --visualize brca_viz

```

## Quick Notes

- **Primary output:** JSON (`pathway_results.json` by default). Always written.
- **Modes:**
    - Point-to-point (`-source/--target` or `-source-dataset/--target-dataset`)
    - Bridge discovery (`-find-bridges`)
    - Entity search (`-search-entity "ENTITY"`)
- **Companion binaries:**
    - Not automatic in this command. `traverse_graph_command` writes only the JSON results.
    - If you need `.npy/.npz` for lossless dataset storage, you must either:
        - run `discover_connections_command` with `-save-npy`, or
        - save arrays separately using `save_to_file`.
- **Additional exports:**
    - `.csv`, `.xlsx`, `.html`, `.parquet`, `.sqlite`, `.md`, etc. are supported **only if `convertto.py` is present**.
    - Works by writing a temporary `results.json` → calling `convertto.py` → collecting generated files.
    - May modify `cwd` and overwrite `results.json` temporarily (backups attempted).
- **Visualizations:**
    - This command **does not produce heavy plots** (no heatmaps, dendrograms, t-SNE).
    - Instead, it provides CLI/table summaries and graph-style exports (`edge-list`, `graphml`).
    - For advanced PNG visualizations, use `discover_connections_command`.
- **To know:**
    - `.csv/.tsv` only safe for 2D arrays (higher-D reshaped or warned). Use `.npy/.npz` for lossless.
    - Some traversal features (e.g., unlimited bridge search) are license-gated.
    - Input expansion supports globs (e.g., `data/*.csv`).
    - Return code: `0` success, `1` failure; use `-verbose` for tracebacks.

**Common flags:**

- `-inputs <patterns...>` : Input file patterns (globs).
- `-source <file>` / `-target <file>` : Define single source/target.
- `-source-dataset` / `-target-dataset` : Point-to-point by dataset name/index.
- `-find-bridges` : Bridge discovery mode.
- `-search-entity "ENTITY"` : Entity search mode.
- `-apply-transform <name>` : Apply registered transform.
- `-output <base>` : Output base path (default: `pathway_results`).
- `-export-formats <fmt,...>` : Extra exports (csv,xlsx,html,parquet,sqlite,md,all).
- `-visualize <dir>` : Save visualizations (heatmaps, t-SNE, dendrograms).
- `-format <edge-list|graphml>` : Export graphs in edge list or GraphML.
- `-save-npy` : Save `.npy` companions for loaded datasets.
- `-clustering <method>` : Semantic clustering (`hierarchical`, `kmeans`, `auto`).
- `-threshold <0-1>` : Filter connections by relevance.
- `-max-connections <int>` : Limit connections per dataset.
- `-verbose` : Debug logging.
- `-log <file>` : Write logs to file.
- `-skip-errors` : Skip bad inputs instead of aborting.

---

## Programmatic usage

```python
from argparse import Namespace
from tensorpack import traverse_graph_command

# Example: bridge discovery
args = Namespace(
    inputs=['data/*.csv'],
    find_bridges=True,
    output='bridges',
    verbose=True,
    export_formats=['json','csv'],
    visualize='viz_output'
)

rc = traverse_graph_command(args)
if rc == 0:
    print("Traversal succeeded")
else:
    print("Traversal failed")

```

## Demo videos

- Entity search demo: `img/entity-search.mp4`
- Semantic analysis demo: `img/sematic-analysis.mp4`


