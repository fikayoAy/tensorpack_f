# discover_connections_command

It finds semantic, mathematical, and entity-based links **between multiple datasets**.

Produces a ranked network of relationships with optional clustering, visualizations, and multi-format exports.

---

---

## What it does

1. Prepares resources (OCR, logging, license check).
2. Loads input datasets with metadata and entity info.
3. Detects type and generates semantic coordinates.
4. Extracts and registers entities from each dataset.
5. Computes:
    - Mathematical connections
    - Entity-based links
    - Semantic/contextual pairwise relationships
6. Merges signals into enhanced connection records, filters/ranks them.
7. Optionally:
    - Runs semantic-aware clustering
    - Generates visualizations (heatmaps, dendrograms, t-SNE)
    
8. Saves a main JSON connection report (+ companion binaries if needed).
9. Prints summary: strongest links, type distribution, runtime stats.

---

## Saving behavior & output extensions

- `.json` → main connection report (metadata + provenance).
- `.npy` / `.npz` → lossless numeric arrays.
- `.csv` / `.xlsx` / `.html` / `.sqlite` → optional exports.
- **Edge list** → text export for graph tools.

**Details:**

- JSON report always written; contains provenance + clustering metadata.
- Companion `.npy` files used for large or high-dimensional arrays.
- Export formats optional; require `convertto.py`.
- Tool falls back gracefully if unsupported options are requested.

---

## CLI Reference

**General syntax:**

```bash
tensorpack discover-connections --inputs <files...> --output <out.json> [options]

```

## Quick Notes

- **Primary output:** JSON (`connection_results.json` by default).
- **Companion binaries:** `.npy` / `.npz` required for lossless numeric storage.
- **Additional exports:** `.csv`, `.xlsx`, `.html`, `.sqlite` via `convertto.py` (optional).
- **Edge list:** Plain text edge list (`connections_edge_list.txt`).
- **Visualizations:** Heatmaps, dendrograms, t-SNE, cluster plots (requires `-visualize`).
- **To know:**
    - JSON is always written; `.npy` not auto-saved unless requested separately.
    - CSV/XLSX/HTML/SQLite are **lossy** (not for raw high-dimensional data).
    - Visualization needs ≥2 datasets and some valid connections.
    - Advanced features may require a license.

### Example

```bash
tensorpack discover-connections 
  --inputs file1.npy file2.csv 
  --output connections.json

```

**Extended example:**

```bash
tensorpack discover-connections --inputs data/*.csv data/notes.txt --num-dims 12 --threshold 0.5 --visualize out/viz --output out/connections.json --export-formats csv,xlsx --verbose

```

**Common flags:**

- `-inputs <files...>` : Input files (globs supported).
- `-apply-transform <name>` : Apply registered transform before analysis.
- `-num-dims <int>` : Semantic coordinate dimensionality.
- `-threshold <float>` : Relevance cutoff.
- `-max-connections <int>` : Limit connections per source.
- `-clustering <method>` : Clustering (hierarchical, kmeans, auto).
- `-visualize <dir>` : Save PNG visualizations.
- `-export-formats <list>` : Extra exports (`csv,xlsx,html,sqlite,all`).
- `-format <edge-list|...>` : Alternative output (edge list).
- `-output <path>` : Main JSON output file.
- `-verbose` : Debug logging.
- `-log <path>` : Write logs to file.
- `-skip-errors` : Skip failing inputs.

---

## Programmatic usage

```python
from argparse import Namespace
from tensorpack import discover_connections_command

args = Namespace(
    inputs=['file1.npy','file2.csv'],
    apply_transform=None,
    num_dims=8,
    threshold=0.35,
    max_connections=10,
    clustering='auto',
    visualize='viz_output',
    export_formats=['json','csv'],
    output='connections.json',
    format=None,
    verbose=True,
    log=None,
    skip_errors=False
)

rc = discover_connections_command(args)
if rc == 0:
    print("Discovery succeeded")
else:
    print("Discovery failed")

```

---

## 

---

## Example media

Below is an example demo video showing semantic connection discovery:

[Semantic connection discovery demo](img/Sematic-connection-discovery.mp4)

---
