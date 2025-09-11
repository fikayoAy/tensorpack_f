# TensorPack

## What it is

TensorPack is a software product with a command-line interface (CLI) that helps convert and analyze data between different formats. It focuses on converting tensors (multi-dimensional arrays) to matrices (2D arrays) and vice versa, and it also helps discover and understand connections between different datasets.

## Pain Point Analysis

### Core Problem

Working with complex, multi-dimensional data (tensors) is difficult. Converting them into usable formats often breaks important relationships, and analyzing connections across different datasets requires time, expertise, and custom code.

### How TensorPack Helps

1. **Seamless Data Conversion**
    - Easily switch between tensors (multi-dimensional arrays) and matrices (2D arrays).
    - Preserve relationships and metadata during transformations.
2. **Custom Data Transformations**
    - Define your own transformation rules using Python, external executables, or config files.
    - Adapt to diverse data sources and domains without rewriting code.
3. **Relationship Discovery**
    - Automatically detect and analyze relationships across datasets.
    - Includes semantic analysis, entity linking, and visualization support.
4. **Explorable Connections**
    - Navigate data as a graph: find pathways, discover bridges, and search across datasets.
    - Export results in multiple formats for further analysis.


### Benefits

- **Reduced Complexity:** Simplifies handling of multi-dimensional, multi-modal data.
- **Semantic Preservation:** Maintains meaning and relationships across all transformations.
- **Automation:** Automatically discovers relationships and patterns across datasets, images, and videos.
- **Flexibility:** Handles standard formats (json, csv, excel) and multi-modal inputs.
- **Custom Transform Extensibility:** Users can add support for proprietary formats (text, tabular, images, videos) via Python, external executables, or configuration files.
- **Rich Visualization & Export:**
    - **Visuals:** Interactive html visualizations, png static plots, publication-ready pdf reports
    - **Data/Analysis:** json, csv, excel, parquet, sqlite
    - **Documentation:** markdown, html
- **Contextual Understanding:** Retains semantic context and relationships across all operations, ensuring results remain meaningful and interpretable.

## How It Works

### 1. Adding Custom Transformations
[| Demo | Play |
|---|---|
| [![Adding Custom Transformations](docs/commands/img/add-transform.png)](https://youtu.be/MPjSGoGCelQ) | 👉 Click the thumbnail to open the demo (opens on YouTube). [Watch on YouTube](https://youtu.be/MPjSGoGCelQ) 🎬 |
]

```bash
# Add a custom transformation
tensorpack add-transform \
  --name "custom_transform" \
  --data-type "matrix" \
  --source-type "python" \
  --source "transform.py" \
  --function-name "transform_func" \
  --properties "dimension=3,normalize=true" \
  --export-formats "json,html,excel"
```

#### Custom file loader Extension via custom Transform
[| Demo | Play |
|---|---|
| [![Custom File Loader Extension](docs/commands/img/add-transform1.png)](https://youtu.be/WqD-OlNqjMs) | 👉 Click the thumbnail to open the demo (opens on YouTube). [Watch on YouTube](https://youtu.be/WqD-OlNqjMs) 🔗 |
]

### Simple walkthrough

🔧 Register a new custom Python transform (parquet handler)
```bash
python tensorpack.py add-transform \
  --name parquet_handler \
  --data-type custom \
  --source-type python \
  --source ./parquet_handler.py \
  --properties handles_format=.parquet,maintains_metadata=true \
  --test-data sample.parquet \
  --test-output test_output.np
```

| Flag | Description |
|---|---|
| `--name parquet_handler` | The registry name for the transform. |
| `--data-type custom` | Category under which the transform is registered (used to filter transforms). |
| `--source-type python` | Indicates the transform is loaded from a Python file. |
| `--source ./parquet_handler.py` | Path to the Python source implementing the transform. |
| `--properties handles_format=.parquet,maintains_metadata=true` | Transform metadata describing handled formats and behavior. |
| `--test-data sample.parquet` | Run the transform on this sample file as a sanity check during registration. |
| `--test-output test_output.np` | Path to save the test run output for inspection. |

See the full registration reference: `docs/commands/add_transfrom.md`.

📦 List installed transforms of type `custom`
```bash
python tensorpack.py list-transforms --data-type custom
```

| Flag | Description |
|---|---|
| `--data-type custom` | Filters the list to transforms registered under the `custom` category. |

▶️ Use the registered transform while traversing and searching entities
```bash
python tensorpack.py traverse-graph \
  --inputs genes.parquet \
  --search-entity "PPARG" \
  --include-metadata \
  --export-formats all \
  --output search.json \
  --apply-transform parquet_handler \
  --verbose
```

| Flag | Description |
|---|---|
| `--inputs genes.parquet` | Input dataset(s) to analyze; transforms that declare `handles_format` can be applied automatically. |
| `--search-entity "PPARG"` | The entity term to search for across datasets. |
| `--include-metadata` | Include dataset and transform metadata in the output for richer context. |
| `--export-formats all` | Export results in all supported formats (json, html, csv, etc.). |
| `--output search.json` | Path to save the exported search results. |
| `--apply-transform parquet_handler` | Explicitly apply the named transform during data loading. |
| `--verbose` | Enable detailed logging during the run. |

TensorPack allows you to extend its file loading capabilities by registering a custom transform that acts as a file loader for new formats. For example, you can add support for `.parquet` files (or any other format not natively supported) by:

1. **Implementing a custom Python transform** that loads the new file type (e.g., using `pandas.read_parquet`).
2. **Registering the transform** with `tensorpack add-transform`, specifying the file type and indicating that the transform handles loading (using the `handles_format` property in the transform metadata).
3. **Using the transform** in CLI commands (e.g., `--apply-transform parquet_handler`) to load and process files of the new type, even though TensorPack itself does not have built-in support for that format.

This approach enables seamless integration of proprietary or emerging file formats into your TensorPack workflows, without modifying the core codebase. The video above demonstrates registering and using a custom `.parquet` file loader as a transform.

### 2. Entity Search and Analysis
[| Demo | Play |
|---|---|
| [![Entity Search](docs/commands/img/entity-search.png)](https://youtu.be/5YMdhLGR6zs) | 👉 Click the thumbnail to open the demo (opens on YouTube). [Watch on YouTube](https://youtu.be/5YMdhLGR6zs) ▶️ |
]

```bash
# Search for entities across datasets
tensorpack traverse-graph \
  --inputs "dataset1.json" "dataset2.csv" \
  --search-entity "Term" \
  --include-metadata \
  --export-formats "json,html,excel,graphml" \
  --output "entity_analysis"
```

### 3. Semantic Connection Discovery
[| Demo | Play |
|---|---|
| [![Semantic Connections](docs/commands/img/sematic1.png)](https://youtu.be/r9pMhrHTcfI) | 👉 Click the thumbnail to open the demo (opens on YouTube). [Watch on YouTube](https://youtu.be/r9pMhrHTcfI) 🔍 |
]

```bash
# Discover semantic connections
tensorpack discover-connections \
  --source "source_data.json" \
  --target "target_data.json" \
  --depth 3 \
  --min-confidence 0.7 \
  --include-context \
  --export-formats "all" \
  --output "semantic_connections"
```

### 4. Semantic Analysis
[| Demo | Play |
|---|---|
| [![Semantic Analysis](docs/commands/img/sematic.png)](https://youtu.be/CdMyyu1ZKeo) | 👉 Click the thumbnail to open the demo (opens on YouTube). [Watch on YouTube](https://youtu.be/CdMyyu1ZKeo) 📊 |
]

```bash
# Perform semantic analysis
tensorpack analyze-semantic \
  --input "dataset.json" \
  --model "transformer-large" \
  --extract-entities \
  --include-relationships \
  --visualize \
  --export-formats "html,json,neo4j" \
  --output "semantic_analysis"
```

Each command supports various export formats:
- `json`: Structured data output
- `csv`: Tabular format
- `excel`: Microsoft Excel workbook
- `html`: Interactive visualizations
- `parquet`: Columnar storage format
- `sqlite`: Relational database
- `pdf`: Publication-ready reports
- `png`: Static visualizations
- `markdown`: Documentation format

Use `--export-formats all` to export in all available formats, or specify individual formats with comma-separated values.

## Command reference

Detailed command documentation is available in the `docs/commands` folder. See the following markdown files for usage, flags, and examples for each command:

- `docs/commands/add_transfrom.md` — Add and manage custom transforms (register loaders, transformers and exporters).
- `docs/commands/discover_connection.md` — Discover semantic and contextual connections between datasets.
- `docs/commands/matrix_to_tensor.md` — Convert matrices back into tensor representations with metadata preservation.
- `docs/commands/tensor_to_matrix.md` — Convert tensors into matrices with optional normalization and metadata extraction.
- `docs/commands/traverse_graph.md` — Traverse dataset graphs, search entities and find pathways across inputs.

There are also example media and additional notes in `docs/commands/img/` and sample datasets in `docs/commands/sample_data/` to help you get started.



