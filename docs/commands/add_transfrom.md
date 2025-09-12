# add_transform_command

Register a **custom transformation** or **entity matcher** into TensorPack’s user transform registry so it can be applied automatically during data loading and processing.

## What it does

1. Configures logging and (optionally) runs a license check.
2. Validates inputs and parses `properties` + `neighbors` (JSON or key=value).
3. Loads the transform from one of several **source types**:
    - **python** → load a function from a `.py` file.
    - **executable** → wrap an external executable.
    - **inline** → parse inline JSON/YAML definition.
    - **config** → load from a config file.
4. Optionally tests the transform on given data:
    - Uses declared `handles_format` (passes file path) or loads file via `load_tensor_from_file`.
    - Test results can be saved with `-test-output`.
5. Registers the transform via `add_transform_to_registry`:
    - Metadata JSON written to registry (`~/.tensorpack/transforms/<data_type>/`).
    - Python source copied into registry if applicable.
    - Matrix-type transforms are also registered with the `MatrixTransformer` instance.
6. Prints concise success/failure output and short usage hints.
7. Returns `0` on success, `1` on failure (with verbose stack traces if enabled).

---

## Saving behavior & output location

**Short answer:**

- Always writes metadata JSON under `~/.tensorpack/transforms/<data_type>/<name>_metadata.json`.
- For Python transforms, also copies the `.py` source into the same registry directory.
- No large binaries are produced.

**Details:**

- Metadata includes: name, data_type, properties, neighbors, source_info, description, version, dependencies, created timestamp.
- Test outputs (if `-test-output` is provided) are saved at the requested path (format depends on transform).
- Dependencies are recorded in metadata but not installed automatically.

---

## CLI Reference

**General syntax:**

```bash
python tensorpack.py add-transform --name <name> --data-type <type> \
  --source-type <python|executable|inline|config> --source <path_or_def> [options]

```

**Common flags:**

- `-name <name>` : Transform/matcher name (required).
- `-data-type <type>` : Dataset type (matrix, json, graph, timeseries, image, audio, genomics, finance, text, biomedical, custom).
- `-source-type <python|executable|inline|config>` : Where to load transform from.
- `-source <path_or_definition>` : File path or inline definition.
- `-function-name <fn>` : Function name inside a Python file (default: `transform`).
- `-properties "<json_or_k=v,...>"` : Transform properties.
- `-neighbors "n1,n2,...”` : Neighbor transforms.
- `-description "<text>"` : Human-readable description.
- `-version <ver>` : Version string.
- `-dependencies "pkg1,pkg2,...”` : Dependency list (informational only).
- `-operation-type <transformation|entity_matching>` : Register as normal transform or entity matcher.
- `-test-data <file>` : Run transform test on this dataset.
- `-test-entity <entity>` : For entity matchers, entity to search for.
- `-test-output <path>` : Save test results.
- `-force` : Register even if test fails.
- `-verbose` : Debug logging.
- `-log <file>` : Write logs to file.

---

## Example CLI invocations

**Add a Python transform:**

```bash
tensorpack add-transform \
  --name my_scale \
  --data-type matrix \
  --source-type python \
  --source ./my_scale.py \
  --function-name scale \
  --description "Scale matrix values"

```

**Add an inline transform:**

```bash
tensorpack add-transform \
  --name inline_norm \
  --data-type matrix \
  --source-type inline \
  --source '{"type":"normalize","parameters":{"norm":"max"}}'

```

**Executable-based transform with test:**

```bash
tensorpack add-transform \
  --name extproc \
  --data-type json \
  --source-type executable \
  --source /usr/local/bin/myproc \
  --test-data sample.json \
  --test-output test_out.json

```

**Register an entity matcher:**

```bash
tensorpack add-transform \
  --name gene_matcher \
  --data-type text \
  --operation-type entity_matching \
  --source-type python \
  --source ./gene_matcher.py \
  --function-name match_entity \
  --test-data docs.json \
  --test-entity BRCA1

```

---

<!-- Embedded demo video (repo root) -->

**Demo (short):**

[Demo: add-transform walkthrough](../../add-transfrom1.mp4)

---

```python
# Python transform file
tensorpack add-transform --name mynorm --data-type matrix \
  --source-type python --source ./mynorm.py --function-name transform

# Inline JSON
tensorpack add-transform --name scaleup --data-type image \
  --source-type inline --source '{"type":"scale","parameters":{"factor":2.0}}'

# Config file
tensorpack add-transform --name cfgnorm --data-type text \
  --source-type config --source ./conf.json

# Executable
tensorpack add-transform --name extproc --data-type csv \
  --source-type executable --source /usr/local/bin/process_csv \
  --properties 'handles_format=.csv'
```

## Quick Notes

- **Primary output:** Metadata JSON saved under `~/.tensorpack/transforms/<data_type>/<name>_metadata.json`.
- **Python transforms:** The source `.py` is also copied into the registry directory.
- **Operation types:**
    - `transformation` (default)
    - `entity_matching` (for entity matchers).
- **Testing:** You can run the transform on sample data during registration (`-test-data`, `-test-entity`).
- **Failure handling:** Returns `0` success, `1` failure. Use `-force` to register even if tests fail.
- **No large binaries:** Unlike dataset commands, this command only writes JSON + optional `.py`.

## Quick Notes (finalized)

### Primary output & metadata

- Writes: `~/.tensorpack/transforms/<data_type>/<name>_metadata.json`
- Metadata fields:
    - `name`, `data_type`, `properties`, `neighbors`, `operation_type`
    - `source_info`: source type, source path, embedding flags, dependencies, etc.
    - `provenance`: timestamp, checksum, user, platform, file sizes, rejection reasons if embedding blocked.

---

### Registry behavior

- All `source_type`s (`python`, `executable`, `inline`, `config`) produce metadata.
- **Embedding:** if `embed`/`embed_unrestricted` is requested, source is persisted in:
    
```
~/.tensorpack/transforms/<data_type>/artifacts/
```

- If no embedding: metadata records `original_source_path` + size.
- Large files: embedding rejected if over `embed_max_mb` (unless `embed_unrestricted` is set).

---

### Source types

- **Python**: path or inline code. Function must accept at least one argument. `.py` copied or embedded if allowed.
- **Executable**: path to binary/script; wrapped runner handles I/O via temp files.
- **Inline**: JSON/YAML string with `"type"` + optional `"parameters"`.
- **Config**: JSON/YAML config file converted into inline transform.

---

### Properties & neighbors

- `properties`: JSON string or `k=v` pairs. Auto-coerced (bool, int, float).
- Special keys: `handles_format`, `handles_formats` → file extensions handled directly.
- `neighbors`: list of related transforms for registry bookkeeping.

---

### Testing

- `-test-data`: run transform on sample.
- Entity matchers (`operation_type=entity_matching`): use `-test-entity`.
- If `handles_format` matches input extension, transform may receive file path directly.
- `-test-output`: write results.
- Failures → return `1`, unless `-force` is set (continue with warnings).

---

### Operation types

- Default: `transformation`.
- `entity_matching`: expects function signature `(data|file_path, entity, dataset_info)` and is logged differently.

---

### MatrixTransformer integration

- If `data_type=matrix` and `operation_type=transformation`, the transform is auto-added to active `MatrixTransformer` instances.

---

### License checks

- Calls `LicenseManager().check_feature_access('custom_transformations')`.
- Denied → abort with message.
- Exception → logs warning, continues with fallback/basic behavior.

---

### Failure handling

- Exit codes: `0` success, `1` failure.
- `-force`: bypass test failures, continue registration, metadata records warnings/rejection reasons.

---

### Provenance & safety

- Provenance includes checksums, size, timestamp, user, platform.
- Large binaries are never stored in metadata JSON (only in artifacts if explicitly embedded).

---

## Practical suggestions

- Executables are not always `.exe` — on Unix-like systems they often have no extension and must be executable (chmod +x) with a shebang.
- Unix: make your script executable and include a shebang, then call it directly. Example:

```bash
chmod +x /usr/local/bin/process_csv
# then register
tensorpack add-transform --name extproc --data-type csv --source-type executable --source /usr/local/bin/process_csv --properties 'handles_format=.csv'
```

- If your transform is a Python script and not executable, either:
  - Make it executable with `#!/usr/bin/env python3` and `chmod +x`, or
  - Create a tiny wrapper executable that invokes the Python script (recommended), because `create_executable_wrapper` calls the path directly and does not prepend an interpreter automatically.

- If the binary/script is large and you want it stored in the registry, pass `--embed` (watch `--embed-max-mb` / `--embed-unrestricted`). Always validate with `--test-data` and `--test-output`.

---

## Metadata handling for file-format transforms (recommended)

When you add a transform that operates on a specific file format (for example, Parquet, NetCDF, HDF5, or custom binary formats), we strongly recommend storing a rich metadata description alongside the transform. This enables TensorPack to surface semantic richness (column types, entity candidates, ranges, missing-value counts, provenance) and makes downstream entity extraction, semantic indexing, and visualizations much more accurate.

Why store rich metadata?
- Enables semantic indexing of string/numeric/date fields.
- Lets transformations and entity matchers decide which fields to process.
- Preserves provenance so transforms can be reproduced later.
- Improves export fidelity when converting back to the original format.

Recommended metadata contract
- Each file-format transform should produce and persist (in the transform registry or alongside the converted tensor) a JSON-friendly metadata dictionary. The transform should return a tuple (numpy_array, metadata_dict). The metadata should at minimum document schema, column names, data types, and useful derived statistics. Example (Parquet-inspired):

```python
metadata = {
    'schema': str(table.schema),
    'columns': df.columns.tolist(),
    'row_count': len(df),
    'column_count': len(df.columns),
    'original_dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
    'parquet_metadata': table.schema.metadata if table.schema.metadata else {},
    'has_index': not df.index.equals(pd.RangeIndex(len(df))),
    # Add tensorpack integration metadata
    'entity_extraction': {
        'string_columns': [col for col in df.columns if df[col].dtype == 'object'],
        'numeric_columns': [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
        'categorical_columns': [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])],
        'date_columns': [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    },
    'value_ranges': {
        col: {'min': df[col].min(), 'max': df[col].max()} 
        for col in df.select_dtypes(include=np.number).columns
    },
    'unique_values': {
        col: df[col].nunique()
        for col in df.columns if df[col].nunique() < len(df) * 0.5  # Only for low-cardinality columns
    },
    'missing_values': {
        col: int(df[col].isna().sum())
        for col in df.columns
    },
    'file_info': {
        'path': file_path,
        'size': os.path.getsize(file_path),
        'last_modified': os.path.getmtime(file_path)
    }
}
```

Generic transform contract (recommended)
- Name: `transform(data, *args, **kwargs)`
- Accepts either:
  - a path / Path-like pointing to a file of the handled format, or
  - a numpy.ndarray (identity/pass-through)
- Returns: `(numpy.ndarray, dict)` where `dict` follows the metadata contract above.

This contract is already implemented as an example in `parquet_handler.py` in the repository; authors of file-format transforms are encouraged to follow that implementation and adapt fields for their format. Using this pattern ensures TensorPack can integrate the transform output into semantic indexing, visualizations, and export pipelines without losing provenance or semantic context.

Storage & registry notes
- Persist the returned `metadata` next to the transform registration (under `~/.tensorpack/transforms/<data_type>/`) or embed a compact metadata summary directly in the transform metadata JSON.
- Avoid storing very large raw arrays inside metadata; prefer file references and concise summaries (min/max, cardinality, sample values).
- Record the `source_format` and `source_path` keys so the system knows how to reconstruct or re-export data in its original format.

