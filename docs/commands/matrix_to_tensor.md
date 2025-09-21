# matrix_to_tensor

This command reconstructs a **multi-dimensional tensor** from a saved matrix and its metadata. It guarantees that the original structure, shape, and meaning of the data can be restored.

### General Syntax

```python
matrix_to_tensor <input_matrix> --output <tensor_out> --metadata <meta_in.json> [--target-shape <dims>] [--target-dtype <dtype>] [--verify] [--error-threshold <float>] [--inspect] [--verbose] [--log <logfile>] [--export-format <fmt>]
```

### **What it does**

- Loads a saved matrix from disk.
- Loads metadata (from `-metadata` or a companion `<input>_metadata.json`).
- If `-inspect` is used, prints metadata and exits without reconstructing.
- Optionally uses a **target shape** (e.g., `-target-shape 3,64,64`) to guide reconstruction.
- Calls `MatrixTransformer.matrix_to_tensor(...)` to rebuild the tensor.
- Optionally casts the tensor to a requested data type (`-target-dtype`).
- If `-verify` is passed (with metadata available), checks the reconstruction by converting back to a matrix and computing reconstruction error. Exits with failure if error exceeds `-error-threshold`.
- Saves the tensor to the requested output format.

### **Why it matters**

- **Reconstruction without loss:** Safely restores the original multi-dimensional tensor.
- **Verification:** Ensures accuracy with optional error checks.
- **Flexible:** Supports shape guidance, dtype conversion, and multiple output formats.
- **Reliable:** Works seamlessly with metadata from `tensor_to_matrix_command`.

---

### **Inputs**

- A matrix file (`.npy`, `.npz`, `.csv`, `.json`).
- Metadata file (`-metadata`) or automatically found `<input>_metadata.json`.
- Optional flags for shape, dtype, verification, and logging.

---

### **Outputs (by format)**

- **`.npy`** → Lossless NumPy binary (recommended). Writes the tensor directly.
- **`.npz`** → Lossless compressed archive. Writes the tensor under a default key (`data`).
- **`.csv` / `.tsv`** → Only for 2D tensors. For N-D tensors, produces `<basename>.npy` plus companion `<basename>.json`.
- **`.json`** → Metadata-only JSON + companion `.npy`. (Numeric arrays are not embedded by default.)
- **Other formats** → Not supported; logs error.

---

### **Examples**

- `-output out.npy` → produces `out.npy` (tensor).
- `-output out.npz` → produces `out.npz` (tensor archive).
- `-output out.csv` with 2D tensor → produces `out.csv`.
- `-output out.csv` with ND tensor → produces `out.npy` (tensor) + `out.json` (metadata).
- `-output out.json` → produces `out.npy` (tensor) + `out.json` (metadata-only).

---

### **Quick notes**

- Use **`.npy`** or **`.npz`** for guaranteed lossless storage.
- JSON saves are metadata-first, with a `_data_path` field pointing to the companion `.npy`.
- To force embedded JSON arrays, an explicit override flag like `-embed-data` would be required.
- With `-verify`, the command computes mean absolute reconstruction error and fails if above `-error-threshold`.
- Verbose mode (`-verbose`) gives detailed logs and tracebacks on errors.

---

### Quick try

```bash
tensorpack matrix_to_tensor sample_6d_matrix.csv --output restored_6d_tensor.json --metadata sample_6d_tensor_meta.json
```

---

## CLI Reference

Common flags for `matrix_to_tensor`:

- `--output <path>`: Path to write the reconstructed tensor. Use `.npy`/`.npz` for lossless storage.
- `--metadata <path>`: Path to the metadata JSON produced by `tensor_to_matrix`. If omitted, the command looks for a companion metadata file near the input.
- `--target-shape <dims>`: A comma-separated list of integers to guide reconstruction (e.g., `3,64,64`).
- `--target-dtype <dtype>`: Cast the reconstructed tensor to a dtype like `float32` or `int64`.
- `--verify`: Re-convert and compute reconstruction error; fails if it exceeds `--error-threshold`.
- `--embed-data`: If present, allows writing JSON with embedded numeric arrays instead of companion binaries.

## Programmatic usage

Call the operation directly from Python when you need finer program control:

```python
from tensorpack import matrix_to_tensor_command

options = {
	'input_matrix': 'sample_6d_matrix.csv',
	'output': 'restored_6d_tensor.json',
	'metadata': 'sample_6d_tensor_meta.json',
	'verify': True
}

matrix_to_tensor_command(**options)
```

![matrix_to_tensor(meta)](img/matrix_to_tensor(meta).png)

![matrix_to_tensor(restored)](img/matrix_to_tensor(restored).png)

![comparing](img/comparing.png)
