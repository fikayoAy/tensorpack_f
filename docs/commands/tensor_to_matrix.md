# tensor_to_matrix

This command takes a complex, multi-dimensional dataset (a **tensor**) and flattens it into a simpler, 2D **matrix** that is easier to analyze and work with. At the same time, it saves extra information (metadata) so the original tensor can be **reconstructed later without losing anything important**.
## Table of Contents

- [General Syntax](#general-syntax)
- [What it does](#what-it-does)
- [Why it matters](#why-it-matters)
- [Inputs](#inputs)
- [Outputs (by format)](#outputs-by-format)
- [Examples](#examples)
- [Quick try](#quick-try)
- [Quick notes](#quick-notes)
- [CLI Reference](#cli-reference)
- [Programmatic usage](#programmatic-usage)

### General Syntax

```bash
tensorpack tensor_to_matrix <input> --output <matrix_out> --meta-file <meta_out.json> [--verbose] [--log <logfile>] [--export-format <fmt>]
```

### **What it does**

- Reads a tensor file (formats like JSON, `.npy`, CSV, images, etc.).
- Converts the tensor into a clean 2D matrix.
- Saves both:
    - The **matrix file** (in your chosen format).
    - A **metadata file** that records details like shape, data type, and structure.

### **Why it matters**

- **Lossless conversion:** No information is lost during the conversion.
- **Reversible:** You can always rebuild the original tensor later.
- **Safe and reliable:** Numeric data and metadata are stored separately to prevent corruption.
- **Efficient:** Works with both small and very large files, with optimized memory handling.

### **Inputs**

- Path to the input tensor file.
- Optional flags like:
    - `-output` → where to save the matrix.
    - `-meta-file` → where to save the metadata.
    - `-verbose` → show detailed logs.
    - `-log` → specify a log file.
    - `-export-format` → override format (e.g., `.npy`, `.csv`).

### **Outputs (by format)**

Here’s what you’ll get depending on the output format you request:

- **`.npy`** → Lossless NumPy binary (recommended). Saves the matrix directly.
- **`.npz`** → Lossless compressed NumPy archive. Saves the matrix under a default key (`data`).
- **`.csv` / `.tsv`** → Text table (only for 2D arrays). For higher-dimensional data, TensorPack saves a `.npy` file plus a companion metadata `.json` instead of making a lossy CSV.
- **`.json`** → By default, JSON files store only metadata, not large numeric arrays. TensorPack will:
    - Save the numeric data separately as `<basename>.npy` (lossless).
    - Save a metadata-only JSON pointing to the binary file (with keys like `shape`, `dtype`, `canonical_binary`).
    - Loader still supports JSON that directly embeds `{"shape","data"}` or simple lists, but new saves prefer the companion binary approach.
- **Other formats** → Not supported (an error will be raised).

---

### **Examples**

- `-output out.npy` → produces `out.npy` (matrix).
- `-output out.npz` → produces `out.npz` (compressed archive).
- `-output out.csv` with 2D data → produces `out.csv`.
- `-output out.csv` with N-D data → produces `out.npy` (matrix) + `out.json` (metadata).
- `-output out.json` → produces `out.npy` (matrix) + `out.json` (metadata-only).

### Quick try

```bash
tensorpack tensor_to_matrix sample_6d_tensor.json --output sample_6d_matrix.csv --meta-file sample_6d_tensor_meta.json
```

![tensor_to_matrix1](img/tensor_to_matrix1.png)

![tensor_to_matrix(meta)](img/tensor_to_matrix(meta).png)

![tensor_to_matrix(2d rep)](img/tensor_to_matrix(2d rep).png)

---

### **Quick notes**

- Use **`.npy`** or **`.npz`** whenever possible — they’re the most reliable, lossless storage formats.
- If you need JSON with the full numeric data embedded (large and lossy), an explicit flag like `-embed-data` can be used to override the default companion-binary policy.
- The reverse command (`matrix_to_tensor_command`) automatically uses the metadata JSON to find and load any companion `.npy` files.

---

## CLI Reference

This section lists the most common flags and their behavior for `tensor_to_matrix`.

- `--output <path>`: Path to write the resulting 2D matrix. Can be `.npy`, `.npz`, `.csv`, `.tsv`, or `.json` (metadata-only).
- `--meta-file <path>`: Path to write metadata JSON describing the original tensor. If omitted, a companion `<output>.json` is created next to the output.
- `--export-format <fmt>`: Force an output format independent of file extension. Use with care; for ND arrays, `.csv` will cause a companion `.npy` to be written instead.
- `--embed-data`: Optional override to embed numeric arrays directly into the JSON output. This is discouraged for large arrays.
- `--verbose` / `--log <file>`: Enable detailed logging and redirect logs to a file.

## Programmatic usage

You can call the command from Python directly without invoking the shell. Example:

```python
from tensorpack import tensor_to_matrix_command

args = {
    'input': 'sample_6d_tensor.json',
    'output': 'sample_6d_matrix.csv',
    'meta_file': 'sample_6d_tensor_meta.json',
    'verbose': True
}

tensor_to_matrix_command(**args)
```
