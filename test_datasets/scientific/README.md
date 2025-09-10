# TensorPack Scientific Test Datasets

This directory contains real-world scientific datasets for testing TensorPack functionality across biological and pharmaceutical domains.

## Datasets

### 1. Drug Information (`drugs.tsv`)
- **Format**: TSV (Tab-Separated Values)
- **Size**: ~5MB
- **Content**: Comprehensive drug database with names, identifiers, and properties
- **Columns**: Drug names, chemical identifiers, therapeutic categories, molecular properties
- **Use Case**: Drug discovery, pharmaceutical research, entity matching

### 2. Gene Data (`genes.tsv`)
- **Format**: TSV (Tab-Separated Values)
- **Size**: ~3MB
- **Content**: Gene information including symbols, descriptions, and functional annotations
- **Columns**: Gene symbols, names, chromosomal locations, functional categories
- **Use Case**: Genomics research, gene expression analysis, biological pathway discovery

### 3. Drug-Gene Interactions (`interactions.tsv`)
- **Format**: TSV (Tab-Separated Values)
- **Size**: ~8MB
- **Content**: Drug-gene interaction data showing relationships and effects
- **Columns**: Drug IDs, gene targets, interaction types, effect strengths, evidence levels
- **Use Case**: Pharmacogenomics, drug target analysis, network biology

### 4. Biological Categories (`categories.tsv`)
- **Format**: TSV (Tab-Separated Values)
- **Size**: ~2MB
- **Content**: Hierarchical classification system for biological entities
- **Columns**: Category IDs, names, parent categories, classification levels
- **Use Case**: Ontology mapping, biological classification, semantic analysis

## Quick Test Commands

```bash
# Test cross-dataset bridge discovery
python tensorpack.py traverse-graph --inputs "drugs.tsv" "genes.tsv" "interactions.tsv" "categories.tsv" \
    --bridge-finder --include-metadata --export-formats all

# Test specific entity search across all datasets
python tensorpack.py traverse-graph --inputs "drugs.tsv" "genes.tsv" "interactions.tsv" "categories.tsv" \
    --search-entity "FESOTERODINE" --include-metadata --export-formats all

# Test comprehensive connection discovery with advanced clustering
python tensorpack.py discover-connections --inputs genes.tsv drugs.tsv interactions.tsv categories.tsv \
    --export-formats all --output connection_results --visualize visualizations \
    --clustering auto --num-dims 16 --threshold 0.3 --verbose

# Test basic tensor conversion
python tensorpack.py tensor_to_matrix test_datasets/scientific/interactions.tsv

# Test entity search for drug names
python tensorpack.py traverse-graph --inputs test_datasets/scientific/*.* \
    --search-entity "aspirin" --generate-viz --export-formats all

# Test cross-format combination
python tensorpack.py combine \
    --inputs test_datasets/scientific/drugs.tsv test_datasets/scientific/genes.tsv \
    --output combined_drug_gene_analysis.npy --mode weighted
```

## Dataset Statistics

- Total files: 4 TSV files
- Total size: ~18MB
- Data types: Tabular (TSV format)
- Entities: ~50,000+ drugs, ~20,000+ genes, ~100,000+ interactions
- Categories: Hierarchical biological classification system
- Update frequency: Quarterly updates from public databases

## Use Cases

### **Drug Discovery**
- **Entity Matching**: Find drugs by name, chemical structure, or therapeutic class
- **Target Discovery**: Identify gene targets for specific drugs
- **Pathway Analysis**: Map drug-gene interaction networks

### **Pharmacogenomics**
- **Drug-Gene Relationships**: Analyze how genes affect drug response
- **Interaction Networks**: Discover complex multi-gene drug interactions
- **Biomarker Discovery**: Find genetic markers for drug efficacy

### **Biological Research**
- **Cross-Domain Bridging**: Connect drugs to biological pathways via genes
- **Semantic Analysis**: Classify entities using biological ontologies
- **Network Biology**: Analyze complex biological interaction networks

### **Data Integration**
- **Multi-Format Discovery**: Connect tabular data across different domains
- **Entity Resolution**: Match entities across different naming conventions
- **Semantic Bridging**: Find conceptual connections between biological entities

## Example Analyses

### **Find Drug Targets**
```bash
# Search for a specific drug and its gene targets
python tensorpack.py traverse-graph --inputs drugs.tsv interactions.tsv genes.tsv \
    --search-entity "WARFARIN" --bridge-finder --export-formats csv xlsx
```

### **Discover Gene-Drug Networks**
```bash
# Analyze connections between genes and drugs
python tensorpack.py discover-connections --inputs genes.tsv drugs.tsv interactions.tsv \
    --clustering hierarchical --threshold 0.4 --visualize gene_drug_network
```

### **Biological Classification Analysis**
```bash
# Map entities to biological categories
python tensorpack.py traverse-graph --inputs categories.tsv drugs.tsv genes.tsv \
    --bridge-finder --include-metadata --generate-viz
```

## Data Sources

- **DrugBank**: Comprehensive drug information database
- **HUGO Gene Nomenclature**: Official gene symbols and names
- **PharmGKB**: Pharmacogenomics knowledge base
- **Gene Ontology**: Biological process and function classifications
- **ChEMBL**: Bioactive drug-like small molecules

## Expected Outputs

### **Bridge Finder Results**
- Cross-dataset entity connections
- Interaction strength measurements
- Network topology analysis
- Semantic relationship maps

### **Entity Search Results**
- Entity locations across datasets
- Contextual information and metadata
- Confidence scores and relevance rankings
- Cross-reference mappings

### **Connection Discovery Results**
- Multi-dimensional relationship analysis
- Clustering patterns and groupings
- Statistical correlation matrices
- Interactive visualizations

## Advanced Features Demonstrated

- **Multi-domain entity matching** across pharmaceutical and genomic data
- **Semantic bridging** between different biological classification systems
- **Network analysis** of complex drug-gene interaction patterns
- **Cross-format compatibility** with TSV, CSV, and other tabular formats
- **Hierarchical clustering** for biological pathway discovery
- **Export flexibility** to multiple formats (CSV, Excel, JSON, GraphML)

Generated on: 2025-09-01 19:30:00
