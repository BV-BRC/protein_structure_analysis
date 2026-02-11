# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based protein structure comparison pipeline for comparing predicted structures from AlphaFold, ESMFold, Chai, and Boltz, with pLDDT confidence score integration.

## Runtime environment

For testing we use the python venv in `/Users/olson/structure-venv`
## Commands

### Install
```bash
pip install -e .
# or
pip install -r requirements.txt
```

### Run CLI
```bash
# Compare two structures
python -m protein_compare compare struct1.pdb struct2.pdb

# Batch comparison (all pairwise)
python -m protein_compare batch *.pdb -o results.csv

# Batch comparison against reference
python -m protein_compare batch *.pdb --reference ref.pdb -o results.csv

# Generate visualization
python -m protein_compare visualize struct1.pdb struct2.pdb -o aligned.pml

# View structure info
python -m protein_compare info structure.pdb

# Generate contact map
python -m protein_compare contacts structure.pdb -o contacts.png
```

### Testing
```bash
pytest
pytest tests/test_specific.py::test_function  # single test
```

## Architecture

### Core Data Flow
1. **StructureLoader** (`io/parser.py`) parses PDB files into **ProteinStructure** dataclass containing Cα/Cβ coordinates, pLDDT scores (from B-factor column), and sequence
2. **StructuralAligner** (`core/alignment.py`) uses tmtools for TM-align, producing **AlignmentResult** with TM-scores, RMSD, transformation matrix, and residue mapping
3. **BatchComparator** (`core/batch.py`) orchestrates comparisons, optionally using joblib for parallel execution, and produces **PairwiseResult** objects
4. **ComparisonReporter** (`io/reporter.py`) generates CSV/JSON/HTML output

### Key Classes
- `ProteinStructure`: Immutable container for parsed structure with `ca_coords`, `plddt`, `sequence`
- `AlignmentResult`: Contains `tm_score`, `rmsd`, `rotation_matrix`, `translation_vector`, `residue_mapping`, `per_residue_distance`
- `PairwiseResult`: Full comparison result including `weighted_rmsd`, `ss_agreement`, `contact_jaccard`, `gdt_ts/gdt_ha`

### Analysis Modules
- `core/metrics.py`: RMSD calculations, confidence-weighted RMSD, GDT-TS/GDT-HA scores
- `core/secondary.py`: DSSP-based secondary structure assignment and comparison
- `core/contacts.py`: Contact map generation (Cα distance < 8Å) and Jaccard similarity
- `core/confidence.py`: pLDDT-based confidence analysis

### Visualization
- `visualization/alignment_viz.py`: PyMOL script generation, divergence plots
- `visualization/contact_maps.py`: Contact map heatmaps
- `visualization/divergence.py`: Divergence region highlighting

## Key Concepts

- **pLDDT scores**: Stored in PDB B-factor column (0-100 scale); high confidence ≥70, low confidence <50
- **TM-score**: >0.5 indicates same fold, >0.4 similar fold
- **Confidence-weighted RMSD**: Weights each residue by min(pLDDT₁, pLDDT₂)/100
- **Divergence threshold**: Default 3Å for counting divergent residues

## Dependencies

Requires Python ≥3.10. Key packages: biopython, tmtools, numpy, scipy, pandas, matplotlib, click, joblib. External: DSSP binary (mkdssp) for secondary structure analysis.
