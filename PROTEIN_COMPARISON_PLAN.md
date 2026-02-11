# Protein Structure Comparison Pipeline

## Overview
A Python-based batch comparison tool for protein structures from computational prediction tools (AlphaFold, ESMFold, Chai, Boltz), with pLDDT confidence score integration.

---

## Project Structure

```
protein_compare/
├── __init__.py
├── cli.py                    # Command-line interface
├── core/
│   ├── __init__.py
│   ├── alignment.py          # TM-align wrapper and structural alignment
│   ├── metrics.py            # RMSD, TM-score calculations
│   ├── secondary.py          # DSSP/STRIDE secondary structure analysis
│   ├── contacts.py           # Contact map generation and comparison
│   └── confidence.py         # pLDDT score handling
├── io/
│   ├── __init__.py
│   ├── parser.py             # PDB file parsing with confidence extraction
│   └── reporter.py           # Output report generation (CSV, JSON)
├── visualization/
│   ├── __init__.py
│   ├── alignment_viz.py      # Aligned structure visualization
│   ├── contact_maps.py       # Contact map plotting
│   └── divergence.py         # Divergence region highlighting
├── utils/
│   ├── __init__.py
│   └── helpers.py            # Utility functions
├── tests/
│   └── ...
└── requirements.txt
```

---

## Implementation Steps

### Step 1: Project Setup & Dependencies

**File: `requirements.txt`**
```
biopython>=1.81          # PDB parsing, structure manipulation
numpy>=1.24              # Numerical operations
scipy>=1.10              # Distance calculations
matplotlib>=3.7          # Visualization
seaborn>=0.12            # Enhanced plots
tmtools>=0.1             # Python bindings for TM-align
biotite>=0.37            # Alternative structure analysis
```

**External tools required:**
- DSSP (via `mkdssp` binary or BioPython wrapper)
- STRIDE (optional fallback)

---

### Step 2: PDB Parsing with Confidence Scores

**File: `io/parser.py`**

```python
# Key functionality:
# - Parse PDB files from AlphaFold/ESMFold (pLDDT in B-factor column)
# - Extract atomic coordinates, residue information
# - Build confidence score arrays per residue
# - Handle multi-chain structures

class StructureLoader:
    def load(path: str) -> Structure
    def extract_plddt(structure: Structure) -> np.ndarray
    def get_ca_atoms(structure: Structure) -> np.ndarray
```

**Notes:**
- AlphaFold/ESMFold store pLDDT scores (0-100) in B-factor column
- Chai/Boltz may use different conventions - detect automatically

---

### Step 3: Structural Alignment (TM-align)

**File: `core/alignment.py`**

```python
# Key functionality:
# - Align two structures using TM-align algorithm
# - Return transformation matrix, aligned coordinates
# - Compute per-residue distance after alignment

class StructuralAligner:
    def align(struct1: Structure, struct2: Structure) -> AlignmentResult
    def apply_transform(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray

@dataclass
class AlignmentResult:
    tm_score: float           # TM-score (0-1, >0.5 = same fold)
    rmsd: float               # Global RMSD in Ångströms
    aligned_length: int       # Number of aligned residues
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    residue_mapping: dict     # Correspondence between residues
```

**Implementation options:**
1. `tmtools` Python package (pure Python TM-align)
2. Subprocess call to TM-align binary (faster for large structures)
3. BioPython `Superimposer` (simpler but less robust)

**Recommendation:** Use `tmtools` for portability, with option to use binary for performance.

---

### Step 4: RMSD and TM-score Metrics

**File: `core/metrics.py`**

```python
# Key functionality:
# - Calculate global RMSD
# - Calculate per-residue RMSD
# - Confidence-weighted RMSD (novel metric for predicted structures)
# - TM-score normalization options

class MetricsCalculator:
    def global_rmsd(coords1, coords2) -> float
    def per_residue_rmsd(coords1, coords2) -> np.ndarray
    def weighted_rmsd(coords1, coords2, weights) -> float
    def tm_score(alignment: AlignmentResult, length: int) -> float

def confidence_weights(plddt1: np.ndarray, plddt2: np.ndarray) -> np.ndarray:
    """Weight by minimum confidence of corresponding residues"""
    return np.minimum(plddt1, plddt2) / 100.0
```

**Confidence-weighted RMSD formula:**
```
RMSD_weighted = sqrt(sum(w_i * d_i^2) / sum(w_i))
where w_i = min(pLDDT_1[i], pLDDT_2[i]) / 100
```

---

### Step 5: Secondary Structure Comparison

**File: `core/secondary.py`**

```python
# Key functionality:
# - Run DSSP on structures to assign secondary structure
# - Compare SS assignments between structures
# - Calculate SS agreement score

class SecondaryStructureAnalyzer:
    def assign_ss(structure: Structure) -> list[str]  # H, E, C per residue
    def compare_ss(ss1: list, ss2: list) -> SSComparisonResult
    def ss_agreement_score(ss1: list, ss2: list) -> float

@dataclass
class SSComparisonResult:
    agreement_score: float    # 0-1, fraction of matching assignments
    helix_agreement: float    # Agreement in helical regions
    sheet_agreement: float    # Agreement in sheet regions
    transitions: list         # Positions where SS differs
```

**DSSP codes:**
- H = α-helix, G = 3₁₀-helix, I = π-helix
- E = β-strand, B = β-bridge
- T = turn, S = bend, C = coil/loop

**Simplified 3-state:** Helix (H,G,I) → H, Sheet (E,B) → E, Coil (others) → C

---

### Step 6: Contact Map Generation & Comparison

**File: `core/contacts.py`**

```python
# Key functionality:
# - Generate residue-residue contact maps (Cα-Cα or Cβ-Cβ)
# - Compare contact maps between structures
# - Identify gained/lost contacts

class ContactMapAnalyzer:
    def compute_contact_map(structure: Structure,
                           cutoff: float = 8.0,
                           atom: str = "CA") -> np.ndarray

    def compare_contacts(map1: np.ndarray,
                        map2: np.ndarray) -> ContactComparison

    def contact_similarity(map1, map2) -> float  # Jaccard index

@dataclass
class ContactComparison:
    shared_contacts: int
    only_in_1: int
    only_in_2: int
    jaccard_score: float      # Intersection / Union
    difference_map: np.ndarray
```

**Contact definition:** Two residues in contact if Cα-Cα distance < 8Å (adjustable)

---

### Step 7: Batch Comparison Engine

**File: `core/batch.py`**

```python
# Key functionality:
# - Compare N structures pairwise or all-vs-all
# - Parallel processing for large batches
# - Progress tracking and resumption

class BatchComparator:
    def __init__(self, structures: list[Structure],
                 reference: Structure = None)

    def compare_all_pairs(self, n_jobs: int = -1) -> pd.DataFrame
    def compare_to_reference(self, n_jobs: int = -1) -> pd.DataFrame

    def get_summary_statistics(self) -> dict

# Output DataFrame columns:
# struct1, struct2, tm_score, rmsd, weighted_rmsd,
# ss_agreement, contact_jaccard, divergent_regions
```

---

### Step 8: Visualization

**File: `visualization/alignment_viz.py`**

```python
# Key functionality:
# - Generate PyMOL/ChimeraX scripts for 3D visualization
# - Color by per-residue RMSD or divergence
# - Highlight low-confidence regions

class AlignmentVisualizer:
    def generate_pymol_script(aligned_structures,
                             color_by: str = "rmsd") -> str

    def save_aligned_pdb(aligned: AlignmentResult,
                        output_path: str)

    def divergence_plot(per_residue_rmsd: np.ndarray,
                       plddt: np.ndarray) -> plt.Figure
```

**File: `visualization/contact_maps.py`**

```python
class ContactMapVisualizer:
    def plot_single_map(contact_map: np.ndarray) -> plt.Figure
    def plot_comparison(map1, map2, diff_map) -> plt.Figure
    def plot_contact_difference_heatmap(...) -> plt.Figure
```

---

### Step 9: CLI Interface

**File: `cli.py`**

```python
# Command-line interface using argparse or click

# Usage examples:
# python -m protein_compare compare struct1.pdb struct2.pdb
# python -m protein_compare batch *.pdb --reference ref.pdb -o results.csv
# python -m protein_compare visualize struct1.pdb struct2.pdb --output aligned.pml

@click.group()
def cli():
    pass

@cli.command()
@click.argument('structures', nargs=-1)
@click.option('--reference', '-r', help='Reference structure')
@click.option('--output', '-o', default='results.csv')
@click.option('--contact-cutoff', default=8.0)
@click.option('--confidence-weighted/--no-confidence-weighted', default=True)
@click.option('--jobs', '-j', default=-1, help='Parallel jobs')
def batch(structures, reference, output, **kwargs):
    """Compare multiple structures in batch mode"""
    ...
```

---

### Step 10: Report Generation

**File: `io/reporter.py`**

```python
class ComparisonReporter:
    def to_csv(results: pd.DataFrame, path: str)
    def to_json(results: pd.DataFrame, path: str)
    def generate_html_report(results, figures) -> str

    def summary_report(results: pd.DataFrame) -> str:
        """Text summary with key statistics"""
```

**Output CSV columns:**
```
structure_1, structure_2, tm_score, rmsd, weighted_rmsd,
aligned_length, ss_agreement, helix_agreement, sheet_agreement,
contact_jaccard, shared_contacts, divergent_residues,
mean_plddt_1, mean_plddt_2
```

---

## Key Algorithms

### TM-score Calculation
```
TM-score = max[ 1/L × Σ 1/(1 + (d_i/d_0)²) ]

where:
- L = length of shorter protein
- d_i = distance between aligned residues after superposition
- d_0 = 1.24 × ³√(L-15) - 1.8 (length-dependent normalization)
```

### Confidence-Weighted Metrics
For predicted structures, weight all metrics by prediction confidence:
- Low-confidence regions (pLDDT < 50) contribute less to RMSD
- Divergence in low-confidence regions flagged separately

---

## Dependencies Summary

| Package | Purpose |
|---------|---------|
| BioPython | PDB parsing, DSSP wrapper |
| tmtools | TM-align algorithm |
| numpy/scipy | Numerical operations |
| pandas | Results tabulation |
| matplotlib/seaborn | Visualization |
| click | CLI interface |
| joblib | Parallel processing |

---

## Testing Strategy

1. **Unit tests** for each module
2. **Integration tests** with sample AlphaFold structures
3. **Validation** against published TM-align results
4. **Benchmark** on AF2 database subset

---

## Files to Create (in order)

1. `requirements.txt` - Dependencies
2. `protein_compare/__init__.py` - Package init
3. `protein_compare/io/parser.py` - PDB loading
4. `protein_compare/core/alignment.py` - TM-align wrapper
5. `protein_compare/core/metrics.py` - RMSD/TM-score
6. `protein_compare/core/secondary.py` - DSSP integration
7. `protein_compare/core/contacts.py` - Contact maps
8. `protein_compare/core/confidence.py` - pLDDT handling
9. `protein_compare/core/batch.py` - Batch comparison
10. `protein_compare/visualization/` - All viz modules
11. `protein_compare/io/reporter.py` - Output generation
12. `protein_compare/cli.py` - CLI interface
13. `tests/` - Test suite
