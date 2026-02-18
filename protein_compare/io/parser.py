"""PDB file parsing with confidence score extraction.

Handles PDB files from AlphaFold, ESMFold, Chai, and Boltz,
extracting pLDDT confidence scores from the B-factor column.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser, Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue


@dataclass
class PAEData:
    """Container for Predicted Aligned Error (PAE) data from AlphaFold.

    PAE represents the expected position error (in Ångströms) of residue j
    when the structure is aligned on residue i. Low PAE values indicate
    high confidence in the relative positions of residue pairs.
    """

    pae_matrix: np.ndarray  # N×N PAE matrix in Ångströms
    max_pae: float  # Maximum PAE value (typically 31.75 for AF2)
    ptm: Optional[float] = None  # Predicted TM-score
    iptm: Optional[float] = None  # Interface pTM (for multimers)

    @property
    def n_residues(self) -> int:
        """Number of residues."""
        return len(self.pae_matrix)

    @property
    def mean_pae(self) -> float:
        """Mean PAE across all residue pairs."""
        return float(np.mean(self.pae_matrix))

    @property
    def median_pae(self) -> float:
        """Median PAE across all residue pairs."""
        return float(np.median(self.pae_matrix))

    def get_domain_pae(self, domain1_indices: list[int], domain2_indices: list[int]) -> float:
        """Get mean PAE between two sets of residues (potential domains).

        Args:
            domain1_indices: Residue indices for first domain.
            domain2_indices: Residue indices for second domain.

        Returns:
            Mean PAE between the two domains.
        """
        submatrix = self.pae_matrix[np.ix_(domain1_indices, domain2_indices)]
        return float(np.mean(submatrix))

    def identify_domains(self, pae_cutoff: float = 5.0, min_domain_size: int = 20) -> list[list[int]]:
        """Identify potential domains based on PAE clustering.

        Residues within a domain have low PAE to each other but high PAE
        to residues in other domains.

        Args:
            pae_cutoff: PAE threshold for considering residues in same domain.
            min_domain_size: Minimum number of residues to form a domain.

        Returns:
            List of domains, each domain is a list of residue indices.
        """
        n = self.n_residues
        # Create connectivity matrix based on PAE cutoff
        connected = self.pae_matrix < pae_cutoff
        # Make symmetric (use AND to be conservative)
        connected = connected & connected.T

        # Simple clustering: group consecutive residues with mutual low PAE
        domains = []
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            # Start new domain
            domain = [i]
            visited.add(i)

            # Expand domain by adding connected residues
            for j in range(i + 1, n):
                if j in visited:
                    continue
                # Check if j is connected to most residues in current domain
                connections = sum(connected[j, k] for k in domain)
                if connections >= len(domain) * 0.5:  # 50% connectivity threshold
                    domain.append(j)
                    visited.add(j)

            if len(domain) >= min_domain_size:
                domains.append(domain)

        # Merge small gaps between domains
        if not domains:
            return [[i for i in range(n)]]  # Single domain

        return domains


@dataclass
class ProteinStructure:
    """Container for protein structure data with confidence scores."""

    name: str
    ca_coords: np.ndarray  # Cα coordinates, shape (n_residues, 3)
    cb_coords: np.ndarray  # Cβ coordinates, shape (n_residues, 3)
    plddt: np.ndarray  # pLDDT scores per residue, shape (n_residues,)
    residue_ids: list[tuple[str, int]]  # (chain_id, residue_number)
    sequence: str  # One-letter amino acid sequence
    source_path: Optional[Path] = None  # Original PDB file path
    biopython_structure: Optional[Structure] = field(default=None, repr=False)

    @property
    def n_residues(self) -> int:
        """Number of residues in the structure."""
        return len(self.ca_coords)

    @property
    def mean_plddt(self) -> float:
        """Mean pLDDT score across all residues."""
        return float(np.mean(self.plddt))

    @property
    def high_confidence_mask(self) -> np.ndarray:
        """Boolean mask for high-confidence residues (pLDDT >= 70)."""
        return self.plddt >= 70.0

    @property
    def low_confidence_mask(self) -> np.ndarray:
        """Boolean mask for low-confidence residues (pLDDT < 50)."""
        return self.plddt < 50.0


class StructureLoader:
    """Load and parse PDB files with pLDDT extraction."""

    # Standard amino acid 3-letter to 1-letter mapping
    AA_MAP = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        # Non-standard
        "MSE": "M",  # Selenomethionine
        "UNK": "X",  # Unknown
    }

    def __init__(self, quiet: bool = True):
        """Initialize the structure loader.

        Args:
            quiet: Suppress BioPython parser warnings.
        """
        self.parser = PDBParser(QUIET=quiet)

    def load(self, path: str | Path) -> ProteinStructure:
        """Load a PDB file and extract structure data.

        Args:
            path: Path to PDB file.

        Returns:
            ProteinStructure with coordinates and confidence scores.

        Raises:
            FileNotFoundError: If PDB file doesn't exist.
            ValueError: If no valid residues found.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDB file not found: {path}")

        structure = self.parser.get_structure(path.stem, str(path))

        ca_coords = []
        cb_coords = []
        plddt_scores = []
        residue_ids = []
        sequence = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    # Skip hetero residues and water
                    if residue.get_id()[0] != " ":
                        continue

                    resname = residue.get_resname()
                    if resname not in self.AA_MAP:
                        continue

                    # Extract Cα atom
                    if "CA" not in residue:
                        continue
                    ca_atom = residue["CA"]
                    ca_coords.append(ca_atom.get_coord())

                    # Extract Cβ atom (use Cα for glycine)
                    if "CB" in residue:
                        cb_coords.append(residue["CB"].get_coord())
                    else:
                        cb_coords.append(ca_atom.get_coord())

                    # Extract pLDDT from B-factor
                    # AlphaFold/ESMFold store pLDDT (0-100) in B-factor column
                    plddt = ca_atom.get_bfactor()
                    plddt_scores.append(plddt)

                    # Store residue info
                    residue_ids.append((chain.get_id(), residue.get_id()[1]))
                    sequence.append(self.AA_MAP[resname])

            # Only process first model
            break

        if not ca_coords:
            raise ValueError(f"No valid residues found in {path}")

        return ProteinStructure(
            name=path.stem,
            ca_coords=np.array(ca_coords),
            cb_coords=np.array(cb_coords),
            plddt=np.array(plddt_scores),
            residue_ids=residue_ids,
            sequence="".join(sequence),
            source_path=path,
            biopython_structure=structure,
        )

    def load_multiple(self, paths: list[str | Path]) -> list[ProteinStructure]:
        """Load multiple PDB files.

        Args:
            paths: List of paths to PDB files.

        Returns:
            List of ProteinStructure objects.
        """
        return [self.load(p) for p in paths]

    @staticmethod
    def extract_plddt(structure: ProteinStructure) -> np.ndarray:
        """Extract pLDDT scores from a structure.

        Args:
            structure: ProteinStructure object.

        Returns:
            Array of pLDDT scores per residue.
        """
        return structure.plddt.copy()

    @staticmethod
    def get_ca_coords(structure: ProteinStructure) -> np.ndarray:
        """Get Cα coordinates from a structure.

        Args:
            structure: ProteinStructure object.

        Returns:
            Array of Cα coordinates, shape (n_residues, 3).
        """
        return structure.ca_coords.copy()

    @staticmethod
    def get_cb_coords(structure: ProteinStructure) -> np.ndarray:
        """Get Cβ coordinates from a structure.

        Args:
            structure: ProteinStructure object.

        Returns:
            Array of Cβ coordinates, shape (n_residues, 3).
        """
        return structure.cb_coords.copy()

    @staticmethod
    def detect_prediction_source(structure: ProteinStructure) -> str:
        """Attempt to detect the source of the predicted structure.

        Heuristics based on pLDDT distribution and structure metadata.

        Args:
            structure: ProteinStructure object.

        Returns:
            One of: "alphafold", "esmfold", "chai", "boltz", "unknown"
        """
        plddt = structure.plddt

        # AlphaFold typically has pLDDT values with specific characteristics
        # ESMFold tends to have slightly different distribution
        # This is a simplified heuristic

        if np.all((plddt >= 0) & (plddt <= 100)):
            # Check for characteristic distributions
            if np.mean(plddt) > 80 and np.std(plddt) < 15:
                return "alphafold"  # High confidence, low variance typical of AF2
            elif np.mean(plddt) > 60:
                return "esmfold"
            else:
                return "unknown"

        return "unknown"

    @staticmethod
    def detect_structure_type(structure: ProteinStructure) -> str:
        """Detect whether structure is predicted or experimental.

        Uses heuristics based on B-factor/pLDDT distribution:
        - Predicted structures (AlphaFold, ESMFold): B-factor contains pLDDT (0-100)
          with values typically clustered in specific ranges
        - Experimental structures: B-factor contains temperature factors,
          often with different distribution patterns

        Args:
            structure: ProteinStructure object.

        Returns:
            "predicted" or "experimental"
        """
        bfactors = structure.plddt  # stored in plddt field regardless of meaning

        # Heuristics to distinguish predicted vs experimental:

        # 1. Check if values are strictly within 0-100 (pLDDT range)
        if np.any(bfactors < 0) or np.any(bfactors > 100):
            return "experimental"  # B-factors can exceed 100

        # 2. Predicted structures typically have pLDDT values with:
        #    - Most values > 50 (confident regions)
        #    - Values often clustered near 70-95
        #    - Relatively discrete-looking distribution
        mean_val = np.mean(bfactors)
        std_val = np.std(bfactors)
        min_val = np.min(bfactors)
        max_val = np.max(bfactors)

        # 3. Experimental B-factors typically:
        #    - Have lower mean (often 15-40)
        #    - Can have very low values near 0
        #    - Often have values < 20 for well-ordered regions

        # If mean is high (>50) and range is reasonable, likely predicted
        if mean_val > 50 and max_val <= 100:
            # Additional check: predicted structures rarely have very low values
            low_value_fraction = np.sum(bfactors < 20) / len(bfactors)
            if low_value_fraction < 0.1:  # Less than 10% below 20
                return "predicted"

        # If mean is low or many low values, likely experimental
        if mean_val < 40:
            return "experimental"

        # Edge cases: check for characteristic predicted structure patterns
        # pLDDT often has values clustered in 70-95 range
        high_confidence_fraction = np.sum((bfactors >= 70) & (bfactors <= 100)) / len(bfactors)
        if high_confidence_fraction > 0.5:
            return "predicted"

        # Default to experimental if uncertain
        return "experimental"


class PAELoader:
    """Load Predicted Aligned Error (PAE) data from AlphaFold output files."""

    @staticmethod
    def load(path: str | Path) -> PAEData:
        """Load PAE data from an AlphaFold JSON file.

        Supports multiple AlphaFold output formats:
        - *_predicted_aligned_error.json (older format)
        - *_scores.json (newer format, includes pTM/ipTM)
        - Full model output JSON

        Args:
            path: Path to PAE JSON file.

        Returns:
            PAEData object with PAE matrix and scores.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is not recognized.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PAE file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return PAELoader._parse_pae_data(data)

    @staticmethod
    def _parse_pae_data(data: dict | list) -> PAEData:
        """Parse PAE data from various AlphaFold JSON formats.

        Args:
            data: Parsed JSON data.

        Returns:
            PAEData object.
        """
        pae_matrix = None
        max_pae = 31.75  # Default AlphaFold max PAE
        ptm = None
        iptm = None

        # Handle list format (older AlphaFold output)
        if isinstance(data, list):
            # Format: [{"predicted_aligned_error": [[...]], "max_predicted_aligned_error": ...}]
            if len(data) > 0 and "predicted_aligned_error" in data[0]:
                pae_matrix = np.array(data[0]["predicted_aligned_error"])
                max_pae = data[0].get("max_predicted_aligned_error", max_pae)
            else:
                raise ValueError("Unrecognized PAE list format")

        # Handle dict format (newer formats)
        elif isinstance(data, dict):
            # Format 1: {"pae": [[...]], "plddt": [...], "ptm": ..., "iptm": ...}
            if "pae" in data:
                pae_matrix = np.array(data["pae"])
                ptm = data.get("ptm")
                iptm = data.get("iptm")
                max_pae = data.get("max_pae", max_pae)

            # Format 2: {"predicted_aligned_error": [[...]], ...}
            elif "predicted_aligned_error" in data:
                pae_matrix = np.array(data["predicted_aligned_error"])
                max_pae = data.get("max_predicted_aligned_error", max_pae)
                ptm = data.get("ptm")
                iptm = data.get("iptm")

            # Format 3: Nested under model key
            elif any(key.startswith("model_") for key in data.keys()):
                # Take the first model
                for key in data:
                    if key.startswith("model_") and "pae" in data[key]:
                        pae_matrix = np.array(data[key]["pae"])
                        ptm = data[key].get("ptm")
                        iptm = data[key].get("iptm")
                        break

            # Format 4: distance_matrix style (residue1, residue2, distance format)
            elif "residue1" in data and "residue2" in data and "distance" in data:
                # Sparse format used by some tools
                res1 = np.array(data["residue1"])
                res2 = np.array(data["residue2"])
                dist = np.array(data["distance"])
                n = max(max(res1), max(res2))
                pae_matrix = np.zeros((n, n))
                for i, j, d in zip(res1, res2, dist):
                    pae_matrix[i-1, j-1] = d  # Convert to 0-indexed

            else:
                raise ValueError(f"Unrecognized PAE dict format. Keys: {list(data.keys())}")

        else:
            raise ValueError(f"Unexpected PAE data type: {type(data)}")

        if pae_matrix is None:
            raise ValueError("Could not extract PAE matrix from data")

        return PAEData(
            pae_matrix=pae_matrix,
            max_pae=max_pae,
            ptm=ptm,
            iptm=iptm,
        )

    @staticmethod
    def find_pae_file(structure_path: str | Path) -> Optional[Path]:
        """Try to find a PAE file associated with a structure file.

        Searches for common AlphaFold PAE file naming patterns.

        Args:
            structure_path: Path to PDB/CIF structure file.

        Returns:
            Path to PAE file if found, None otherwise.
        """
        structure_path = Path(structure_path)
        base_dir = structure_path.parent
        stem = structure_path.stem

        # Common PAE file patterns
        patterns = [
            f"{stem}_predicted_aligned_error.json",
            f"{stem}_scores.json",
            f"{stem}_pae.json",
            f"{stem}.pae.json",
            # AlphaFold DB patterns
            f"AF-{stem}-F1-predicted_aligned_error_v4.json",
            # ColabFold patterns
            f"{stem}_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000_scores.json",
        ]

        # Also try removing common suffixes
        for suffix in ["_relaxed", "_unrelaxed", "_model_1", "_rank_001"]:
            if stem.endswith(suffix):
                base_stem = stem[:-len(suffix)]
                patterns.append(f"{base_stem}_scores.json")
                patterns.append(f"{base_stem}_predicted_aligned_error.json")

        for pattern in patterns:
            pae_path = base_dir / pattern
            if pae_path.exists():
                return pae_path

        # Search for any JSON file with "pae" or "scores" in the name
        for json_file in base_dir.glob("*.json"):
            name_lower = json_file.name.lower()
            if "pae" in name_lower or "scores" in name_lower or "aligned_error" in name_lower:
                # Verify it contains PAE data
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    # Quick check for PAE-like structure
                    if isinstance(data, list) and len(data) > 0:
                        if "predicted_aligned_error" in data[0]:
                            return json_file
                    elif isinstance(data, dict):
                        if "pae" in data or "predicted_aligned_error" in data:
                            return json_file
                except (json.JSONDecodeError, KeyError):
                    continue

        return None
