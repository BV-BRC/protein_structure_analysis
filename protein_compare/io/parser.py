"""PDB file parsing with confidence score extraction.

Handles PDB files from AlphaFold, ESMFold, Chai, and Boltz,
extracting pLDDT confidence scores from the B-factor column.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser, Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue


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
