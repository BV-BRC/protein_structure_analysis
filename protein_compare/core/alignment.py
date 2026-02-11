"""Structural alignment using TM-align algorithm.

Provides wrapper around tmtools for TM-align structural alignment,
computing TM-scores and transformation matrices.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import tmtools
    HAS_TMTOOLS = True
except ImportError:
    HAS_TMTOOLS = False

from protein_compare.io.parser import ProteinStructure


@dataclass
class AlignmentResult:
    """Result of structural alignment between two proteins."""

    tm_score_1: float  # TM-score normalized by length of structure 1
    tm_score_2: float  # TM-score normalized by length of structure 2
    rmsd: float  # RMSD of aligned residues in Ångströms
    aligned_length: int  # Number of aligned residues
    seq_identity: float  # Sequence identity of aligned residues
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    translation_vector: np.ndarray  # Translation vector (3,)
    aligned_coords_1: np.ndarray  # Aligned coords from structure 1
    aligned_coords_2: np.ndarray  # Aligned coords from structure 2 (transformed)
    residue_mapping: list[tuple[int, int]]  # (idx1, idx2) pairs of aligned residues
    per_residue_distance: np.ndarray  # Distance per aligned residue pair

    @property
    def tm_score(self) -> float:
        """Return the maximum TM-score (more robust metric)."""
        return max(self.tm_score_1, self.tm_score_2)

    @property
    def tm_score_avg(self) -> float:
        """Return average TM-score."""
        return (self.tm_score_1 + self.tm_score_2) / 2

    def is_same_fold(self, threshold: float = 0.5) -> bool:
        """Check if structures share the same fold (TM-score > 0.5)."""
        return self.tm_score >= threshold

    def is_same_superfamily(self, threshold: float = 0.4) -> bool:
        """Check if structures are in same superfamily (TM-score > 0.4)."""
        return self.tm_score >= threshold


class StructuralAligner:
    """Align protein structures using TM-align algorithm."""

    def __init__(self, use_binary: bool = False, binary_path: Optional[str] = None):
        """Initialize the aligner.

        Args:
            use_binary: Use external TM-align binary instead of tmtools.
            binary_path: Path to TM-align binary (if use_binary=True).
        """
        if not HAS_TMTOOLS and not use_binary:
            raise ImportError(
                "tmtools not installed. Install with: pip install tmtools\n"
                "Or set use_binary=True and provide TM-align binary path."
            )
        self.use_binary = use_binary
        self.binary_path = binary_path

    def align(
        self,
        struct1: ProteinStructure,
        struct2: ProteinStructure,
    ) -> AlignmentResult:
        """Align two protein structures.

        Args:
            struct1: First protein structure.
            struct2: Second protein structure.

        Returns:
            AlignmentResult with scores and transformation.
        """
        if self.use_binary:
            return self._align_binary(struct1, struct2)
        return self._align_tmtools(struct1, struct2)

    def _align_tmtools(self, struct1: ProteinStructure, struct2: ProteinStructure) -> AlignmentResult:
        """Align using tmtools Python package."""
        coords1 = struct1.ca_coords
        coords2 = struct2.ca_coords
        seq1 = struct1.sequence
        seq2 = struct2.sequence

        # Run TM-align
        result = tmtools.tm_align(coords1, coords2, seq1, seq2)

        # Extract aligned residue pairs from the alignment
        # tmtools returns seqxA, seqyA (aligned sequences with gaps)
        residue_mapping = self._parse_alignment((result.seqxA, result.seqyA, result.seqM))

        # Get aligned coordinates
        aligned_idx1 = [m[0] for m in residue_mapping]
        aligned_idx2 = [m[1] for m in residue_mapping]

        aligned_coords_1 = coords1[aligned_idx1]
        aligned_coords_2 = coords2[aligned_idx2]

        # Apply transformation to structure 2 to align with structure 1
        # tmtools returns u, t such that: coords1_aligned = coords1 @ u.T + t
        # To get coords2 aligned to coords1: (coords2 - t) @ u
        transformed_coords_2 = (aligned_coords_2 - result.t) @ result.u

        # Calculate per-residue distances
        per_residue_dist = np.linalg.norm(
            aligned_coords_1 - transformed_coords_2, axis=1
        )

        # Calculate sequence identity
        seq_identity = self._calc_seq_identity(seq1, seq2, residue_mapping)

        return AlignmentResult(
            tm_score_1=result.tm_norm_chain1,
            tm_score_2=result.tm_norm_chain2,
            rmsd=result.rmsd,
            aligned_length=len(residue_mapping),
            seq_identity=seq_identity,
            rotation_matrix=result.u,
            translation_vector=result.t,
            aligned_coords_1=aligned_coords_1,
            aligned_coords_2=transformed_coords_2,
            residue_mapping=residue_mapping,
            per_residue_distance=per_residue_dist,
        )

    def _align_binary(self, struct1: ProteinStructure, struct2: ProteinStructure) -> AlignmentResult:
        """Align using external TM-align binary."""
        raise NotImplementedError(
            "Binary TM-align not yet implemented. Use tmtools instead."
        )

    @staticmethod
    def apply_transform(
        coords: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> np.ndarray:
        """Apply rotation and translation to coordinates.

        Args:
            coords: Coordinates to transform, shape (n, 3).
            rotation: 3x3 rotation matrix.
            translation: Translation vector (3,).

        Returns:
            Transformed coordinates.
        """
        return coords @ rotation.T + translation

    @staticmethod
    def _parse_alignment(alignment: tuple) -> list[tuple[int, int]]:
        """Parse alignment tuple to get residue mapping.

        Args:
            alignment: Tuple of (aligned_seq1, aligned_seq2, alignment_string)

        Returns:
            List of (idx1, idx2) tuples for aligned residues.
        """
        seq1_aligned, seq2_aligned, _ = alignment
        mapping = []

        idx1 = 0
        idx2 = 0

        for c1, c2 in zip(seq1_aligned, seq2_aligned):
            if c1 != "-" and c2 != "-":
                mapping.append((idx1, idx2))
            if c1 != "-":
                idx1 += 1
            if c2 != "-":
                idx2 += 1

        return mapping

    @staticmethod
    def _calc_seq_identity(
        seq1: str,
        seq2: str,
        mapping: list[tuple[int, int]],
    ) -> float:
        """Calculate sequence identity of aligned residues.

        Args:
            seq1: Sequence of structure 1.
            seq2: Sequence of structure 2.
            mapping: Aligned residue pairs.

        Returns:
            Sequence identity as fraction (0-1).
        """
        if not mapping:
            return 0.0

        matches = sum(1 for i, j in mapping if seq1[i] == seq2[j])
        return matches / len(mapping)


def calculate_tm_score_manual(
    coords1: np.ndarray,
    coords2: np.ndarray,
    length_normalize: int,
) -> float:
    """Calculate TM-score manually from aligned coordinates.

    TM-score = (1/L) × Σ 1/(1 + (d_i/d_0)²)

    where:
        L = normalization length
        d_i = distance between aligned residues
        d_0 = 1.24 × ³√(L-15) - 1.8

    Args:
        coords1: First set of coordinates, shape (n, 3).
        coords2: Second set of coordinates (aligned), shape (n, 3).
        length_normalize: Length for normalization (typically shorter protein).

    Returns:
        TM-score value.
    """
    if len(coords1) != len(coords2):
        raise ValueError("Coordinate arrays must have same length")

    L = length_normalize
    if L <= 15:
        d0 = 0.5  # For very short proteins
    else:
        d0 = 1.24 * ((L - 15) ** (1/3)) - 1.8
        d0 = max(d0, 0.5)  # Minimum d0

    # Calculate distances
    distances = np.linalg.norm(coords1 - coords2, axis=1)

    # TM-score formula
    tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L

    return float(tm_score)
