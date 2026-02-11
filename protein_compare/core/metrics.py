"""RMSD and TM-score metrics calculation.

Provides functions for calculating structural similarity metrics
including confidence-weighted RMSD for predicted structures.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from protein_compare.core.alignment import AlignmentResult


@dataclass
class MetricsResult:
    """Container for structure comparison metrics."""

    global_rmsd: float
    weighted_rmsd: float
    per_residue_rmsd: np.ndarray
    tm_score: float
    tm_score_1: float
    tm_score_2: float
    gdt_ts: float  # Global Distance Test - Total Score
    gdt_ha: float  # Global Distance Test - High Accuracy
    max_deviation: float  # Maximum per-residue deviation


class MetricsCalculator:
    """Calculate structural comparison metrics."""

    @staticmethod
    def global_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate global RMSD between two coordinate sets.

        RMSD = sqrt(mean(sum((r1 - r2)^2)))

        Args:
            coords1: First coordinate array, shape (n, 3).
            coords2: Second coordinate array, shape (n, 3).

        Returns:
            RMSD in Ångströms.

        Raises:
            ValueError: If arrays have different lengths.
        """
        if len(coords1) != len(coords2):
            raise ValueError(
                f"Coordinate arrays must have same length: {len(coords1)} vs {len(coords2)}"
            )

        diff = coords1 - coords2
        sq_dist = np.sum(diff ** 2, axis=1)
        return float(np.sqrt(np.mean(sq_dist)))

    @staticmethod
    def per_residue_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Calculate per-residue RMSD (distance) between aligned coordinates.

        Args:
            coords1: First coordinate array, shape (n, 3).
            coords2: Second coordinate array, shape (n, 3).

        Returns:
            Array of per-residue distances in Ångströms.
        """
        if len(coords1) != len(coords2):
            raise ValueError("Coordinate arrays must have same length")

        return np.linalg.norm(coords1 - coords2, axis=1)

    @staticmethod
    def weighted_rmsd(
        coords1: np.ndarray,
        coords2: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Calculate weighted RMSD.

        RMSD_weighted = sqrt(sum(w_i * d_i^2) / sum(w_i))

        Args:
            coords1: First coordinate array, shape (n, 3).
            coords2: Second coordinate array, shape (n, 3).
            weights: Weight per residue, shape (n,).

        Returns:
            Weighted RMSD in Ångströms.
        """
        if len(coords1) != len(coords2) or len(coords1) != len(weights):
            raise ValueError("All arrays must have same length")

        diff = coords1 - coords2
        sq_dist = np.sum(diff ** 2, axis=1)

        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return float("inf")

        return float(np.sqrt(np.sum(weights * sq_dist) / weight_sum))

    @staticmethod
    def confidence_weighted_rmsd(
        coords1: np.ndarray,
        coords2: np.ndarray,
        plddt1: np.ndarray,
        plddt2: np.ndarray,
    ) -> float:
        """Calculate confidence-weighted RMSD using pLDDT scores.

        Weight by minimum confidence of corresponding residues:
        w_i = min(pLDDT_1[i], pLDDT_2[i]) / 100

        Args:
            coords1: First coordinate array, shape (n, 3).
            coords2: Second coordinate array, shape (n, 3).
            plddt1: pLDDT scores for structure 1, shape (n,).
            plddt2: pLDDT scores for structure 2, shape (n,).

        Returns:
            Confidence-weighted RMSD in Ångströms.
        """
        weights = confidence_weights(plddt1, plddt2)
        return MetricsCalculator.weighted_rmsd(coords1, coords2, weights)

    @staticmethod
    def tm_score(
        coords1: np.ndarray,
        coords2: np.ndarray,
        length_normalize: int,
    ) -> float:
        """Calculate TM-score from aligned coordinates.

        TM-score = (1/L) × Σ 1/(1 + (d_i/d_0)²)

        Args:
            coords1: First coordinate array (aligned), shape (n, 3).
            coords2: Second coordinate array (aligned), shape (n, 3).
            length_normalize: Length for normalization.

        Returns:
            TM-score value (0-1).
        """
        if len(coords1) != len(coords2):
            raise ValueError("Coordinate arrays must have same length")

        L = length_normalize
        if L <= 15:
            d0 = 0.5
        else:
            d0 = 1.24 * ((L - 15) ** (1/3)) - 1.8
            d0 = max(d0, 0.5)

        distances = np.linalg.norm(coords1 - coords2, axis=1)
        tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L

        return float(tm_score)

    @staticmethod
    def gdt_score(
        coords1: np.ndarray,
        coords2: np.ndarray,
        cutoffs: list[float],
    ) -> float:
        """Calculate GDT score with given distance cutoffs.

        GDT = (1/n_cutoffs) × Σ (residues within cutoff) / total_residues

        Args:
            coords1: First coordinate array, shape (n, 3).
            coords2: Second coordinate array, shape (n, 3).
            cutoffs: List of distance cutoffs in Ångströms.

        Returns:
            GDT score (0-1).
        """
        distances = np.linalg.norm(coords1 - coords2, axis=1)
        n_residues = len(distances)

        scores = []
        for cutoff in cutoffs:
            within = np.sum(distances <= cutoff)
            scores.append(within / n_residues)

        return float(np.mean(scores))

    @staticmethod
    def gdt_ts(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate GDT-TS (Total Score).

        Uses cutoffs: 1, 2, 4, 8 Å

        Args:
            coords1: First coordinate array, shape (n, 3).
            coords2: Second coordinate array, shape (n, 3).

        Returns:
            GDT-TS score (0-1).
        """
        return MetricsCalculator.gdt_score(coords1, coords2, [1.0, 2.0, 4.0, 8.0])

    @staticmethod
    def gdt_ha(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate GDT-HA (High Accuracy).

        Uses cutoffs: 0.5, 1, 2, 4 Å

        Args:
            coords1: First coordinate array, shape (n, 3).
            coords2: Second coordinate array, shape (n, 3).

        Returns:
            GDT-HA score (0-1).
        """
        return MetricsCalculator.gdt_score(coords1, coords2, [0.5, 1.0, 2.0, 4.0])

    def calculate_all(
        self,
        alignment: AlignmentResult,
        plddt1: Optional[np.ndarray] = None,
        plddt2: Optional[np.ndarray] = None,
        length1: Optional[int] = None,
        length2: Optional[int] = None,
    ) -> MetricsResult:
        """Calculate all metrics from an alignment result.

        Args:
            alignment: AlignmentResult from structural alignment.
            plddt1: pLDDT scores for structure 1 (aligned residues).
            plddt2: pLDDT scores for structure 2 (aligned residues).
            length1: Full length of structure 1 (for TM-score normalization).
            length2: Full length of structure 2 (for TM-score normalization).

        Returns:
            MetricsResult with all calculated metrics.
        """
        coords1 = alignment.aligned_coords_1
        coords2 = alignment.aligned_coords_2

        # Basic RMSD
        global_rmsd = self.global_rmsd(coords1, coords2)
        per_res_rmsd = self.per_residue_rmsd(coords1, coords2)

        # Weighted RMSD
        if plddt1 is not None and plddt2 is not None:
            weighted_rmsd = self.confidence_weighted_rmsd(
                coords1, coords2, plddt1, plddt2
            )
        else:
            weighted_rmsd = global_rmsd

        # TM-scores
        len1 = length1 or len(coords1)
        len2 = length2 or len(coords2)

        tm_score_1 = self.tm_score(coords1, coords2, len1)
        tm_score_2 = self.tm_score(coords1, coords2, len2)

        # GDT scores
        gdt_ts = self.gdt_ts(coords1, coords2)
        gdt_ha = self.gdt_ha(coords1, coords2)

        return MetricsResult(
            global_rmsd=global_rmsd,
            weighted_rmsd=weighted_rmsd,
            per_residue_rmsd=per_res_rmsd,
            tm_score=max(tm_score_1, tm_score_2),
            tm_score_1=tm_score_1,
            tm_score_2=tm_score_2,
            gdt_ts=gdt_ts,
            gdt_ha=gdt_ha,
            max_deviation=float(np.max(per_res_rmsd)),
        )


def confidence_weights(plddt1: np.ndarray, plddt2: np.ndarray) -> np.ndarray:
    """Calculate confidence weights from pLDDT scores.

    Weight by minimum confidence of corresponding residues:
    w_i = min(pLDDT_1[i], pLDDT_2[i]) / 100

    Args:
        plddt1: pLDDT scores for structure 1.
        plddt2: pLDDT scores for structure 2.

    Returns:
        Weight array (0-1 per residue).
    """
    return np.minimum(plddt1, plddt2) / 100.0


def rmsd_from_distance_matrix(
    dm1: np.ndarray,
    dm2: np.ndarray,
) -> float:
    """Calculate RMSD between two distance matrices.

    Useful for comparing internal geometry without alignment.

    Args:
        dm1: First distance matrix, shape (n, n).
        dm2: Second distance matrix, shape (n, n).

    Returns:
        RMSD between distance matrices.
    """
    if dm1.shape != dm2.shape:
        raise ValueError("Distance matrices must have same shape")

    # Use upper triangle only (symmetric matrices)
    n = dm1.shape[0]
    indices = np.triu_indices(n, k=1)

    diff = dm1[indices] - dm2[indices]
    return float(np.sqrt(np.mean(diff ** 2)))


def lddt_score(
    coords1: np.ndarray,
    coords2: np.ndarray,
    cutoff: float = 15.0,
    thresholds: list[float] = [0.5, 1.0, 2.0, 4.0],
) -> float:
    """Calculate lDDT (local Distance Difference Test) score.

    lDDT measures the fraction of preserved local distances.

    Args:
        coords1: Reference coordinates, shape (n, 3).
        coords2: Model coordinates, shape (n, 3).
        cutoff: Distance cutoff for considering residue pairs.
        thresholds: Distance difference thresholds.

    Returns:
        lDDT score (0-1).
    """
    n = len(coords1)
    if n != len(coords2):
        raise ValueError("Coordinate arrays must have same length")

    # Calculate distance matrices
    dm1 = cdist(coords1, coords1)
    dm2 = cdist(coords2, coords2)

    # Find pairs within cutoff in reference
    mask = (dm1 < cutoff) & (np.eye(n) == 0)

    if not np.any(mask):
        return 0.0

    # Calculate fraction preserved for each threshold
    preserved = []
    for thresh in thresholds:
        diff = np.abs(dm1 - dm2)
        within = np.sum((diff < thresh) & mask)
        total = np.sum(mask)
        preserved.append(within / total if total > 0 else 0)

    return float(np.mean(preserved))
