"""Helper utility functions."""

import numpy as np
from typing import Optional


def normalize_coords(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center and scale coordinates.

    Args:
        coords: Coordinate array, shape (n, 3).

    Returns:
        Tuple of (normalized_coords, centroid, scale).
    """
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    scale = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    return normalized, centroid, scale


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Calculate optimal rotation matrix using Kabsch algorithm.

    Args:
        P: First coordinate set, shape (n, 3).
        Q: Second coordinate set, shape (n, 3).

    Returns:
        3x3 rotation matrix that minimizes RMSD when applied to P.
    """
    # Center both sets
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)

    # Covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection case
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])

    # Optimal rotation
    R = Vt.T @ D @ U.T

    return R


def superimpose(
    mobile: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Superimpose mobile onto target using Kabsch algorithm.

    Args:
        mobile: Mobile coordinates to transform, shape (n, 3).
        target: Target coordinates (reference), shape (n, 3).

    Returns:
        Tuple of (transformed_coords, rotation_matrix, translation).
    """
    # Get centroids
    mobile_center = np.mean(mobile, axis=0)
    target_center = np.mean(target, axis=0)

    # Center
    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    # Get rotation
    R = kabsch_rotation(mobile_centered, target_centered)

    # Apply transformation
    transformed = mobile_centered @ R.T + target_center

    # Translation = target_center - R @ mobile_center
    translation = target_center - mobile_center @ R.T

    return transformed, R, translation


def sequence_identity(seq1: str, seq2: str) -> float:
    """Calculate sequence identity between two aligned sequences.

    Args:
        seq1: First sequence.
        seq2: Second sequence.

    Returns:
        Sequence identity as fraction (0-1).
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have same length")

    if len(seq1) == 0:
        return 0.0

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
    length = sum(1 for a, b in zip(seq1, seq2) if a != '-' or b != '-')

    return matches / length if length > 0 else 0.0


def pairwise_sequence_alignment(
    seq1: str,
    seq2: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
) -> tuple[str, str, float]:
    """Simple Needleman-Wunsch global alignment.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        match_score: Score for matches.
        mismatch_score: Score for mismatches.
        gap_penalty: Gap penalty.

    Returns:
        Tuple of (aligned_seq1, aligned_seq2, score).
    """
    n, m = len(seq1), len(seq2)

    # Initialize score matrix
    score = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        score[i, 0] = i * gap_penalty
    for j in range(m + 1):
        score[0, j] = j * gap_penalty

    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = match_score if seq1[i-1] == seq2[j-1] else mismatch_score
            score[i, j] = max(
                score[i-1, j-1] + match,
                score[i-1, j] + gap_penalty,
                score[i, j-1] + gap_penalty,
            )

    # Traceback
    aligned1, aligned2 = [], []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match = match_score if seq1[i-1] == seq2[j-1] else mismatch_score
            if score[i, j] == score[i-1, j-1] + match:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                i -= 1
                j -= 1
                continue

        if i > 0 and score[i, j] == score[i-1, j] + gap_penalty:
            aligned1.append(seq1[i-1])
            aligned2.append('-')
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j-1])
            j -= 1

    return ''.join(reversed(aligned1)), ''.join(reversed(aligned2)), score[n, m]


def format_residue_range(residues: list[int]) -> str:
    """Format list of residue numbers as ranges.

    Args:
        residues: List of residue numbers.

    Returns:
        Formatted string like "1-5, 10-15, 20".
    """
    if not residues:
        return ""

    sorted_res = sorted(set(residues))
    ranges = []
    start = sorted_res[0]
    end = sorted_res[0]

    for r in sorted_res[1:]:
        if r == end + 1:
            end = r
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = r

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    return ", ".join(ranges)
