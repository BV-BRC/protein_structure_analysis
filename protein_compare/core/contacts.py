"""Contact map generation and comparison.

Provides functionality to compute residue-residue contact maps
and compare them between protein structures.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.spatial.distance import cdist

from protein_compare.io.parser import ProteinStructure


@dataclass
class ContactComparison:
    """Result of contact map comparison."""

    shared_contacts: int  # Contacts present in both
    only_in_1: int  # Contacts only in structure 1
    only_in_2: int  # Contacts only in structure 2
    jaccard_score: float  # Intersection / Union
    precision: float  # Shared / (Shared + Only_in_2)
    recall: float  # Shared / (Shared + Only_in_1)
    f1_score: float  # Harmonic mean of precision and recall
    difference_map: np.ndarray  # Signed difference map
    contact_map_1: np.ndarray  # Contact map of structure 1
    contact_map_2: np.ndarray  # Contact map of structure 2

    @property
    def total_contacts_1(self) -> int:
        """Total contacts in structure 1."""
        return self.shared_contacts + self.only_in_1

    @property
    def total_contacts_2(self) -> int:
        """Total contacts in structure 2."""
        return self.shared_contacts + self.only_in_2


class ContactMapAnalyzer:
    """Analyze and compare residue-residue contact maps."""

    def __init__(
        self,
        cutoff: float = 8.0,
        atom: Literal["CA", "CB"] = "CA",
        min_seq_sep: int = 4,
    ):
        """Initialize the contact map analyzer.

        Args:
            cutoff: Distance cutoff in Ångströms for defining contacts.
            atom: Which atom to use (CA for Cα, CB for Cβ).
            min_seq_sep: Minimum sequence separation for contacts.
        """
        self.cutoff = cutoff
        self.atom = atom
        self.min_seq_sep = min_seq_sep

    def compute_distance_matrix(
        self,
        structure: ProteinStructure,
    ) -> np.ndarray:
        """Compute distance matrix from structure.

        Args:
            structure: ProteinStructure object.

        Returns:
            Distance matrix, shape (n_residues, n_residues).
        """
        if self.atom == "CA":
            coords = structure.ca_coords
        else:
            coords = structure.cb_coords

        return cdist(coords, coords)

    def compute_contact_map(
        self,
        structure: ProteinStructure,
        cutoff: Optional[float] = None,
    ) -> np.ndarray:
        """Compute binary contact map from structure.

        Args:
            structure: ProteinStructure object.
            cutoff: Optional distance cutoff override.

        Returns:
            Binary contact map, shape (n_residues, n_residues).
        """
        cutoff = cutoff or self.cutoff
        dm = self.compute_distance_matrix(structure)
        n = len(dm)

        # Create contact map
        contact_map = (dm < cutoff).astype(np.int8)

        # Apply minimum sequence separation
        for i in range(n):
            for j in range(n):
                if abs(i - j) < self.min_seq_sep:
                    contact_map[i, j] = 0

        return contact_map

    def compute_contact_map_from_coords(
        self,
        coords: np.ndarray,
        cutoff: Optional[float] = None,
    ) -> np.ndarray:
        """Compute binary contact map from coordinates.

        Args:
            coords: Coordinate array, shape (n, 3).
            cutoff: Optional distance cutoff override.

        Returns:
            Binary contact map, shape (n, n).
        """
        cutoff = cutoff or self.cutoff
        dm = cdist(coords, coords)
        n = len(dm)

        contact_map = (dm < cutoff).astype(np.int8)

        for i in range(n):
            for j in range(n):
                if abs(i - j) < self.min_seq_sep:
                    contact_map[i, j] = 0

        return contact_map

    def compare_contacts(
        self,
        map1: np.ndarray,
        map2: np.ndarray,
    ) -> ContactComparison:
        """Compare two contact maps.

        Args:
            map1: First contact map (binary).
            map2: Second contact map (binary).

        Returns:
            ContactComparison with comparison metrics.

        Raises:
            ValueError: If maps have different shapes.
        """
        if map1.shape != map2.shape:
            raise ValueError(
                f"Contact maps must have same shape: {map1.shape} vs {map2.shape}"
            )

        # Use upper triangle only (symmetric)
        n = map1.shape[0]
        triu_indices = np.triu_indices(n, k=self.min_seq_sep)

        contacts1 = map1[triu_indices] > 0
        contacts2 = map2[triu_indices] > 0

        # Calculate overlaps
        shared = np.sum(contacts1 & contacts2)
        only_1 = np.sum(contacts1 & ~contacts2)
        only_2 = np.sum(~contacts1 & contacts2)

        total = shared + only_1 + only_2

        # Jaccard similarity
        jaccard = shared / total if total > 0 else 1.0

        # Precision, recall, F1
        precision = shared / (shared + only_2) if (shared + only_2) > 0 else 1.0
        recall = shared / (shared + only_1) if (shared + only_1) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Difference map: 1 = only in 1, -1 = only in 2, 0 = same
        diff_map = map1.astype(np.int8) - map2.astype(np.int8)

        return ContactComparison(
            shared_contacts=int(shared),
            only_in_1=int(only_1),
            only_in_2=int(only_2),
            jaccard_score=float(jaccard),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            difference_map=diff_map,
            contact_map_1=map1,
            contact_map_2=map2,
        )

    def compare_structures(
        self,
        struct1: ProteinStructure,
        struct2: ProteinStructure,
        residue_mapping: Optional[list[tuple[int, int]]] = None,
    ) -> ContactComparison:
        """Compare contact maps of two structures.

        Args:
            struct1: First protein structure.
            struct2: Second protein structure.
            residue_mapping: Optional residue correspondence.

        Returns:
            ContactComparison with comparison metrics.
        """
        if residue_mapping:
            # Extract aligned residues
            idx1 = [m[0] for m in residue_mapping]
            idx2 = [m[1] for m in residue_mapping]

            if self.atom == "CA":
                coords1 = struct1.ca_coords[idx1]
                coords2 = struct2.ca_coords[idx2]
            else:
                coords1 = struct1.cb_coords[idx1]
                coords2 = struct2.cb_coords[idx2]

            map1 = self.compute_contact_map_from_coords(coords1)
            map2 = self.compute_contact_map_from_coords(coords2)
        else:
            map1 = self.compute_contact_map(struct1)
            map2 = self.compute_contact_map(struct2)

        return self.compare_contacts(map1, map2)

    @staticmethod
    def contact_similarity(
        map1: np.ndarray,
        map2: np.ndarray,
    ) -> float:
        """Calculate contact similarity (Jaccard index).

        Args:
            map1: First contact map.
            map2: Second contact map.

        Returns:
            Jaccard similarity score (0-1).
        """
        if map1.shape != map2.shape:
            raise ValueError("Contact maps must have same shape")

        # Use upper triangle
        n = map1.shape[0]
        triu_indices = np.triu_indices(n, k=1)

        c1 = map1[triu_indices] > 0
        c2 = map2[triu_indices] > 0

        intersection = np.sum(c1 & c2)
        union = np.sum(c1 | c2)

        return float(intersection / union) if union > 0 else 1.0

    @staticmethod
    def contact_order(contact_map: np.ndarray) -> float:
        """Calculate relative contact order.

        Contact order = (1/N) × Σ |i - j| × C_ij
        where C_ij is 1 if residues i,j are in contact.

        Args:
            contact_map: Binary contact map.

        Returns:
            Relative contact order.
        """
        n = contact_map.shape[0]
        total_contacts = 0
        weighted_sum = 0

        for i in range(n):
            for j in range(i + 1, n):
                if contact_map[i, j] > 0:
                    total_contacts += 1
                    weighted_sum += abs(j - i)

        if total_contacts == 0:
            return 0.0

        return weighted_sum / (total_contacts * n)

    def get_contacts_list(
        self,
        contact_map: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Get list of contacting residue pairs.

        Args:
            contact_map: Binary contact map.

        Returns:
            List of (residue_i, residue_j) pairs.
        """
        n = contact_map.shape[0]
        contacts = []

        for i in range(n):
            for j in range(i + self.min_seq_sep, n):
                if contact_map[i, j] > 0:
                    contacts.append((i, j))

        return contacts

    def long_range_contacts(
        self,
        contact_map: np.ndarray,
        min_sep: int = 24,
    ) -> int:
        """Count long-range contacts.

        Args:
            contact_map: Binary contact map.
            min_sep: Minimum sequence separation for long-range.

        Returns:
            Number of long-range contacts.
        """
        n = contact_map.shape[0]
        count = 0

        for i in range(n):
            for j in range(i + min_sep, n):
                if contact_map[i, j] > 0:
                    count += 1

        return count

    def contact_density(
        self,
        contact_map: np.ndarray,
    ) -> float:
        """Calculate contact density.

        Args:
            contact_map: Binary contact map.

        Returns:
            Fraction of possible contacts that are present.
        """
        n = contact_map.shape[0]

        # Maximum possible contacts with sequence separation constraint
        max_contacts = (n - self.min_seq_sep) * (n - self.min_seq_sep + 1) // 2

        if max_contacts == 0:
            return 0.0

        actual = np.sum(contact_map[np.triu_indices(n, k=self.min_seq_sep)])
        return float(actual / max_contacts)


def compare_distance_matrices(
    dm1: np.ndarray,
    dm2: np.ndarray,
) -> dict:
    """Compare two distance matrices.

    Args:
        dm1: First distance matrix.
        dm2: Second distance matrix.

    Returns:
        Dict with comparison metrics.
    """
    if dm1.shape != dm2.shape:
        raise ValueError("Distance matrices must have same shape")

    n = dm1.shape[0]
    triu_indices = np.triu_indices(n, k=1)

    d1 = dm1[triu_indices]
    d2 = dm2[triu_indices]

    diff = d1 - d2

    return {
        "mean_distance_diff": float(np.mean(np.abs(diff))),
        "max_distance_diff": float(np.max(np.abs(diff))),
        "rmsd_distances": float(np.sqrt(np.mean(diff ** 2))),
        "correlation": float(np.corrcoef(d1, d2)[0, 1]),
    }
