"""Batch comparison engine for multiple protein structures.

Provides parallel processing of pairwise structure comparisons
with progress tracking and result aggregation.
"""

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional, Callable
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from protein_compare.io.parser import ProteinStructure, StructureLoader
from protein_compare.core.alignment import StructuralAligner, AlignmentResult
from protein_compare.core.metrics import MetricsCalculator
from protein_compare.core.secondary import SecondaryStructureAnalyzer
from protein_compare.core.contacts import ContactMapAnalyzer
from protein_compare.core.confidence import ConfidenceAnalyzer


@dataclass
class PairwiseResult:
    """Result of comparing two structures."""

    struct1_name: str
    struct2_name: str
    tm_score: float
    tm_score_1: float
    tm_score_2: float
    rmsd: float
    weighted_rmsd: float
    aligned_length: int
    seq_identity: float
    ss_agreement: float
    contact_jaccard: float
    gdt_ts: float
    gdt_ha: float
    mean_plddt_1: float
    mean_plddt_2: float
    n_divergent_residues: int
    alignment: Optional[AlignmentResult] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "structure_1": self.struct1_name,
            "structure_2": self.struct2_name,
            "tm_score": self.tm_score,
            "tm_score_1": self.tm_score_1,
            "tm_score_2": self.tm_score_2,
            "rmsd": self.rmsd,
            "weighted_rmsd": self.weighted_rmsd,
            "aligned_length": self.aligned_length,
            "seq_identity": self.seq_identity,
            "ss_agreement": self.ss_agreement,
            "contact_jaccard": self.contact_jaccard,
            "gdt_ts": self.gdt_ts,
            "gdt_ha": self.gdt_ha,
            "mean_plddt_1": self.mean_plddt_1,
            "mean_plddt_2": self.mean_plddt_2,
            "n_divergent_residues": self.n_divergent_residues,
        }


class BatchComparator:
    """Compare multiple protein structures in batch mode."""

    def __init__(
        self,
        structures: Optional[list[ProteinStructure]] = None,
        reference: Optional[ProteinStructure] = None,
        contact_cutoff: float = 8.0,
        confidence_weighted: bool = True,
        compute_ss: bool = True,
        compute_contacts: bool = True,
        divergence_threshold: float = 3.0,
    ):
        """Initialize the batch comparator.

        Args:
            structures: List of structures to compare.
            reference: Optional reference structure for all-vs-reference mode.
            contact_cutoff: Distance cutoff for contact maps.
            confidence_weighted: Use pLDDT-weighted RMSD.
            compute_ss: Compute secondary structure comparison.
            compute_contacts: Compute contact map comparison.
            divergence_threshold: RMSD threshold for counting divergent residues.
        """
        self.structures = structures or []
        self.reference = reference
        self.contact_cutoff = contact_cutoff
        self.confidence_weighted = confidence_weighted
        self.compute_ss = compute_ss
        self.compute_contacts = compute_contacts
        self.divergence_threshold = divergence_threshold

        # Initialize analyzers
        self.aligner = StructuralAligner()
        self.metrics = MetricsCalculator()
        self.ss_analyzer = SecondaryStructureAnalyzer()
        self.contact_analyzer = ContactMapAnalyzer(cutoff=contact_cutoff)
        self.confidence_analyzer = ConfidenceAnalyzer()

        # Cache for secondary structures
        self._ss_cache: dict[str, list[str]] = {}

    def add_structure(self, structure: ProteinStructure) -> None:
        """Add a structure to the comparison set.

        Args:
            structure: ProteinStructure to add.
        """
        self.structures.append(structure)

    def add_structures_from_paths(self, paths: list[str | Path]) -> None:
        """Load and add structures from PDB file paths.

        Args:
            paths: List of PDB file paths.
        """
        loader = StructureLoader()
        for path in paths:
            try:
                struct = loader.load(path)
                self.structures.append(struct)
            except Exception as e:
                warnings.warn(f"Failed to load {path}: {e}")

    def set_reference(self, reference: ProteinStructure) -> None:
        """Set the reference structure.

        Args:
            reference: Reference structure for comparisons.
        """
        self.reference = reference

    def _get_ss(self, structure: ProteinStructure) -> list[str]:
        """Get secondary structure, using cache.

        Args:
            structure: ProteinStructure object.

        Returns:
            List of SS codes.
        """
        if structure.name not in self._ss_cache:
            try:
                ss = self.ss_analyzer.assign_ss(structure, simplify=True)
            except Exception:
                ss = ["C"] * structure.n_residues
            self._ss_cache[structure.name] = ss
        return self._ss_cache[structure.name]

    def compare_pair(
        self,
        struct1: ProteinStructure,
        struct2: ProteinStructure,
        store_alignment: bool = False,
    ) -> PairwiseResult:
        """Compare two structures.

        Args:
            struct1: First structure.
            struct2: Second structure.
            store_alignment: Whether to store full alignment result.

        Returns:
            PairwiseResult with comparison metrics.
        """
        # Structural alignment
        alignment = self.aligner.align(struct1, struct2)

        # Get aligned pLDDT scores
        idx1 = [m[0] for m in alignment.residue_mapping]
        idx2 = [m[1] for m in alignment.residue_mapping]
        plddt1_aligned = struct1.plddt[idx1]
        plddt2_aligned = struct2.plddt[idx2]

        # Confidence-weighted RMSD
        if self.confidence_weighted:
            weighted_rmsd = self.metrics.confidence_weighted_rmsd(
                alignment.aligned_coords_1,
                alignment.aligned_coords_2,
                plddt1_aligned,
                plddt2_aligned,
            )
        else:
            weighted_rmsd = alignment.rmsd

        # GDT scores
        gdt_ts = self.metrics.gdt_ts(
            alignment.aligned_coords_1,
            alignment.aligned_coords_2,
        )
        gdt_ha = self.metrics.gdt_ha(
            alignment.aligned_coords_1,
            alignment.aligned_coords_2,
        )

        # Secondary structure comparison
        ss_agreement = 0.0
        if self.compute_ss:
            try:
                ss1 = self._get_ss(struct1)
                ss2 = self._get_ss(struct2)
                ss_result = self.ss_analyzer.compare_ss(
                    ss1, ss2, alignment.residue_mapping
                )
                ss_agreement = ss_result.agreement_score
            except Exception:
                pass

        # Contact map comparison
        contact_jaccard = 0.0
        if self.compute_contacts:
            try:
                contact_result = self.contact_analyzer.compare_structures(
                    struct1, struct2, alignment.residue_mapping
                )
                contact_jaccard = contact_result.jaccard_score
            except Exception:
                pass

        # Count divergent residues
        n_divergent = int(np.sum(
            alignment.per_residue_distance > self.divergence_threshold
        ))

        return PairwiseResult(
            struct1_name=struct1.name,
            struct2_name=struct2.name,
            tm_score=alignment.tm_score,
            tm_score_1=alignment.tm_score_1,
            tm_score_2=alignment.tm_score_2,
            rmsd=alignment.rmsd,
            weighted_rmsd=weighted_rmsd,
            aligned_length=alignment.aligned_length,
            seq_identity=alignment.seq_identity,
            ss_agreement=ss_agreement,
            contact_jaccard=contact_jaccard,
            gdt_ts=gdt_ts,
            gdt_ha=gdt_ha,
            mean_plddt_1=struct1.mean_plddt,
            mean_plddt_2=struct2.mean_plddt,
            n_divergent_residues=n_divergent,
            alignment=alignment if store_alignment else None,
        )

    def compare_all_pairs(
        self,
        n_jobs: int = -1,
        store_alignments: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """Compare all pairs of structures.

        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs).
            store_alignments: Whether to store alignment details.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            DataFrame with comparison results.
        """
        if len(self.structures) < 2:
            raise ValueError("Need at least 2 structures to compare")

        pairs = list(combinations(range(len(self.structures)), 2))
        n_pairs = len(pairs)

        def compare_one(i: int, j: int) -> PairwiseResult:
            return self.compare_pair(
                self.structures[i],
                self.structures[j],
                store_alignments,
            )

        # Parallel execution
        if n_jobs == 1:
            results = []
            for idx, (i, j) in enumerate(pairs):
                results.append(compare_one(i, j))
                if progress_callback:
                    progress_callback(idx + 1, n_pairs)
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(compare_one)(i, j) for i, j in pairs
            )

        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in results])
        return df

    def compare_to_reference(
        self,
        n_jobs: int = -1,
        store_alignments: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        """Compare all structures to the reference.

        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs).
            store_alignments: Whether to store alignment details.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            DataFrame with comparison results.
        """
        if self.reference is None:
            raise ValueError("No reference structure set")

        if len(self.structures) == 0:
            raise ValueError("No structures to compare")

        n_structs = len(self.structures)

        def compare_one(i: int) -> PairwiseResult:
            return self.compare_pair(
                self.reference,
                self.structures[i],
                store_alignments,
            )

        # Parallel execution
        if n_jobs == 1:
            results = []
            for idx in range(n_structs):
                results.append(compare_one(idx))
                if progress_callback:
                    progress_callback(idx + 1, n_structs)
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(compare_one)(i) for i in range(n_structs)
            )

        df = pd.DataFrame([r.to_dict() for r in results])
        return df

    def get_summary_statistics(self, results: pd.DataFrame) -> dict:
        """Calculate summary statistics from results.

        Args:
            results: DataFrame from compare_all_pairs or compare_to_reference.

        Returns:
            Dict with summary statistics.
        """
        return {
            "n_comparisons": len(results),
            "tm_score": {
                "mean": float(results["tm_score"].mean()),
                "std": float(results["tm_score"].std()),
                "min": float(results["tm_score"].min()),
                "max": float(results["tm_score"].max()),
            },
            "rmsd": {
                "mean": float(results["rmsd"].mean()),
                "std": float(results["rmsd"].std()),
                "min": float(results["rmsd"].min()),
                "max": float(results["rmsd"].max()),
            },
            "weighted_rmsd": {
                "mean": float(results["weighted_rmsd"].mean()),
                "std": float(results["weighted_rmsd"].std()),
                "min": float(results["weighted_rmsd"].min()),
                "max": float(results["weighted_rmsd"].max()),
            },
            "ss_agreement": {
                "mean": float(results["ss_agreement"].mean()),
                "std": float(results["ss_agreement"].std()),
            },
            "contact_jaccard": {
                "mean": float(results["contact_jaccard"].mean()),
                "std": float(results["contact_jaccard"].std()),
            },
            "n_same_fold": int((results["tm_score"] >= 0.5).sum()),
            "n_different_fold": int((results["tm_score"] < 0.5).sum()),
        }

    def get_distance_matrix(self, results: pd.DataFrame, metric: str = "rmsd") -> np.ndarray:
        """Build distance matrix from pairwise results.

        Args:
            results: DataFrame from compare_all_pairs.
            metric: Which metric to use (rmsd, tm_score, weighted_rmsd).

        Returns:
            Square distance matrix.
        """
        names = list(set(results["structure_1"]) | set(results["structure_2"]))
        name_to_idx = {name: i for i, name in enumerate(names)}
        n = len(names)

        # Initialize matrix
        if metric == "tm_score":
            # TM-score is similarity, convert to distance
            matrix = np.zeros((n, n))
        else:
            matrix = np.zeros((n, n))

        # Fill matrix
        for _, row in results.iterrows():
            i = name_to_idx[row["structure_1"]]
            j = name_to_idx[row["structure_2"]]
            value = row[metric]

            if metric == "tm_score":
                value = 1.0 - value  # Convert to distance

            matrix[i, j] = value
            matrix[j, i] = value

        return matrix

    def cluster_structures(
        self,
        results: pd.DataFrame,
        metric: str = "tm_score",
        threshold: float = 0.5,
    ) -> list[list[str]]:
        """Cluster structures based on similarity.

        Simple single-linkage clustering.

        Args:
            results: DataFrame from compare_all_pairs.
            metric: Similarity metric to use.
            threshold: Similarity threshold for same cluster.

        Returns:
            List of clusters (each cluster is list of structure names).
        """
        names = list(set(results["structure_1"]) | set(results["structure_2"]))
        name_to_idx = {name: i for i, name in enumerate(names)}

        # Build adjacency based on threshold
        n = len(names)
        adjacent = [set() for _ in range(n)]

        for _, row in results.iterrows():
            i = name_to_idx[row["structure_1"]]
            j = name_to_idx[row["structure_2"]]

            if metric == "tm_score":
                similar = row[metric] >= threshold
            else:
                similar = row[metric] <= threshold

            if similar:
                adjacent[i].add(j)
                adjacent[j].add(i)

        # Find connected components (clusters)
        visited = [False] * n
        clusters = []

        def dfs(node: int, cluster: list[int]) -> None:
            visited[node] = True
            cluster.append(node)
            for neighbor in adjacent[node]:
                if not visited[neighbor]:
                    dfs(neighbor, cluster)

        for i in range(n):
            if not visited[i]:
                cluster: list[int] = []
                dfs(i, cluster)
                clusters.append([names[idx] for idx in cluster])

        return clusters
