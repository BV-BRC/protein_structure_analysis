"""Divergence region identification and visualization.

Provides tools to identify and visualize regions of structural
divergence between protein structures.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from protein_compare.io.parser import ProteinStructure
from protein_compare.core.alignment import AlignmentResult


@dataclass
class DivergentRegion:
    """A region of structural divergence."""

    start: int  # Start residue (aligned position)
    end: int  # End residue (aligned position)
    mean_distance: float  # Mean RMSD in region
    max_distance: float  # Max RMSD in region
    mean_plddt_1: Optional[float] = None  # Mean pLDDT in structure 1
    mean_plddt_2: Optional[float] = None  # Mean pLDDT in structure 2

    @property
    def length(self) -> int:
        """Length of the region."""
        return self.end - self.start

    @property
    def is_low_confidence(self) -> bool:
        """Check if region has low confidence in either structure."""
        if self.mean_plddt_1 is None or self.mean_plddt_2 is None:
            return False
        return self.mean_plddt_1 < 70 or self.mean_plddt_2 < 70


class DivergenceAnalyzer:
    """Identify and analyze regions of structural divergence."""

    def __init__(
        self,
        distance_threshold: float = 3.0,
        min_region_length: int = 3,
    ):
        """Initialize the analyzer.

        Args:
            distance_threshold: RMSD threshold for divergence.
            min_region_length: Minimum length for a divergent region.
        """
        self.distance_threshold = distance_threshold
        self.min_region_length = min_region_length

    def identify_divergent_regions(
        self,
        alignment: AlignmentResult,
        plddt1: Optional[np.ndarray] = None,
        plddt2: Optional[np.ndarray] = None,
    ) -> list[DivergentRegion]:
        """Identify contiguous regions of structural divergence.

        Args:
            alignment: AlignmentResult from structural alignment.
            plddt1: Optional pLDDT scores for structure 1.
            plddt2: Optional pLDDT scores for structure 2.

        Returns:
            List of DivergentRegion objects.
        """
        distances = alignment.per_residue_distance
        divergent = distances > self.distance_threshold

        regions = []
        start = None

        for i, is_div in enumerate(divergent):
            if is_div and start is None:
                start = i
            elif not is_div and start is not None:
                if i - start >= self.min_region_length:
                    region = self._create_region(
                        start, i, distances, alignment.residue_mapping,
                        plddt1, plddt2
                    )
                    regions.append(region)
                start = None

        # Handle region at end
        if start is not None and len(distances) - start >= self.min_region_length:
            region = self._create_region(
                start, len(distances), distances, alignment.residue_mapping,
                plddt1, plddt2
            )
            regions.append(region)

        return regions

    def _create_region(
        self,
        start: int,
        end: int,
        distances: np.ndarray,
        residue_mapping: list[tuple[int, int]],
        plddt1: Optional[np.ndarray],
        plddt2: Optional[np.ndarray],
    ) -> DivergentRegion:
        """Create a DivergentRegion object.

        Args:
            start: Start position.
            end: End position.
            distances: Per-residue distances.
            residue_mapping: Residue correspondence.
            plddt1: pLDDT scores for structure 1.
            plddt2: pLDDT scores for structure 2.

        Returns:
            DivergentRegion object.
        """
        region_dist = distances[start:end]

        mean_plddt_1 = None
        mean_plddt_2 = None

        if plddt1 is not None and plddt2 is not None:
            idx1 = [residue_mapping[i][0] for i in range(start, end)]
            idx2 = [residue_mapping[i][1] for i in range(start, end)]
            mean_plddt_1 = float(np.mean(plddt1[idx1]))
            mean_plddt_2 = float(np.mean(plddt2[idx2]))

        return DivergentRegion(
            start=start,
            end=end,
            mean_distance=float(np.mean(region_dist)),
            max_distance=float(np.max(region_dist)),
            mean_plddt_1=mean_plddt_1,
            mean_plddt_2=mean_plddt_2,
        )

    def get_divergence_summary(self, regions: list[DivergentRegion]) -> dict:
        """Get summary statistics for divergent regions.

        Args:
            regions: List of DivergentRegion objects.

        Returns:
            Dict with summary statistics.
        """
        if not regions:
            return {
                "n_regions": 0,
                "total_residues": 0,
                "mean_distance": 0.0,
                "max_distance": 0.0,
                "n_low_confidence": 0,
            }

        return {
            "n_regions": len(regions),
            "total_residues": sum(r.length for r in regions),
            "mean_distance": float(np.mean([r.mean_distance for r in regions])),
            "max_distance": float(max(r.max_distance for r in regions)),
            "n_low_confidence": sum(1 for r in regions if r.is_low_confidence),
            "regions": [
                {
                    "start": r.start,
                    "end": r.end,
                    "length": r.length,
                    "mean_rmsd": r.mean_distance,
                    "low_confidence": r.is_low_confidence,
                }
                for r in regions
            ],
        }


class DivergenceVisualizer:
    """Visualize structural divergence."""

    def __init__(self):
        """Initialize the visualizer."""
        pass

    def plot_divergence_profile(
        self,
        alignment: AlignmentResult,
        regions: list[DivergentRegion],
        plddt1: Optional[np.ndarray] = None,
        plddt2: Optional[np.ndarray] = None,
        figsize: tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """Create comprehensive divergence profile plot.

        Args:
            alignment: AlignmentResult.
            regions: List of divergent regions.
            plddt1: Optional pLDDT for structure 1.
            plddt2: Optional pLDDT for structure 2.
            figsize: Figure size.

        Returns:
            Matplotlib Figure.
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={'height_ratios': [2, 1, 1]})

        n = len(alignment.per_residue_distance)
        x = range(n)

        # Top: Per-residue distance with highlighted regions
        ax1 = axes[0]
        ax1.bar(x, alignment.per_residue_distance, width=1.0, color='steelblue', alpha=0.7)

        # Highlight divergent regions
        for region in regions:
            rect = Rectangle(
                (region.start - 0.5, 0),
                region.length,
                max(alignment.per_residue_distance) * 1.1,
                facecolor='red' if not region.is_low_confidence else 'orange',
                alpha=0.2,
                edgecolor='darkred',
                linewidth=2,
            )
            ax1.add_patch(rect)

        ax1.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Threshold (3Å)')
        ax1.set_ylabel('Distance (Å)')
        ax1.set_title('Per-residue Structural Divergence')
        ax1.legend(loc='upper right')

        # Middle: pLDDT scores
        ax2 = axes[1]
        if plddt1 is not None and plddt2 is not None:
            idx1 = [m[0] for m in alignment.residue_mapping]
            idx2 = [m[1] for m in alignment.residue_mapping]

            ax2.plot(x, plddt1[idx1], 'b-', alpha=0.7, label='Structure 1')
            ax2.plot(x, plddt2[idx2], 'g-', alpha=0.7, label='Structure 2')
            ax2.axhline(y=70, color='gray', linestyle='--', alpha=0.5)
            ax2.fill_between(x, 0, 50, color='red', alpha=0.1)
            ax2.set_ylabel('pLDDT')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='lower right')
        else:
            ax2.text(0.5, 0.5, 'No confidence scores',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_ylabel('pLDDT')

        # Bottom: Region annotations
        ax3 = axes[2]
        ax3.set_xlim(-0.5, n - 0.5)
        ax3.set_ylim(0, 1)

        for i, region in enumerate(regions):
            color = 'red' if not region.is_low_confidence else 'orange'
            rect = Rectangle(
                (region.start - 0.5, 0.2),
                region.length,
                0.6,
                facecolor=color,
                edgecolor='black',
                alpha=0.7,
            )
            ax3.add_patch(rect)
            # Add label
            ax3.text(
                region.start + region.length / 2,
                0.5,
                f'{region.length}aa\n{region.mean_distance:.1f}Å',
                ha='center', va='center', fontsize=8,
            )

        ax3.set_xlabel('Aligned residue position')
        ax3.set_ylabel('Regions')
        ax3.set_yticks([])

        # Legend for regions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='High-confidence divergent'),
            Patch(facecolor='orange', alpha=0.7, label='Low-confidence divergent'),
        ]
        ax3.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        return fig

    def plot_divergence_heatmap(
        self,
        alignment: AlignmentResult,
        struct1: ProteinStructure,
        struct2: ProteinStructure,
        figsize: tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        """Plot heatmap of divergence along sequence.

        Args:
            alignment: AlignmentResult.
            struct1: First structure.
            struct2: Second structure.
            figsize: Figure size.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        n = len(alignment.per_residue_distance)

        # Create data for heatmap
        data = np.zeros((4, n))

        # Row 0: Distance
        data[0] = alignment.per_residue_distance / 10.0  # Normalize

        # Row 1-2: pLDDT
        idx1 = [m[0] for m in alignment.residue_mapping]
        idx2 = [m[1] for m in alignment.residue_mapping]
        data[1] = struct1.plddt[idx1] / 100.0
        data[2] = struct2.plddt[idx2] / 100.0

        # Row 3: Combined confidence
        data[3] = np.minimum(data[1], data[2])

        im = ax.imshow(data, aspect='auto', cmap='RdYlGn')

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Distance', 'pLDDT (1)', 'pLDDT (2)', 'Min pLDDT'])
        ax.set_xlabel('Aligned residue position')
        ax.set_title('Divergence and Confidence Overview')

        plt.colorbar(im, ax=ax, shrink=0.6)
        plt.tight_layout()
        return fig

    def create_divergence_table(
        self,
        regions: list[DivergentRegion],
        residue_mapping: list[tuple[int, int]],
    ) -> str:
        """Create text table of divergent regions.

        Args:
            regions: List of divergent regions.
            residue_mapping: Residue correspondence.

        Returns:
            Formatted table string.
        """
        if not regions:
            return "No divergent regions identified."

        lines = [
            "Divergent Regions Summary",
            "=" * 70,
            f"{"Region":<10} {"Aligned":<15} {"Length":<8} {"Mean RMSD":<12} {"Confidence":<15}",
            "-" * 70,
        ]

        for i, region in enumerate(regions):
            # Get original residue numbers
            start_1, start_2 = residue_mapping[region.start]
            end_1, end_2 = residue_mapping[region.end - 1]

            aligned_range = f"{region.start}-{region.end}"
            conf_status = "Low" if region.is_low_confidence else "High"

            lines.append(
                f"{i+1:<10} {aligned_range:<15} {region.length:<8} "
                f"{region.mean_distance:<12.2f} {conf_status:<15}"
            )

        lines.append("-" * 70)
        lines.append(f"Total divergent residues: {sum(r.length for r in regions)}")

        return "\n".join(lines)
