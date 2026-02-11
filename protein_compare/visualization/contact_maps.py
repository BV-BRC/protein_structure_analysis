"""Contact map visualization.

Provides plotting functions for residue-residue contact maps
and their comparisons.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from protein_compare.core.contacts import ContactComparison


class ContactMapVisualizer:
    """Visualize contact maps and their comparisons."""

    def __init__(
        self,
        cmap_single: str = "Blues",
        cmap_diff: str = "RdBu_r",
    ):
        """Initialize the visualizer.

        Args:
            cmap_single: Colormap for single contact maps.
            cmap_diff: Colormap for difference maps.
        """
        self.cmap_single = cmap_single
        self.cmap_diff = cmap_diff

    def plot_single_map(
        self,
        contact_map: np.ndarray,
        title: str = "Contact Map",
        figsize: tuple[int, int] = (8, 8),
        show_diagonal: bool = False,
    ) -> plt.Figure:
        """Plot a single contact map.

        Args:
            contact_map: Binary contact map array.
            title: Plot title.
            figsize: Figure size.
            show_diagonal: Whether to show diagonal.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Mask diagonal if requested
        plot_data = contact_map.copy().astype(float)
        if not show_diagonal:
            np.fill_diagonal(plot_data, np.nan)

        im = ax.imshow(
            plot_data,
            cmap=self.cmap_single,
            aspect='equal',
            origin='lower',
            interpolation='nearest',
        )

        ax.set_xlabel('Residue index')
        ax.set_ylabel('Residue index')
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Contact')

        # Add contact count
        n_contacts = np.sum(contact_map[np.triu_indices(len(contact_map), k=1)])
        ax.text(0.02, 0.98, f'Contacts: {n_contacts}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_comparison(
        self,
        comparison: ContactComparison,
        title: str = "Contact Map Comparison",
        figsize: tuple[int, int] = (16, 6),
    ) -> plt.Figure:
        """Plot contact map comparison (two maps and their difference).

        Args:
            comparison: ContactComparison result.
            title: Plot title.
            figsize: Figure size.

        Returns:
            Matplotlib Figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Structure 1 contacts
        ax1 = axes[0]
        im1 = ax1.imshow(
            comparison.contact_map_1,
            cmap=self.cmap_single,
            aspect='equal',
            origin='lower',
        )
        ax1.set_title('Structure 1')
        ax1.set_xlabel('Residue')
        ax1.set_ylabel('Residue')

        # Structure 2 contacts
        ax2 = axes[1]
        im2 = ax2.imshow(
            comparison.contact_map_2,
            cmap=self.cmap_single,
            aspect='equal',
            origin='lower',
        )
        ax2.set_title('Structure 2')
        ax2.set_xlabel('Residue')

        # Difference map
        ax3 = axes[2]
        im3 = ax3.imshow(
            comparison.difference_map,
            cmap=self.cmap_diff,
            aspect='equal',
            origin='lower',
            vmin=-1,
            vmax=1,
        )
        ax3.set_title('Difference (1 - 2)')
        ax3.set_xlabel('Residue')

        # Add colorbar for difference
        cbar = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['Only in 2', 'Both/Neither', 'Only in 1'])

        # Add statistics
        stats_text = (
            f"Shared: {comparison.shared_contacts}\n"
            f"Only in 1: {comparison.only_in_1}\n"
            f"Only in 2: {comparison.only_in_2}\n"
            f"Jaccard: {comparison.jaccard_score:.3f}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def plot_contact_difference_heatmap(
        self,
        comparison: ContactComparison,
        title: str = "Contact Difference Heatmap",
        figsize: tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """Plot enhanced difference heatmap with annotations.

        Args:
            comparison: ContactComparison result.
            title: Plot title.
            figsize: Figure size.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create RGB image for better visualization
        n = comparison.difference_map.shape[0]
        rgb_map = np.ones((n, n, 3))  # White background

        # Red for only in structure 1
        only_1 = comparison.difference_map == 1
        rgb_map[only_1] = [1, 0.3, 0.3]  # Light red

        # Blue for only in structure 2
        only_2 = comparison.difference_map == -1
        rgb_map[only_2] = [0.3, 0.3, 1]  # Light blue

        # Green for shared contacts
        shared = (comparison.contact_map_1 == 1) & (comparison.contact_map_2 == 1)
        rgb_map[shared] = [0.3, 0.8, 0.3]  # Light green

        ax.imshow(rgb_map, aspect='equal', origin='lower')

        # Legend
        legend_elements = [
            Patch(facecolor=[0.3, 0.8, 0.3], label=f'Shared ({comparison.shared_contacts})'),
            Patch(facecolor=[1, 0.3, 0.3], label=f'Only in 1 ({comparison.only_in_1})'),
            Patch(facecolor=[0.3, 0.3, 1], label=f'Only in 2 ({comparison.only_in_2})'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_xlabel('Residue index')
        ax.set_ylabel('Residue index')
        ax.set_title(f'{title}\nJaccard similarity: {comparison.jaccard_score:.3f}')

        plt.tight_layout()
        return fig

    def plot_contact_order(
        self,
        contact_map: np.ndarray,
        title: str = "Contact Order Distribution",
        figsize: tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Plot distribution of sequence separations for contacts.

        Args:
            contact_map: Binary contact map.
            title: Plot title.
            figsize: Figure size.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        n = contact_map.shape[0]
        separations = []

        for i in range(n):
            for j in range(i + 1, n):
                if contact_map[i, j] > 0:
                    separations.append(j - i)

        if separations:
            # Histogram
            bins = range(0, max(separations) + 5, 5)
            ax.hist(separations, bins=bins, edgecolor='black', alpha=0.7)

            # Add vertical lines for short/medium/long range
            ax.axvline(x=6, color='green', linestyle='--', label='Short (<6)')
            ax.axvline(x=12, color='orange', linestyle='--', label='Medium (6-12)')
            ax.axvline(x=24, color='red', linestyle='--', label='Long (>24)')

            # Statistics
            short = sum(1 for s in separations if s < 6)
            medium = sum(1 for s in separations if 6 <= s < 12)
            long_range = sum(1 for s in separations if 12 <= s < 24)
            very_long = sum(1 for s in separations if s >= 24)

            stats_text = (
                f"Short (<6): {short}\n"
                f"Medium (6-12): {medium}\n"
                f"Long (12-24): {long_range}\n"
                f"Very long (>24): {very_long}"
            )
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   va='top', ha='right', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Sequence separation')
        ax.set_ylabel('Number of contacts')
        ax.set_title(title)
        ax.legend(loc='upper left')

        plt.tight_layout()
        return fig

    def plot_residue_contact_profile(
        self,
        contact_map: np.ndarray,
        title: str = "Residue Contact Profile",
        figsize: tuple[int, int] = (12, 4),
    ) -> plt.Figure:
        """Plot number of contacts per residue.

        Args:
            contact_map: Binary contact map.
            title: Plot title.
            figsize: Figure size.

        Returns:
            Matplotlib Figure.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Count contacts per residue
        n = contact_map.shape[0]
        contacts_per_residue = np.sum(contact_map, axis=1) - np.diag(contact_map)

        ax.bar(range(n), contacts_per_residue, width=1.0, edgecolor='none')
        ax.axhline(y=np.mean(contacts_per_residue), color='red', linestyle='--',
                  label=f'Mean: {np.mean(contacts_per_residue):.1f}')

        ax.set_xlabel('Residue index')
        ax.set_ylabel('Number of contacts')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, n)

        plt.tight_layout()
        return fig
