"""Report generation for structure comparison results.

Provides CSV, JSON, and HTML report generation for
batch comparison results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


class ComparisonReporter:
    """Generate reports from structure comparison results."""

    def __init__(self, results: Optional[pd.DataFrame] = None):
        """Initialize the reporter.

        Args:
            results: Optional DataFrame of comparison results.
        """
        self.results = results

    def set_results(self, results: pd.DataFrame) -> None:
        """Set the results DataFrame.

        Args:
            results: DataFrame of comparison results.
        """
        self.results = results

    def to_csv(self, path: str | Path, **kwargs) -> None:
        """Save results to CSV file.

        Args:
            path: Output file path.
            **kwargs: Additional arguments to pandas to_csv.
        """
        if self.results is None:
            raise ValueError("No results to save")

        self.results.to_csv(path, index=False, **kwargs)

    def to_json(
        self,
        path: str | Path,
        include_metadata: bool = True,
        **kwargs,
    ) -> None:
        """Save results to JSON file.

        Args:
            path: Output file path.
            include_metadata: Include generation metadata.
            **kwargs: Additional arguments to json.dump.
        """
        if self.results is None:
            raise ValueError("No results to save")

        output = {
            "comparisons": self.results.to_dict(orient="records"),
        }

        if include_metadata:
            output["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "n_comparisons": len(self.results),
                "tool": "protein_compare",
                "version": "0.1.0",
            }

        with open(path, "w") as f:
            json.dump(output, f, indent=2, **kwargs)

    def to_excel(
        self,
        path: str | Path,
        include_summary: bool = True,
    ) -> None:
        """Save results to Excel file with optional summary sheet.

        Args:
            path: Output file path.
            include_summary: Include summary statistics sheet.
        """
        if self.results is None:
            raise ValueError("No results to save")

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.results.to_excel(writer, sheet_name="Comparisons", index=False)

            if include_summary:
                summary = self.generate_summary_df()
                summary.to_excel(writer, sheet_name="Summary", index=True)

    def generate_summary_df(self) -> pd.DataFrame:
        """Generate summary statistics DataFrame.

        Returns:
            DataFrame with summary statistics.
        """
        if self.results is None:
            raise ValueError("No results to summarize")

        numeric_cols = [
            "tm_score", "rmsd", "weighted_rmsd", "aligned_length",
            "seq_identity", "ss_agreement", "contact_jaccard",
            "gdt_ts", "gdt_ha", "mean_plddt_1", "mean_plddt_2",
        ]

        stats = []
        for col in numeric_cols:
            if col in self.results.columns:
                stats.append({
                    "metric": col,
                    "mean": self.results[col].mean(),
                    "std": self.results[col].std(),
                    "min": self.results[col].min(),
                    "max": self.results[col].max(),
                    "median": self.results[col].median(),
                })

        return pd.DataFrame(stats).set_index("metric")

    def summary_report(self) -> str:
        """Generate text summary report.

        Returns:
            Formatted text report.
        """
        if self.results is None:
            raise ValueError("No results to summarize")

        lines = [
            "=" * 60,
            "PROTEIN STRUCTURE COMPARISON REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of comparisons: {len(self.results)}",
            "",
            "-" * 60,
            "STRUCTURAL SIMILARITY",
            "-" * 60,
            f"TM-score:  {self.results['tm_score'].mean():.3f} ± {self.results['tm_score'].std():.3f}",
            f"           (range: {self.results['tm_score'].min():.3f} - {self.results['tm_score'].max():.3f})",
            f"RMSD:      {self.results['rmsd'].mean():.2f} ± {self.results['rmsd'].std():.2f} Å",
            f"           (range: {self.results['rmsd'].min():.2f} - {self.results['rmsd'].max():.2f} Å)",
        ]

        if "weighted_rmsd" in self.results.columns:
            lines.extend([
                f"Weighted RMSD: {self.results['weighted_rmsd'].mean():.2f} ± {self.results['weighted_rmsd'].std():.2f} Å",
            ])

        lines.extend([
            "",
            "-" * 60,
            "FOLD CLASSIFICATION",
            "-" * 60,
            f"Same fold (TM ≥ 0.5):     {(self.results['tm_score'] >= 0.5).sum()} ({100 * (self.results['tm_score'] >= 0.5).mean():.1f}%)",
            f"Similar fold (TM ≥ 0.4): {(self.results['tm_score'] >= 0.4).sum()} ({100 * (self.results['tm_score'] >= 0.4).mean():.1f}%)",
            f"Different fold (TM < 0.4): {(self.results['tm_score'] < 0.4).sum()} ({100 * (self.results['tm_score'] < 0.4).mean():.1f}%)",
        ])

        if "ss_agreement" in self.results.columns:
            lines.extend([
                "",
                "-" * 60,
                "SECONDARY STRUCTURE",
                "-" * 60,
                f"SS Agreement: {self.results['ss_agreement'].mean():.1%} ± {self.results['ss_agreement'].std():.1%}",
            ])

        if "contact_jaccard" in self.results.columns:
            lines.extend([
                "",
                "-" * 60,
                "CONTACT MAPS",
                "-" * 60,
                f"Contact Jaccard: {self.results['contact_jaccard'].mean():.3f} ± {self.results['contact_jaccard'].std():.3f}",
            ])

        if "mean_plddt_1" in self.results.columns:
            lines.extend([
                "",
                "-" * 60,
                "CONFIDENCE SCORES",
                "-" * 60,
                f"Mean pLDDT (struct 1): {self.results['mean_plddt_1'].mean():.1f}",
                f"Mean pLDDT (struct 2): {self.results['mean_plddt_2'].mean():.1f}",
            ])

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def generate_html_report(
        self,
        output_path: Optional[str | Path] = None,
        include_plots: bool = False,
    ) -> str:
        """Generate HTML report.

        Args:
            output_path: Optional path to save HTML file.
            include_plots: Include embedded plot images.

        Returns:
            HTML string.
        """
        if self.results is None:
            raise ValueError("No results to report")

        # Calculate statistics
        n_same_fold = int((self.results["tm_score"] >= 0.5).sum())
        n_comparisons = len(self.results)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Protein Structure Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-box {{
            background: #3498db;
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-box.warning {{
            background: #e67e22;
        }}
        .metric-box.success {{
            background: #27ae60;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>🧬 Protein Structure Comparison Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary-card">
        <h2>Summary</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value">{n_comparisons}</div>
                <div class="metric-label">Comparisons</div>
            </div>
            <div class="metric-box success">
                <div class="metric-value">{self.results['tm_score'].mean():.3f}</div>
                <div class="metric-label">Mean TM-score</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{self.results['rmsd'].mean():.2f}Å</div>
                <div class="metric-label">Mean RMSD</div>
            </div>
            <div class="metric-box success">
                <div class="metric-value">{n_same_fold}</div>
                <div class="metric-label">Same Fold (TM≥0.5)</div>
            </div>
        </div>
    </div>

    <div class="summary-card">
        <h2>Statistical Summary</h2>
        {self.generate_summary_df().to_html(classes='stats-table', float_format='%.3f')}
    </div>

    <div class="summary-card">
        <h2>All Comparisons</h2>
        {self.results.to_html(classes='comparison-table', index=False, float_format='%.3f')}
    </div>

    <div class="footer">
        <p>Generated by protein_compare v0.1.0</p>
    </div>
</body>
</html>
"""

        if output_path:
            Path(output_path).write_text(html)

        return html

    def get_best_matches(
        self,
        n: int = 10,
        metric: str = "tm_score",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Get top N best (or worst) matches.

        Args:
            n: Number of matches to return.
            metric: Metric to sort by.
            ascending: Sort ascending (True for worst matches).

        Returns:
            DataFrame with top matches.
        """
        if self.results is None:
            raise ValueError("No results available")

        return self.results.nlargest(n, metric) if not ascending else self.results.nsmallest(n, metric)

    def filter_results(
        self,
        min_tm_score: Optional[float] = None,
        max_rmsd: Optional[float] = None,
        min_plddt: Optional[float] = None,
    ) -> pd.DataFrame:
        """Filter results by criteria.

        Args:
            min_tm_score: Minimum TM-score threshold.
            max_rmsd: Maximum RMSD threshold.
            min_plddt: Minimum mean pLDDT threshold.

        Returns:
            Filtered DataFrame.
        """
        if self.results is None:
            raise ValueError("No results available")

        df = self.results.copy()

        if min_tm_score is not None:
            df = df[df["tm_score"] >= min_tm_score]

        if max_rmsd is not None:
            df = df[df["rmsd"] <= max_rmsd]

        if min_plddt is not None:
            df = df[
                (df["mean_plddt_1"] >= min_plddt) &
                (df["mean_plddt_2"] >= min_plddt)
            ]

        return df
