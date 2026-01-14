"""Output formatters for evaluation results."""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.core.results import EvaluationResult


class JSONFormatter:
    """Format results as JSON."""

    def __init__(self, indent: int = 2):
        self.indent = indent

    def format(self, result: EvaluationResult) -> str:
        """Format EvaluationResult as JSON string."""
        return json.dumps(result.to_dict(), indent=self.indent, default=str)

    def save(self, result: EvaluationResult, path: str) -> None:
        """Save EvaluationResult to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.format(result))


class CSVFormatter:
    """Format results as CSV."""

    def format_retrieval(self, result: EvaluationResult) -> str:
        """Format retrieval metrics as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        k_values = list(result.aggregate.mean_precision_at_k.keys()) or [1, 3, 5, 10]
        header = ["query", "mrr", "latency_ms"]
        for k in k_values:
            header.extend([f"precision@{k}", f"recall@{k}", f"hit_rate@{k}"])
        header.append("error")
        writer.writerow(header)

        # Data rows
        for m in result.retrieval_metrics:
            row = [m.query, m.mrr, m.latency_ms]
            for k in k_values:
                row.extend([
                    m.precision_at_k.get(k, ""),
                    m.recall_at_k.get(k, ""),
                    m.hit_rate_at_k.get(k, ""),
                ])
            row.append(m.error or "")
            writer.writerow(row)

        return output.getvalue()

    def format_generation(self, result: EvaluationResult) -> str:
        """Format generation metrics as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "query",
            "relevance_score",
            "completeness_score",
            "grounding_score",
            "latency_ms",
            "error",
        ])

        # Data rows
        for m in result.generation_metrics:
            writer.writerow([
                m.query,
                m.relevance_score,
                m.completeness_score,
                m.grounding_score,
                m.latency_ms,
                m.error or "",
            ])

        return output.getvalue()

    def format_hallucination(self, result: EvaluationResult) -> str:
        """Format hallucination results as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "query",
            "hallucination_score",
            "total_claims",
            "entailed",
            "neutral",
            "contradicted",
            "flagged_claims",
            "error",
        ])

        # Data rows
        for r in result.hallucination_results:
            flagged = "; ".join(c.text for c in r.flagged_claims)
            writer.writerow([
                r.query,
                r.hallucination_score,
                r.total_claims,
                r.entailed_count,
                r.neutral_count,
                r.contradicted_count,
                flagged,
                r.error or "",
            ])

        return output.getvalue()

    def save(self, result: EvaluationResult, base_path: str) -> List[str]:
        """Save EvaluationResult to CSV files.

        Args:
            result: The evaluation result
            base_path: Base path without extension

        Returns:
            List of created file paths
        """
        created_files = []
        Path(base_path).parent.mkdir(parents=True, exist_ok=True)

        if result.retrieval_metrics:
            path = f"{base_path}_retrieval.csv"
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.format_retrieval(result))
            created_files.append(path)

        if result.generation_metrics:
            path = f"{base_path}_generation.csv"
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.format_generation(result))
            created_files.append(path)

        if result.hallucination_results:
            path = f"{base_path}_hallucination.csv"
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.format_hallucination(result))
            created_files.append(path)

        return created_files


class MarkdownFormatter:
    """Format results as Markdown report."""

    def format(self, result: EvaluationResult) -> str:
        """Format EvaluationResult as Markdown report."""
        lines = []

        # Header
        lines.append("# RAG Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {result.timestamp.isoformat()}")
        lines.append(f"**Configuration:** {result.config_name}")
        lines.append(f"**Total Queries:** {result.aggregate.total_queries}")
        lines.append("")

        # Summary section
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        agg = result.aggregate

        # Retrieval summary
        if agg.mean_mrr > 0:
            lines.append(f"| Mean MRR | {agg.mean_mrr:.4f} |")
            for k, v in sorted(agg.mean_precision_at_k.items()):
                lines.append(f"| Mean Precision@{k} | {v:.4f} |")
            for k, v in sorted(agg.mean_recall_at_k.items()):
                lines.append(f"| Mean Recall@{k} | {v:.4f} |")
            lines.append(
                f"| Mean Retrieval Latency | {agg.mean_retrieval_latency_ms:.1f}ms |"
            )

        # Generation summary
        if agg.mean_relevance_score > 0:
            lines.append(f"| Mean Relevance | {agg.mean_relevance_score:.4f} |")
            lines.append(f"| Mean Completeness | {agg.mean_completeness_score:.4f} |")
            lines.append(f"| Mean Grounding | {agg.mean_grounding_score:.4f} |")

        # Hallucination summary
        if agg.total_claims > 0:
            lines.append(
                f"| Mean Hallucination Score | {agg.mean_hallucination_score:.4f} |"
            )
            lines.append(f"| Total Claims | {agg.total_claims} |")
            lines.append(f"| Entailed Claims | {agg.total_entailed} |")
            lines.append(f"| Contradicted Claims | {agg.total_contradicted} |")
            lines.append(
                f"| Queries with Hallucination | {agg.queries_with_hallucination} |"
            )

        # Error summary
        total_errors = (
            agg.retrieval_errors + agg.generation_errors + agg.hallucination_errors
        )
        if total_errors > 0:
            lines.append(f"| Total Errors | {total_errors} |")

        lines.append("")

        # Detailed retrieval results
        if result.retrieval_metrics:
            lines.append("## Retrieval Results")
            lines.append("")
            lines.append(self._format_retrieval_table(result))
            lines.append("")

        # Detailed generation results
        if result.generation_metrics:
            lines.append("## Generation Quality Results")
            lines.append("")
            lines.append(self._format_generation_table(result))
            lines.append("")

        # Detailed hallucination results
        if result.hallucination_results:
            lines.append("## Hallucination Detection Results")
            lines.append("")
            lines.append(self._format_hallucination_table(result))
            lines.append("")

            # Flagged claims section
            flagged = [
                (r.query, c)
                for r in result.hallucination_results
                for c in r.flagged_claims
            ]
            if flagged:
                lines.append("### Flagged Claims")
                lines.append("")
                for query, claim in flagged[:10]:  # Limit to 10
                    lines.append(f"- **Query:** {query[:50]}...")
                    lines.append(f"  - Claim: {claim.text}")
                    lines.append(f"  - Confidence: {claim.nli_confidence:.2f}")
                    lines.append("")

        return "\n".join(lines)

    def _format_retrieval_table(self, result: EvaluationResult) -> str:
        """Format retrieval metrics as Markdown table."""
        lines = []
        lines.append("| Query | MRR | P@5 | R@5 | Latency |")
        lines.append("|-------|-----|-----|-----|---------|")

        for m in result.retrieval_metrics[:20]:  # Limit rows
            query = m.query[:40] + "..." if len(m.query) > 40 else m.query
            p5 = m.precision_at_k.get(5, 0)
            r5 = m.recall_at_k.get(5, 0)
            lines.append(f"| {query} | {m.mrr:.3f} | {p5:.3f} | {r5:.3f} | {m.latency_ms:.0f}ms |")

        if len(result.retrieval_metrics) > 20:
            lines.append(f"| ... ({len(result.retrieval_metrics) - 20} more) | | | | |")

        return "\n".join(lines)

    def _format_generation_table(self, result: EvaluationResult) -> str:
        """Format generation metrics as Markdown table."""
        lines = []
        lines.append("| Query | Relevance | Completeness | Grounding |")
        lines.append("|-------|-----------|--------------|-----------|")

        for m in result.generation_metrics[:20]:
            query = m.query[:40] + "..." if len(m.query) > 40 else m.query
            lines.append(
                f"| {query} | {m.relevance_score:.3f} | {m.completeness_score:.3f} | {m.grounding_score:.3f} |"
            )

        if len(result.generation_metrics) > 20:
            lines.append(f"| ... ({len(result.generation_metrics) - 20} more) | | | |")

        return "\n".join(lines)

    def _format_hallucination_table(self, result: EvaluationResult) -> str:
        """Format hallucination results as Markdown table."""
        lines = []
        lines.append("| Query | Score | Claims | Entailed | Contradicted |")
        lines.append("|-------|-------|--------|----------|--------------|")

        for r in result.hallucination_results[:20]:
            query = r.query[:40] + "..." if len(r.query) > 40 else r.query
            lines.append(
                f"| {query} | {r.hallucination_score:.3f} | {r.total_claims} | {r.entailed_count} | {r.contradicted_count} |"
            )

        if len(result.hallucination_results) > 20:
            lines.append(
                f"| ... ({len(result.hallucination_results) - 20} more) | | | | |"
            )

        return "\n".join(lines)

    def save(self, result: EvaluationResult, path: str) -> None:
        """Save EvaluationResult to Markdown file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.format(result))


def format_results(
    result: EvaluationResult,
    output_format: str = "json",
    output_path: Optional[str] = None,
) -> str:
    """Format and optionally save evaluation results.

    Args:
        result: The evaluation result
        output_format: One of 'json', 'csv', 'markdown'
        output_path: Optional path to save results

    Returns:
        Formatted string representation
    """
    if output_format == "json":
        formatter = JSONFormatter()
        formatted = formatter.format(result)
        if output_path:
            formatter.save(result, output_path)
    elif output_format == "csv":
        formatter = CSVFormatter()
        formatted = formatter.format_retrieval(result)  # Primary CSV
        if output_path:
            formatter.save(result, output_path.replace(".csv", ""))
    elif output_format == "markdown":
        formatter = MarkdownFormatter()
        formatted = formatter.format(result)
        if output_path:
            formatter.save(result, output_path)
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    return formatted


def print_summary(result: EvaluationResult) -> None:
    """Print a brief summary of evaluation results to console."""
    print("\n" + "=" * 60)
    print("           RAG EVALUATION SUMMARY")
    print("=" * 60)

    agg = result.aggregate

    print(f"\nTotal Queries: {agg.total_queries}")

    if agg.mean_mrr > 0:
        print("\n📊 Retrieval Metrics:")
        print(f"   MRR: {agg.mean_mrr:.4f}")
        for k in sorted(agg.mean_precision_at_k.keys()):
            print(f"   Precision@{k}: {agg.mean_precision_at_k[k]:.4f}")
        print(f"   Avg Latency: {agg.mean_retrieval_latency_ms:.1f}ms")

    if agg.mean_relevance_score > 0:
        print("\n📝 Generation Quality:")
        print(f"   Relevance: {agg.mean_relevance_score:.4f}")
        print(f"   Completeness: {agg.mean_completeness_score:.4f}")
        print(f"   Grounding: {agg.mean_grounding_score:.4f}")

    if agg.total_claims > 0:
        print("\n🔍 Hallucination Detection:")
        print(f"   Mean Score: {agg.mean_hallucination_score:.4f}")
        print(f"   Total Claims: {agg.total_claims}")
        support_rate = agg.total_entailed / agg.total_claims if agg.total_claims else 0
        print(f"   Claim Support Rate: {support_rate:.2%}")
        print(f"   Queries w/ Hallucination: {agg.queries_with_hallucination}")

    total_errors = agg.retrieval_errors + agg.generation_errors + agg.hallucination_errors
    if total_errors > 0:
        print(f"\n⚠️  Total Errors: {total_errors}")

    print("\n" + "=" * 60)
