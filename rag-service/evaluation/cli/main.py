#!/usr/bin/env python3
"""RAG Evaluation Framework CLI."""

import argparse
import asyncio
import sys
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic evaluation dataset
  python -m evaluation.cli.main generate \\
      --rag-url http://localhost:8000 \\
      --output data/eval_dataset.json \\
      --num-questions 100

  # Run full evaluation
  python -m evaluation.cli.main eval \\
      --dataset data/eval_dataset.json \\
      --output results/eval.json

  # Run retrieval-only evaluation
  python -m evaluation.cli.main eval \\
      --dataset data/eval_dataset.json \\
      --retrieval-only \\
      --output results/retrieval.json

  # Compare configurations
  python -m evaluation.cli.main compare \\
      --dataset data/eval_dataset.json \\
      --configs configs/comparison.json \\
      --output results/comparison.json
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation on dataset",
    )
    eval_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to evaluation dataset JSON",
    )
    eval_parser.add_argument(
        "--rag-url",
        default="http://localhost:8000",
        help="RAG service URL (default: http://localhost:8000)",
    )
    eval_parser.add_argument(
        "--admin-token",
        default="",
        help="Admin API token",
    )
    eval_parser.add_argument(
        "--output",
        default="results.json",
        help="Output file path (default: results.json)",
    )
    eval_parser.add_argument(
        "--output-format",
        choices=["json", "csv", "markdown"],
        default="json",
        help="Output format (default: json)",
    )
    eval_parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run retrieval evaluation only",
    )
    eval_parser.add_argument(
        "--skip-hallucination",
        action="store_true",
        help="Skip hallucination detection",
    )
    eval_parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for @k metrics (default: 1 3 5 10)",
    )
    eval_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    # generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic evaluation dataset",
    )
    gen_parser.add_argument(
        "--rag-url",
        default="http://localhost:8000",
        help="RAG service URL",
    )
    gen_parser.add_argument(
        "--admin-token",
        default="",
        help="Admin API token",
    )
    gen_parser.add_argument(
        "--output",
        required=True,
        help="Output dataset file path",
    )
    gen_parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Number of questions per type (default: 50)",
    )
    gen_parser.add_argument(
        "--question-types",
        nargs="+",
        default=["factual", "procedural", "comparison"],
        help="Question types to generate",
    )
    gen_parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="LLM model for generation (default: gpt-4.1-mini)",
    )
    gen_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    # compare command
    cmp_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple configurations",
    )
    cmp_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to evaluation dataset",
    )
    cmp_parser.add_argument(
        "--configs",
        required=True,
        help="JSON file with configurations to compare",
    )
    cmp_parser.add_argument(
        "--rag-url",
        default="http://localhost:8000",
        help="RAG service URL",
    )
    cmp_parser.add_argument(
        "--admin-token",
        default="",
        help="Admin API token",
    )
    cmp_parser.add_argument(
        "--output",
        default="comparison.json",
        help="Output file path",
    )
    cmp_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser


async def run_evaluation(args) -> int:
    """Run evaluation command."""
    from evaluation.client.rag_client import RAGClient
    from evaluation.core.config import EvaluationConfig
    from evaluation.core.runner import EvaluationRunner
    from evaluation.datagen.dataset import EvaluationDataset
    from evaluation.output.formatters import format_results, print_summary

    print(f"Loading dataset from {args.dataset}...")
    dataset = EvaluationDataset.load(args.dataset)
    print(f"Loaded {len(dataset)} questions")

    config = EvaluationConfig(
        rag_url=args.rag_url,
        admin_token=args.admin_token,
        k_values=args.k_values,
        output_format=args.output_format,
        output_path=args.output,
        verbose=args.verbose,
    )

    errors = config.validate()
    if errors:
        print(f"Configuration errors: {errors}")
        return 1

    client = RAGClient(
        base_url=config.rag_url,
        admin_token=config.admin_token,
    )

    runner = EvaluationRunner(
        config=config,
        rag_client=client,
        dataset=dataset,
    )

    def progress(current, total, message=""):
        if args.verbose:
            print(f"\r{message} [{current}/{total}]", end="", flush=True)

    try:
        async with runner:
            if args.retrieval_only:
                print("\nRunning retrieval-only evaluation...")
                result = await runner.run_retrieval_only(
                    progress_callback=lambda c, t: progress(c, t, "Retrieval")
                )
            else:
                print("\nRunning full evaluation...")
                result = await runner.run_full_evaluation(progress_callback=progress)

        print("\n")

        # Save results
        format_results(result, args.output_format, args.output)
        print(f"Results saved to: {args.output}")

        # Print summary
        print_summary(result)

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


async def run_generate(args) -> int:
    """Run generate command."""
    from evaluation.client.rag_client import RAGClient
    from evaluation.datagen.question_generator import create_generator_from_openai

    print(f"Connecting to RAG service at {args.rag_url}...")

    client = RAGClient(
        base_url=args.rag_url,
        admin_token=args.admin_token,
    )

    generator = create_generator_from_openai(
        model=args.model,
        question_types=args.question_types,
    )

    def progress(current, total, message=""):
        print(f"\r{message}", end="", flush=True)

    try:
        async with client:
            print("Generating evaluation dataset...")
            dataset = await generator.generate_from_rag_client(
                rag_client=client,
                questions_per_type=args.num_questions,
                progress_callback=progress if args.verbose else None,
            )

        print(f"\n\nGenerated {len(dataset)} questions")
        print(f"Type distribution: {dataset.get_type_counts()}")

        # Save dataset
        dataset.save(args.output)
        print(f"Dataset saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


async def run_compare(args) -> int:
    """Run compare command."""
    import json

    from evaluation.client.rag_client import RAGClient
    from evaluation.core.config import EvaluationConfig
    from evaluation.core.runner import EvaluationRunner
    from evaluation.datagen.dataset import EvaluationDataset
    from evaluation.output.formatters import JSONFormatter

    print(f"Loading dataset from {args.dataset}...")
    dataset = EvaluationDataset.load(args.dataset)

    print(f"Loading configurations from {args.configs}...")
    with open(args.configs) as f:
        configs = json.load(f)

    print(f"Comparing {len(configs)} configurations...")

    config = EvaluationConfig(
        rag_url=args.rag_url,
        admin_token=args.admin_token,
    )

    client = RAGClient(
        base_url=config.rag_url,
        admin_token=config.admin_token,
    )

    runner = EvaluationRunner(
        config=config,
        rag_client=client,
        dataset=dataset,
    )

    def progress(config_name, current, total):
        if args.verbose:
            print(f"\r{config_name}: [{current}/{total}]", end="", flush=True)

    try:
        async with runner:
            results = await runner.compare_configurations(
                configs=configs,
                progress_callback=progress,
            )

        print("\n")

        # Print comparison summary
        print("=" * 70)
        print("           CONFIGURATION COMPARISON RESULTS")
        print("=" * 70)
        print(f"\n{'Config':<20} {'MRR':<10} {'P@5':<10} {'R@5':<10} {'Latency':<10}")
        print("-" * 70)

        for name, result in results.items():
            agg = result.aggregate
            mrr = f"{agg.mean_mrr:.4f}"
            p5 = f"{agg.mean_precision_at_k.get(5, 0):.4f}"
            r5 = f"{agg.mean_recall_at_k.get(5, 0):.4f}"
            lat = f"{agg.mean_retrieval_latency_ms:.0f}ms"
            print(f"{name:<20} {mrr:<10} {p5:<10} {r5:<10} {lat:<10}")

        print("=" * 70)

        # Save results
        comparison_data = {
            name: result.to_dict() for name, result in results.items()
        }

        with open(args.output, "w") as f:
            json.dump(comparison_data, f, indent=2, default=str)

        print(f"\nResults saved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "eval":
        return asyncio.run(run_evaluation(args))
    elif args.command == "generate":
        return asyncio.run(run_generate(args))
    elif args.command == "compare":
        return asyncio.run(run_compare(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
