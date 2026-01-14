"""Example of how to enable and use unified retrieval in the RAG service."""

import asyncio
from typing import Dict, Any

from app.unified_retrieval import (
    load_unified_retriever,
    create_example_unified_config,
    LegacyConfigMapper
)
from app.pipelines.retriever_factory import HybridRetrieverFactory, RetrieverConfig, RetrieverMode
from app.core.logging import get_logger

logger = get_logger(__name__)


async def example_1_simple_unified_retriever():
    """Example 1: Create a simple unified retriever."""
    print("\n=== Example 1: Simple Unified Retriever ===")
    
    # Load a pre-configured unified retriever
    retriever = load_unified_retriever("simple")
    
    # Use the retriever
    query = "What are the meal allowances for travel?"
    docs = await retriever.aget_relevant_documents(query)
    
    print(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs[:3]):
        print(f"\nDocument {i+1}: {doc.page_content[:200]}...")
        
    # Get metrics
    metrics = retriever.get_pipeline_metrics()
    print(f"\nMetrics: {metrics}")


async def example_2_unified_via_factory():
    """Example 2: Create unified retriever through the factory."""
    print("\n=== Example 2: Unified Retriever via Factory ===")
    
    # Assume we have vectorstore, llm, and embeddings initialized
    # For this example, we'll create a mock factory
    
    # Create unified config
    unified_config = create_example_unified_config("balanced")
    
    # Create retriever config with unified mode
    config = RetrieverConfig(
        mode=RetrieverMode.UNIFIED,
        k=10,
        unified_config=unified_config
    )
    
    print(f"Created unified config: {config.mode}")
    print(f"Pipeline has {len(unified_config['pipeline_config']['strategies'])} strategies")


async def example_3_migrate_legacy_config():
    """Example 3: Migrate legacy configuration to unified."""
    print("\n=== Example 3: Migrate Legacy Config ===")
    
    # Create a legacy config
    legacy_config = RetrieverConfig(
        mode=RetrieverMode.ADVANCED,
        use_bm25=True,
        use_multi_query=True,
        use_compression=True,
        use_reranking=True,
        k=15
    )
    
    # Create mapper (normally you'd pass actual components)
    mapper = LegacyConfigMapper()
    
    # Map to unified retriever
    # Note: This would normally create a functional retriever with actual components
    print(f"Legacy config mode: {legacy_config.mode}")
    print(f"Features enabled: BM25={legacy_config.use_bm25}, MultiQuery={legacy_config.use_multi_query}")
    print("Would map to unified retriever with parallel retrieval strategies")


async def example_4_streaming_with_unified():
    """Example 4: Enable unified retrieval in streaming chat."""
    print("\n=== Example 4: Streaming Chat with Unified ===")
    
    # Example request that would enable unified retrieval
    chat_request = {
        "message": "What are the kilometric rates for personal vehicles?",
        "use_rag": True,
        "enable_unified_retrieval": True,  # This enables unified retrieval
        "provider": "openai",
        "model": "gpt-4"
    }
    
    print("Chat request with unified retrieval enabled:")
    print(f"- Message: {chat_request['message']}")
    print(f"- Unified retrieval: {chat_request['enable_unified_retrieval']}")
    print("\nThis would use the unified retrieval pipeline in the streaming endpoint")


def example_5_configuration_options():
    """Example 5: Show available configuration options."""
    print("\n=== Example 5: Configuration Options ===")
    
    # Environment variables for unified retrieval
    env_vars = {
        "RAG_ENABLE_UNIFIED_RETRIEVAL": "true",
        "RAG_UNIFIED_RETRIEVAL_MODE": "balanced",  # simple, balanced, or advanced
        "RAG_UNIFIED_RETRIEVAL_CACHE_TTL": "3600",
        "RAG_UNIFIED_RETRIEVAL_FALLBACK": "true"
    }
    
    print("Environment variables for unified retrieval:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")
        
    # Available modes
    modes = ["simple", "balanced", "advanced", "table_focused", "custom_travel_planning"]
    print("\nAvailable unified retrieval modes:")
    for mode in modes:
        config = create_example_unified_config(mode)
        strategy_count = len(config["pipeline_config"]["strategies"])
        print(f"  - {mode}: {strategy_count} strategies")


async def main():
    """Run all examples."""
    print("=== Unified Retrieval Integration Examples ===")
    print("This shows how to enable and use the unified retrieval system")
    
    # Note: Examples 1-3 would need actual components to run
    # Here we're showing the API and configuration
    
    try:
        # Example 1 would work with actual components
        # await example_1_simple_unified_retriever()
        pass
    except Exception as e:
        print(f"Example 1 requires initialized components: {e}")
    
    await example_2_unified_via_factory()
    await example_3_migrate_legacy_config()
    await example_4_streaming_with_unified()
    example_5_configuration_options()
    
    print("\n=== Summary ===")
    print("The unified retrieval system is now integrated and can be enabled:")
    print("1. Via RetrieverMode.UNIFIED in the factory")
    print("2. By setting enable_unified_retrieval=True in chat requests")
    print("3. Through environment variable RAG_ENABLE_UNIFIED_RETRIEVAL=true")
    print("4. With full backward compatibility for existing code")


if __name__ == "__main__":
    asyncio.run(main())