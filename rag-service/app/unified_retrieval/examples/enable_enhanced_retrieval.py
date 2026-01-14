"""Example: Enable Enhanced Retrieval Features

This example shows how to integrate the enhanced retrieval features
into the existing unified retrieval system.
"""

import os
from typing import Optional
import redis
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.core.vectorstore import get_vectorstore
from app.unified_retrieval.unified_retriever import UnifiedRetriever
from app.unified_retrieval.config.loader import load_pipeline_config
from app.unified_retrieval.strategies import (
    SemanticCacheStrategy,
    EnhancedQueryStrategy,
    ContextualBoostStrategy,
    AdaptiveHybridStrategy,
    HybridRetrievalStrategy,
    AuthorityBoostStrategy
)


def create_enhanced_retriever(
    enable_cache: bool = True,
    enable_expansion: bool = True,
    enable_contextual: bool = True,
    enable_adaptive: bool = True,
    redis_url: Optional[str] = None
) -> UnifiedRetriever:
    """Create retriever with enhanced features
    
    Args:
        enable_cache: Enable semantic caching
        enable_expansion: Enable query expansion
        enable_contextual: Enable contextual scoring
        enable_adaptive: Enable adaptive hybrid fusion
        redis_url: Redis URL for caching
        
    Returns:
        Configured UnifiedRetriever
    """
    # Initialize components
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0)
    vectorstore = get_vectorstore()
    
    # Initialize Redis if caching is enabled
    redis_client = None
    if enable_cache and redis_url:
        redis_client = redis.from_url(redis_url, decode_responses=True)
    
    # Create retriever
    retriever = UnifiedRetriever(
        vectorstore=vectorstore,
        embeddings=embeddings,
        llm=llm
    )
    
    # Build strategy pipeline
    strategies = []
    
    # 1. Semantic Cache (check first)
    if enable_cache and redis_client:
        strategies.append(
            SemanticCacheStrategy(
                embeddings=embeddings,
                redis_client=redis_client,
                similarity_threshold=0.95,
                ttl=3600
            )
        )
    
    # 2. Query Expansion
    if enable_expansion:
        strategies.append(
            EnhancedQueryStrategy(
                spacy_model="en_core_web_sm"
            )
        )
    
    # 3. Hybrid Retrieval
    strategies.append(
        HybridRetrievalStrategy(
            vectorstore=vectorstore,
            embeddings=embeddings,
            vector_weight=0.5,
            bm25_weight=0.5,
            k=30
        )
    )
    
    # 4. Adaptive Fusion
    if enable_adaptive:
        strategies.append(
            AdaptiveHybridStrategy()
        )
    
    # 5. Contextual Boost
    if enable_contextual:
        strategies.append(
            ContextualBoostStrategy(
                embeddings=embeddings,
                memory_size=10,
                entity_boost=0.2,
                topic_boost=0.15
            )
        )
    
    # 6. Authority Boost
    strategies.append(
        AuthorityBoostStrategy(
            authority_keywords=["official", "policy", "directive"],
            boost_factor=0.2
        )
    )
    
    # Configure retriever
    retriever.configure_pipeline(strategies)
    
    return retriever


def create_enhanced_retriever_from_config(
    config_name: str = "comprehensive_enhanced"
) -> UnifiedRetriever:
    """Create retriever from predefined configuration
    
    Args:
        config_name: Name of pipeline configuration
        
    Returns:
        Configured UnifiedRetriever
    """
    # Load configuration
    config_path = "app/unified_retrieval/config/enhanced_pipelines.yaml"
    pipeline_config = load_pipeline_config(config_path, config_name)
    
    # Initialize components
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0)
    vectorstore = get_vectorstore()
    
    # Create retriever with configuration
    retriever = UnifiedRetriever(
        vectorstore=vectorstore,
        embeddings=embeddings,
        llm=llm,
        pipeline_config=pipeline_config
    )
    
    return retriever


async def example_usage():
    """Example usage of enhanced retriever"""
    # Create enhanced retriever
    retriever = create_enhanced_retriever(
        enable_cache=True,
        enable_expansion=True,
        enable_contextual=True,
        enable_adaptive=True,
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    
    # Example queries
    queries = [
        "How do I submit a TD claim?",
        "What are the per diem rates for international travel?",
        "Can CAF members get travel advances?"
    ]
    
    # Simulate conversation
    conversation_history = []
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Retrieve with enhancements
        results = await retriever.aretrieve(
            query,
            k=5,
            conversation_history=conversation_history
        )
        
        # Display results
        print(f"Retrieved {len(results)} documents:")
        for i, doc in enumerate(results[:3]):
            print(f"{i+1}. {doc.page_content[:100]}...")
            if "contextual_score" in doc.metadata:
                print(f"   Contextual score: {doc.metadata['contextual_score']:.3f}")
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({
            "role": "assistant", 
            "content": f"Found {len(results)} relevant documents."
        })


async def benchmark_enhancements():
    """Benchmark enhanced vs standard retrieval"""
    import time
    
    # Create retrievers
    standard_retriever = UnifiedRetriever(
        vectorstore=get_vectorstore(),
        embeddings=OpenAIEmbeddings()
    )
    
    enhanced_retriever = create_enhanced_retriever(
        enable_cache=True,
        enable_expansion=True,
        enable_contextual=True,
        enable_adaptive=True
    )
    
    # Test queries
    test_queries = [
        "TD procedures",
        "temporary duty procedures for military members",
        "How do Canadian Armed Forces members claim travel expenses?",
        "What is the per diem allowance rate?"
    ]
    
    print("Benchmarking retrieval performance...\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        
        # Standard retrieval
        start = time.time()
        standard_results = await standard_retriever.aretrieve(query, k=5)
        standard_time = time.time() - start
        
        # Enhanced retrieval
        start = time.time()
        enhanced_results = await enhanced_retriever.aretrieve(query, k=5)
        enhanced_time = time.time() - start
        
        print(f"  Standard: {len(standard_results)} docs in {standard_time:.3f}s")
        print(f"  Enhanced: {len(enhanced_results)} docs in {enhanced_time:.3f}s")
        
        # Compare top results
        if enhanced_results:
            print(f"  Top enhanced result: {enhanced_results[0].page_content[:80]}...")
        print()


if __name__ == "__main__":
    import asyncio
    
    # Run examples
    asyncio.run(example_usage())
    
    # Run benchmark
    asyncio.run(benchmark_enhancements())