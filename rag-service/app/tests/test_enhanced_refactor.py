import pytest
from unittest.mock import MagicMock
from app.pipelines.enhanced_retrieval import EnhancedRetrievalPipeline

def test_enhanced_pipeline_instantiation():
    retriever = MagicMock()
    compressor = MagicMock()
    reranker = MagicMock()
    processor = MagicMock()
    table_rewriter = MagicMock()
    
    pipeline = EnhancedRetrievalPipeline(
        retriever=retriever,
        compressor=compressor,
        reranker=reranker,
        processor=processor,
        table_rewriter=table_rewriter
    )
    
    assert pipeline.workflow is not None
    assert pipeline.nodes is not None
