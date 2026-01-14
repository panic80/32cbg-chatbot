from langgraph.graph import StateGraph, END
from app.pipelines.definitions import RetrievalState, QueryType
from app.pipelines.nodes.retrieval_nodes import RetrievalNodes

class EnhancedGraphBuilder:
    """Builder for the enhanced retrieval LangGraph."""

    def __init__(self, nodes: RetrievalNodes):
        self.nodes = nodes

    def build(self) -> StateGraph:
        """Build and compile the workflow graph."""
        workflow = StateGraph(RetrievalState)
        
        # Add nodes
        workflow.add_node("understand_query", self.nodes.understand_query)
        workflow.add_node("expand_query", self.nodes.expand_query)
        workflow.add_node("retrieve_documents", self.nodes.retrieve_documents)
        workflow.add_node("compress_documents", self.nodes.compress_documents)
        workflow.add_node("rerank_documents", self.nodes.rerank_documents)
        workflow.add_node("synthesize_answer", self.nodes.synthesize_answer)
        workflow.add_node("fallback_retrieval", self.nodes.fallback_retrieval)
        workflow.add_node("handle_error", self.nodes.handle_error)
        
        # Add edges with conditional routing
        workflow.add_conditional_edges(
            "understand_query",
            self._route_by_query_type,
            {
                "expand_query": "expand_query",
                "retrieve_documents": "retrieve_documents"
            }
        )
        workflow.add_edge("expand_query", "retrieve_documents")
        workflow.add_conditional_edges(
            "retrieve_documents",
            self._check_retrieval_quality,
            {
                "handle_error": "handle_error",
                "fallback_retrieval": "fallback_retrieval",
                "compress_documents": "compress_documents"
            }
        )
        workflow.add_edge("compress_documents", "rerank_documents")
        workflow.add_edge("rerank_documents", "synthesize_answer")
        workflow.add_edge("fallback_retrieval", "compress_documents")
        workflow.add_edge("synthesize_answer", END)
        workflow.add_edge("handle_error", END)
        
        # Set entry point
        workflow.set_entry_point("understand_query")
        
        return workflow.compile()

    def _route_by_query_type(self, state: RetrievalState) -> str:
        """Route based on query type."""
        # state["query_type"] is already a string like "multi_hop" or "complex"
        if state["query_type"] in [QueryType.MULTI_HOP.value, QueryType.COMPLEX.value]:
            return "expand_query"
        return "retrieve_documents"
    
    def _check_retrieval_quality(self, state: RetrievalState) -> str:
        """Check retrieval quality and route accordingly."""
        if state.get("error"):
            return "handle_error"
        
        if not state["retrieved_documents"]:
            return "fallback_retrieval"
        
        # Check quality threshold
        min_docs = {
            QueryType.SIMPLE.value: 1,
            QueryType.TABLE.value: 2,
            QueryType.COMPLEX.value: 3,
            QueryType.MULTI_HOP.value: 5,
            QueryType.COMPARISON.value: 4
        }
        
        required = min_docs.get(state["query_type"], 2)
        if len(state["retrieved_documents"]) < required:
            return "fallback_retrieval"
        
        return "compress_documents"
