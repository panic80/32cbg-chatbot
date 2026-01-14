"""Streaming support for chat responses."""

import asyncio
import json
from typing import AsyncIterator, Dict, Any, Optional, List
from datetime import datetime
import uuid

from fastapi import Request
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document

from app.core.logging import get_logger
from app.models.query import ChatRequest, Source

logger = get_logger(__name__)


class StreamingCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self, queue: asyncio.Queue):
        """Initialize with an async queue."""
        self.queue = queue
        self.tokens = []
        
    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Called when LLM starts generating."""
        await self.queue.put({
            "type": "start",
            "data": {"message": "Starting response generation..."}
        })
        
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token."""
        self.tokens.append(token)
        await self.queue.put({
            "type": "token",
            "data": {"token": token}
        })
        
    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes generating."""
        await self.queue.put({
            "type": "end",
            "data": {"message": "Response generation complete"}
        })
        
    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when LLM encounters an error."""
        await self.queue.put({
            "type": "error",
            "data": {"error": str(error)}
        })


class RetrievalStreamingHandler:
    """Handles streaming of retrieval results."""
    
    def __init__(self, queue: asyncio.Queue):
        """Initialize with an async queue."""
        self.queue = queue
        
    async def on_retrieval_start(self, query: str):
        """Called when retrieval starts."""
        await self.queue.put({
            "type": "retrieval_start",
            "data": {"query": query}
        })
        
    async def on_document_retrieved(self, doc: Document, score: float, index: int):
        """Called when a document is retrieved."""
        source = Source(
            id=doc.metadata.get("id", f"source_{index}"),
            text=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
            title=doc.metadata.get("title"),
            url=doc.metadata.get("source"),
            section=doc.metadata.get("section"),
            page=doc.metadata.get("page_number"),
            score=score,
            metadata=doc.metadata
        )
        
        await self.queue.put({
            "type": "source",
            "data": source.model_dump()
        })
        
    async def on_retrieval_end(self, num_docs: int):
        """Called when retrieval completes."""
        await self.queue.put({
            "type": "retrieval_end",
            "data": {"num_documents": num_docs}
        })


async def stream_chat_response(
    request: Request,
    chat_request: ChatRequest,
    llm,
    retrieval_pipeline,
    vector_store,
    document_store
) -> AsyncIterator[str]:
    """Stream chat response with SSE format."""
    queue = asyncio.Queue()
    streaming_handler = StreamingCallbackHandler(queue)
    retrieval_handler = RetrievalStreamingHandler(queue)
    
    async def process_request():
        """Process the chat request and stream results."""
        try:
            # Stream retrieval if RAG is enabled
            if chat_request.use_rag and retrieval_pipeline:
                await retrieval_handler.on_retrieval_start(chat_request.message)
                
                # Retrieve documents
                results = await retrieval_pipeline.retrieve(
                    query=chat_request.message,
                    k=10  # Retrieve more for streaming
                )
                
                # Stream each source as it's processed
                for i, (doc, score) in enumerate(results[:5]):  # Limit sources to 5
                    await retrieval_handler.on_document_retrieved(doc, score, i)
                    await asyncio.sleep(0.01)  # Small delay for smooth streaming
                
                await retrieval_handler.on_retrieval_end(len(results))
                
                # Build context from results without source number labels
                context = "\n".join([
                    f"{doc.page_content}\n"
                    for i, (doc, _) in enumerate(results)
                ])
            else:
                context = ""
            
            # Build messages
            from langchain_core.messages import SystemMessage, HumanMessage
            
            base_system_prompt = """You are a helpful assistant for Canadian Forces members seeking information about travel instructions and policies.
Always provide accurate, specific information based on the official documentation provided.
If you're not certain about something, clearly state that.

IMPORTANT RULES:
1. When multiple sources are present, prioritize the source that provides the most specific and complete information (e.g., actual dollar amounts over references to appendices)
2. NEVER mention source numbers, citations, or reference which source you used
3. Do NOT say things like "according to Source X" or "as stated in the documentation"
4. Give direct, clear answers without referencing the documentation structure
5. If specific values are found, state them directly without qualification
6. Always use proper markdown formatting in your responses:
   - Tables for structured data
   - **Bold** for important values or headers
   - Bullet points or numbered lists for multiple items
   - Clear section headers when appropriate

CRITICAL: When answering questions about rates, allowances, or tables:
- ALWAYS include the actual dollar amounts or specific values found in the documentation
- If you find a table structure (with | characters), preserve and present it as a markdown table
- For meal allowances, include breakfast, lunch, and dinner rates with specific dollar amounts
- For kilometric rates, include the cents per kilometer values
- For incidental allowances, include the daily rates
- If the documentation contains a complete table, reproduce it in your response
- Do not summarize or generalize when specific values are available"""

            system_prompt = base_system_prompt

            if chat_request.additional_instructions:
                system_prompt = f"{base_system_prompt}\n\nADDITIONAL DIRECTIVES:\n{chat_request.additional_instructions.strip()}"

            messages = [SystemMessage(content=system_prompt)]
            
            # Add chat history
            for msg in chat_request.chat_history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(SystemMessage(content=msg.content))
            
            # Add context and query
            if context:
                query_with_context = f"""Based on the following documentation, answer the user's question:

{context}

User Question: {chat_request.message}"""
                messages.append(HumanMessage(content=query_with_context))
            else:
                messages.append(HumanMessage(content=chat_request.message))
            
            # Stream LLM response
            response = await llm.ainvoke(
                messages,
                callbacks=[streaming_handler],
                stream=True
            )
            
            # Send completion metadata
            await queue.put({
                "type": "metadata",
                "data": {
                    "model": chat_request.model or getattr(llm, 'model_name', 'unknown'),
                    "provider": str(chat_request.provider),
                    "conversation_id": chat_request.conversation_id or str(uuid.uuid4())
                }
            })
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            await queue.put({
                "type": "error",
                "data": {"error": str(e)}
            })
        finally:
            # Signal end of stream
            await queue.put(None)
    
    # Start processing in background
    asyncio.create_task(process_request())
    
    # Stream results from queue
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=30.0)
            if item is None:
                break
                
            # Format as SSE
            event_data = json.dumps(item)
            yield f"data: {event_data}\n\n"
            
        except asyncio.TimeoutError:
            # Send keepalive
            yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
        except Exception as e:
            logger.error(f"Queue streaming error: {e}")
            error_data = json.dumps({
                "type": "error",
                "data": {"error": "Streaming error occurred"}
            })
            yield f"data: {error_data}\n\n"
            break
    
    # Send final close event
    yield f"data: {json.dumps({'type': 'close'})}\n\n"


def create_streaming_response(
    request: Request,
    chat_request: ChatRequest,
    llm,
    retrieval_pipeline,
    vector_store,
    document_store
) -> StreamingResponse:
    """Create a streaming response for chat."""
    return StreamingResponse(
        stream_chat_response(
            request,
            chat_request,
            llm,
            retrieval_pipeline,
            vector_store,
            document_store
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
        }
    )
