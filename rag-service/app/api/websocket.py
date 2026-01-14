"""WebSocket endpoints for real-time updates."""

from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from typing import Dict, Set, Any
import json
import asyncio
from datetime import datetime

from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        logger.info(f"Client {client_id} connected via WebSocket")
        
    def disconnect(self, client_id: str):
        """Remove connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            logger.info(f"Client {client_id} disconnected")
            
    async def send_message(self, client_id: str, message: dict):
        """Send message to specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
                self.connection_metadata[client_id]["last_activity"] = datetime.utcnow()
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
                
    async def broadcast(self, message: dict, exclude: Set[str] = None):
        """Broadcast message to all connected clients."""
        exclude = exclude or set()
        disconnected = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id not in exclude:
                try:
                    await websocket.send_json(message)
                    self.connection_metadata[client_id]["last_activity"] = datetime.utcnow()
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected.append(client_id)
                    
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
            
    async def send_progress(
        self, 
        client_id: str, 
        task_id: str,
        progress: float,
        status: str,
        message: str = None,
        details: Dict[str, Any] = None
    ):
        """Send progress update to client."""
        progress_message = {
            "type": "progress",
            "task_id": task_id,
            "progress": progress,  # 0.0 to 1.0
            "status": status,  # "in_progress", "completed", "failed"
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.send_message(client_id, progress_message)
        
    def get_active_connections(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active connections."""
        return {
            client_id: {
                **metadata,
                "duration": (datetime.utcnow() - metadata["connected_at"]).total_seconds()
            }
            for client_id, metadata in self.connection_metadata.items()
        }


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, client_id)
    
    try:
        # Send initial connection confirmation
        await manager.send_message(client_id, {
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 second timeout
                )
                
                # Parse message
                try:
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "ping":
                        # Respond to ping
                        await manager.send_message(client_id, {
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                    elif message.get("type") == "subscribe":
                        # Subscribe to specific events
                        event_type = message.get("event_type")
                        await manager.send_message(client_id, {
                            "type": "subscription",
                            "event_type": event_type,
                            "status": "subscribed",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                    else:
                        # Echo unknown messages
                        await manager.send_message(client_id, {
                            "type": "echo",
                            "original": message,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                except json.JSONDecodeError:
                    await manager.send_message(client_id, {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send keep-alive ping
                try:
                    await manager.send_message(client_id, {
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except Exception:
                    # Connection is dead
                    break
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)


@router.get("/ws/connections")
async def get_active_connections():
    """Get information about active WebSocket connections."""
    return {
        "connections": manager.get_active_connections(),
        "total": len(manager.active_connections)
    }


class IngestionProgressTracker:
    """Tracks progress of ingestion tasks."""
    
    def __init__(self, connection_manager: ConnectionManager):
        """Initialize progress tracker."""
        self.manager = connection_manager
        self.tasks: Dict[str, Dict[str, Any]] = {}
        
    async def start_task(
        self, 
        task_id: str, 
        client_id: str,
        total_items: int,
        task_type: str = "ingestion"
    ):
        """Start tracking a new task."""
        self.tasks[task_id] = {
            "client_id": client_id,
            "task_type": task_type,
            "total_items": total_items,
            "completed_items": 0,
            "status": "in_progress",
            "started_at": datetime.utcnow(),
            "errors": []
        }
        
        await self.manager.send_progress(
            client_id=client_id,
            task_id=task_id,
            progress=0.0,
            status="in_progress",
            message=f"Started {task_type} task with {total_items} items",
            details={"total_items": total_items}
        )
        
    async def update_progress(
        self,
        task_id: str,
        completed_items: int = None,
        increment: int = None,
        message: str = None,
        error: str = None
    ):
        """Update task progress."""
        if task_id not in self.tasks:
            logger.warning(f"Unknown task ID: {task_id}")
            return
            
        task = self.tasks[task_id]
        
        # Update completed items
        if completed_items is not None:
            task["completed_items"] = completed_items
        elif increment:
            task["completed_items"] += increment
            
        # Track errors
        if error:
            task["errors"].append({
                "message": error,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        # Calculate progress
        progress = task["completed_items"] / task["total_items"] if task["total_items"] > 0 else 0
        
        # Send update
        await self.manager.send_progress(
            client_id=task["client_id"],
            task_id=task_id,
            progress=progress,
            status=task["status"],
            message=message,
            details={
                "completed": task["completed_items"],
                "total": task["total_items"],
                "errors": len(task["errors"])
            }
        )
        
    async def complete_task(
        self,
        task_id: str,
        success: bool = True,
        message: str = None
    ):
        """Mark task as completed."""
        if task_id not in self.tasks:
            logger.warning(f"Unknown task ID: {task_id}")
            return
            
        task = self.tasks[task_id]
        task["status"] = "completed" if success else "failed"
        task["completed_at"] = datetime.utcnow()
        
        duration = (task["completed_at"] - task["started_at"]).total_seconds()
        
        await self.manager.send_progress(
            client_id=task["client_id"],
            task_id=task_id,
            progress=1.0 if success else task["completed_items"] / task["total_items"],
            status=task["status"],
            message=message or f"Task {task['status']}",
            details={
                "completed": task["completed_items"],
                "total": task["total_items"],
                "errors": len(task["errors"]),
                "duration": duration
            }
        )
        
        # Clean up completed task after a delay
        await asyncio.sleep(60)  # Keep for 1 minute
        if task_id in self.tasks:
            del self.tasks[task_id]


# Global progress tracker
progress_tracker = IngestionProgressTracker(manager)