"""
Server-Sent Events Manager
Manages SSE connections and broadcasts alerts to connected clients.
"""

import asyncio
import json
from typing import Set, Dict, Any, AsyncGenerator
from datetime import datetime
from fastapi.responses import StreamingResponse

from models.alerts import VideoAlert
from utils.logging import get_logger

logger = get_logger(__name__)


class SSEManager:
    """Manages Server-Sent Events connections and alert broadcasting"""
    
    def __init__(self):
        """Initialize SSE manager"""
        self.connections: Set[asyncio.Queue] = set()
        self.alert_queue = asyncio.Queue()
        self.broadcast_task = None
        
        logger.info("SSE Manager initialized")
    
    async def start_broadcast_service(self):
        """Start the alert broadcasting service"""
        if self.broadcast_task is None or self.broadcast_task.done():
            self.broadcast_task = asyncio.create_task(self._broadcast_alerts())
            logger.info("SSE broadcast service started")
    
    async def stop_broadcast_service(self):
        """Stop the alert broadcasting service"""
        if self.broadcast_task and not self.broadcast_task.done():
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
            logger.info("SSE broadcast service stopped")
    
    async def add_connection(self) -> asyncio.Queue:
        """
        Add a new SSE connection
        
        Returns:
            Queue for the connection
        """
        connection_queue = asyncio.Queue()
        self.connections.add(connection_queue)
        logger.info(f"Added SSE connection. Total connections: {len(self.connections)}")
        return connection_queue
    
    async def remove_connection(self, connection_queue: asyncio.Queue):
        """
        Remove an SSE connection
        
        Args:
            connection_queue: Queue of the connection to remove
        """
        self.connections.discard(connection_queue)
        logger.info(f"Removed SSE connection. Total connections: {len(self.connections)}")
    
    async def broadcast_alert(self, alert: VideoAlert):
        """
        Broadcast alert to all connected clients and save to database
        
        Args:
            alert: VideoAlert to broadcast
        """
        try:
            # Save alert to database first
            await self._save_alert_to_database(alert)
            
            # Put alert in broadcast queue
            await self.alert_queue.put(alert)
            logger.info(f"Queued alert for broadcast: {alert.id} (connections: {len(self.connections)})")
        except Exception as e:
            logger.error(f"Error queuing alert for broadcast: {str(e)}")
    
    async def _save_alert_to_database(self, alert: VideoAlert):
        """
        Save alert to database
        
        Args:
            alert: VideoAlert to save
        """
        try:
            from models.database import get_db, db_manager
            
            # Get database session
            db_gen = get_db()
            db = next(db_gen)
            
            try:
                # Prepare alert data for database
                alert_data = {
                    "id": alert.id,
                    "pipeline": "video_surveillance",
                    "type": "video",
                    "timestamp": alert.timestamp,
                    "confidence": alert.confidence,
                    "status": alert.status.value,
                    "data": alert.model_dump(mode='json')
                }
                
                # Create alert in database
                db_alert = db_manager.create_alert(db, alert_data)
                logger.info(f"Saved alert to database: {alert.id}")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error saving alert to database: {str(e)}")
    
    async def _broadcast_alerts(self):
        """Background task to broadcast alerts to all connections"""
        try:
            while True:
                # Wait for alert
                alert = await self.alert_queue.get()
                
                if not self.connections:
                    logger.debug("No SSE connections, skipping alert broadcast")
                    continue
                
                # Prepare alert data for SSE
                alert_data = {
                    "type": "alert",
                    "data": alert.model_dump(mode='json'),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Broadcast to all connections
                disconnected_connections = set()
                
                for connection_queue in self.connections:
                    try:
                        # Non-blocking put with timeout
                        await asyncio.wait_for(
                            connection_queue.put(alert_data),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("SSE connection queue full, marking for removal")
                        disconnected_connections.add(connection_queue)
                    except Exception as e:
                        logger.warning(f"Error sending to SSE connection: {str(e)}")
                        disconnected_connections.add(connection_queue)
                
                # Remove disconnected connections
                for conn in disconnected_connections:
                    self.connections.discard(conn)
                
                logger.info(f"Broadcasted alert {alert.id} to {len(self.connections)} connections")
                
        except asyncio.CancelledError:
            logger.info("Alert broadcast task cancelled")
        except Exception as e:
            logger.error(f"Error in alert broadcast task: {str(e)}")
    
    async def create_sse_response(self) -> StreamingResponse:
        """
        Create SSE response for a client connection
        
        Returns:
            StreamingResponse for SSE
        """
        connection_queue = await self.add_connection()
        
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                # Send initial connection confirmation
                initial_data = {
                    "type": "connection",
                    "data": {"status": "connected", "timestamp": datetime.now().isoformat()},
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(initial_data)}\n\n"
                
                # Send events from queue
                while True:
                    try:
                        # Wait for event with timeout for heartbeat
                        event_data = await asyncio.wait_for(
                            connection_queue.get(),
                            timeout=30.0  # 30 second heartbeat
                        )
                        
                        # Send the event
                        yield f"data: {json.dumps(event_data)}\n\n"
                        
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        heartbeat_data = {
                            "type": "heartbeat",
                            "data": {"timestamp": datetime.now().isoformat()},
                            "timestamp": datetime.now().isoformat()
                        }
                        yield f"data: {json.dumps(heartbeat_data)}\n\n"
                        
            except asyncio.CancelledError:
                logger.info("SSE connection cancelled by client")
            except Exception as e:
                logger.error(f"SSE connection error: {str(e)}")
                # Send error message
                error_data = {
                    "type": "error",
                    "data": {"message": "Connection error", "timestamp": datetime.now().isoformat()},
                    "timestamp": datetime.now().isoformat()
                }
                try:
                    yield f"data: {json.dumps(error_data)}\n\n"
                except:
                    pass
            finally:
                # Clean up connection
                await self.remove_connection(connection_queue)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )


# Global SSE manager instance
sse_manager = SSEManager()