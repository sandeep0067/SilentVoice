"""
WebSocket support for real-time predictions.
"""

import json
import logging
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

from api.services import inference_service


logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manager for WebSocket connections.
    
    Handles multiple concurrent WebSocket connections.
    """
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_count = 0
        
        logger.info("ConnectionManager initialized")
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accept and register a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_count += 1
        
        logger.info(f"WebSocket connected: {client_id} (total: {self.connection_count})")
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "message": "WebSocket connection established"
        })
    
    def disconnect(self, client_id: str) -> None:
        """
        Disconnect a WebSocket client.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            self.connection_count -= 1
            logger.info(f"WebSocket disconnected: {client_id} (total: {self.connection_count})")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str) -> bool:
        """
        Send a message to a specific client.
        
        Args:
            message: Message to send
            client_id: Client identifier
            
        Returns:
            True if sent successfully, False otherwise
        """
        if client_id not in self.active_connections:
            return False
        
        try:
            await self.active_connections[client_id].send_json(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {e}")
            self.disconnect(client_id)
            return False
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        disconnected_clients = []
        
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return self.connection_count
    
    def get_connected_clients(self) -> list:
        """Get list of connected client IDs."""
        return list(self.active_connections.keys())


# Global connection manager instance
connection_manager = ConnectionManager()


async def handle_websocket_prediction(websocket: WebSocket, client_id: str):
    """
    Handle WebSocket prediction requests.
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "predict":
                # Handle prediction request
                features = data.get("features")
                return_probabilities = data.get("return_probabilities", False)
                language = data.get("language")
                
                if not features:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Features required"
                    })
                    continue
                
                # Make prediction
                result = inference_service.predict(
                    features=features,
                    return_probabilities=return_probabilities,
                    language=language
                )
                
                # Send result
                await websocket.send_json({
                    "type": "prediction_result",
                    "request_id": data.get("request_id"),
                    **result
                })
                
            elif message_type == "batch_predict":
                # Handle batch prediction
                sequences = data.get("sequences")
                return_probabilities = data.get("return_probabilities", False)
                language = data.get("language")
                
                if not sequences:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Sequences required"
                    })
                    continue
                
                # Make batch prediction
                result = inference_service.predict_batch(
                    sequences=sequences,
                    return_probabilities=return_probabilities,
                    language=language
                )
                
                # Send result
                await websocket.send_json({
                    "type": "batch_prediction_result",
                    "request_id": data.get("request_id"),
                    **result
                })
                
            elif message_type == "translate":
                # Handle translation request
                class_id = data.get("class_id")
                confidence = data.get("confidence", 1.0)
                language = data.get("language")
                
                if class_id is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Class ID required"
                    })
                    continue
                
                # Translate
                result = inference_service.translate(
                    class_id=class_id,
                    confidence=confidence,
                    language=language
                )
                
                # Send result
                await websocket.send_json({
                    "type": "translation_result",
                    "request_id": data.get("request_id"),
                    **result
                })
                
            elif message_type == "ping":
                # Handle ping/pong
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": data.get("timestamp")
                })
                
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: {client_id}")
        connection_manager.disconnect(client_id)
    
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        connection_manager.disconnect(client_id)
