"""
Kafka Manager for event streaming and message queuing.

This module provides async Kafka producer and consumer functionality
for real-time event streaming and inter-service communication.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError

from ..core.logging import LoggerMixin
from ..core.exceptions import ServiceError


@dataclass
class KafkaMessage:
    """Kafka message structure."""
    topic: str
    key: Optional[str]
    value: Dict[str, Any]
    headers: Dict[str, str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.headers is None:
            self.headers = {}


class KafkaManager(LoggerMixin):
    """Async Kafka manager for event streaming."""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        client_id: str = "omni-agent-hub",
        group_id: str = "omni-agents"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.group_id = group_id
        
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
        self._initialized = False
        self._consuming = False
        
        self.logger.info("Kafka manager initialized", servers=bootstrap_servers)
    
    async def initialize(self) -> None:
        """Initialize Kafka producer and consumer."""
        try:
            self.logger.info("Initializing Kafka connections")
            
            # Initialize producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=f"{self.client_id}-producer",
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retry_backoff_ms=1000,
                request_timeout_ms=30000
            )
            await self.producer.start()
            
            # Test connection
            await self._test_connection()
            
            self._initialized = True
            self.logger.info("Kafka initialized successfully")
            
        except Exception as e:
            self.log_error(e, {"operation": "kafka_initialization"})
            raise ServiceError(f"Failed to initialize Kafka: {str(e)}")
    
    async def close(self) -> None:
        """Close Kafka connections."""
        try:
            self._consuming = False
            
            # Close consumers
            for consumer in self.consumers.values():
                await consumer.stop()
            
            if self.consumer:
                await self.consumer.stop()
            
            # Close producer
            if self.producer:
                await self.producer.stop()
            
            self.logger.info("Kafka connections closed")
            
        except Exception as e:
            self.log_error(e, {"operation": "kafka_close"})
    
    async def _test_connection(self) -> None:
        """Test Kafka connection."""
        try:
            # Send a test message
            await self.producer.send_and_wait(
                "test-topic",
                value={"test": True, "timestamp": datetime.utcnow().isoformat()}
            )
            self.logger.info("Kafka connection test successful")
            
        except Exception as e:
            raise ServiceError(f"Kafka connection test failed: {str(e)}")
    
    def _ensure_initialized(self) -> None:
        """Ensure Kafka is initialized."""
        if not self._initialized or not self.producer:
            raise ServiceError("Kafka not initialized")
    
    # Producer Methods
    async def send_message(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
        headers: Dict[str, str] = None
    ) -> bool:
        """Send a message to Kafka topic."""
        self._ensure_initialized()
        
        try:
            kafka_message = KafkaMessage(
                topic=topic,
                key=key,
                value=message,
                headers=headers or {}
            )
            
            # Add standard headers
            kafka_message.headers.update({
                "source": "omni-agent-hub",
                "timestamp": kafka_message.timestamp.isoformat(),
                "message_id": f"{topic}_{datetime.utcnow().timestamp()}"
            })
            
            # Send message
            await self.producer.send_and_wait(
                topic,
                value=kafka_message.value,
                key=kafka_message.key,
                headers=[(k, v.encode('utf-8')) for k, v in kafka_message.headers.items()]
            )
            
            self.logger.debug(
                "Message sent to Kafka",
                topic=topic,
                key=key,
                message_size=len(json.dumps(message))
            )
            
            return True
            
        except Exception as e:
            self.log_error(e, {"operation": "kafka_send", "topic": topic})
            return False
    
    async def send_agent_event(
        self,
        event_type: str,
        agent_name: str,
        session_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Send agent-specific event."""
        message = {
            "event_type": event_type,
            "agent_name": agent_name,
            "session_id": session_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_message(
            topic="agent-events",
            message=message,
            key=f"{agent_name}:{session_id}"
        )
    
    async def send_system_event(
        self,
        event_type: str,
        component: str,
        data: Dict[str, Any],
        severity: str = "info"
    ) -> bool:
        """Send system-level event."""
        message = {
            "event_type": event_type,
            "component": component,
            "severity": severity,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_message(
            topic="system-events",
            message=message,
            key=component
        )
    
    # Consumer Methods
    async def subscribe_to_topic(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], None],
        group_id: Optional[str] = None
    ) -> bool:
        """Subscribe to a Kafka topic with message handler."""
        try:
            consumer_group = group_id or f"{self.group_id}-{topic}"
            
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=consumer_group,
                client_id=f"{self.client_id}-consumer-{topic}",
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            await consumer.start()
            self.consumers[topic] = consumer
            self.message_handlers[topic] = handler
            
            # Start consuming in background
            asyncio.create_task(self._consume_messages(topic, consumer, handler))
            
            self.logger.info(
                "Subscribed to Kafka topic",
                topic=topic,
                group_id=consumer_group
            )
            
            return True
            
        except Exception as e:
            self.log_error(e, {"operation": "kafka_subscribe", "topic": topic})
            return False
    
    async def _consume_messages(
        self,
        topic: str,
        consumer: AIOKafkaConsumer,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """Consume messages from a topic."""
        self._consuming = True
        
        try:
            async for message in consumer:
                if not self._consuming:
                    break
                
                try:
                    # Process message
                    await handler(message.value)
                    
                    self.logger.debug(
                        "Message processed",
                        topic=topic,
                        offset=message.offset,
                        partition=message.partition
                    )
                    
                except Exception as e:
                    self.log_error(e, {
                        "operation": "message_processing",
                        "topic": topic,
                        "offset": message.offset
                    })
                    
        except Exception as e:
            self.log_error(e, {"operation": "kafka_consume", "topic": topic})
        
        finally:
            self.logger.info("Stopped consuming messages", topic=topic)
    
    # Event Handlers
    async def handle_agent_event(self, message: Dict[str, Any]):
        """Handle agent events."""
        event_type = message.get("event_type")
        agent_name = message.get("agent_name")
        
        self.logger.info(
            "Agent event received",
            event_type=event_type,
            agent_name=agent_name,
            session_id=message.get("session_id")
        )
        
        # Process based on event type
        if event_type == "task_started":
            await self._handle_task_started(message)
        elif event_type == "task_completed":
            await self._handle_task_completed(message)
        elif event_type == "error_occurred":
            await self._handle_error_occurred(message)
    
    async def handle_system_event(self, message: Dict[str, Any]):
        """Handle system events."""
        event_type = message.get("event_type")
        component = message.get("component")
        severity = message.get("severity", "info")
        
        self.logger.info(
            "System event received",
            event_type=event_type,
            component=component,
            severity=severity
        )
        
        # Process based on severity
        if severity in ["error", "critical"]:
            await self._handle_critical_event(message)
    
    async def _handle_task_started(self, message: Dict[str, Any]):
        """Handle task started event."""
        # Could trigger monitoring, logging, etc.
        pass
    
    async def _handle_task_completed(self, message: Dict[str, Any]):
        """Handle task completed event."""
        # Could trigger analytics, cleanup, etc.
        pass
    
    async def _handle_error_occurred(self, message: Dict[str, Any]):
        """Handle error event."""
        # Could trigger alerts, recovery, etc.
        pass
    
    async def _handle_critical_event(self, message: Dict[str, Any]):
        """Handle critical system event."""
        # Could trigger alerts, emergency procedures, etc.
        pass
    
    # Utility Methods
    async def get_topic_info(self, topic: str) -> Dict[str, Any]:
        """Get information about a topic."""
        try:
            # This would require admin client in production
            return {
                "topic": topic,
                "status": "active" if topic in self.consumers else "inactive",
                "consumer_count": 1 if topic in self.consumers else 0
            }
            
        except Exception as e:
            self.log_error(e, {"operation": "get_topic_info", "topic": topic})
            return {"topic": topic, "status": "error", "error": str(e)}
    
    async def list_topics(self) -> List[str]:
        """List available topics."""
        try:
            # This would require admin client in production
            return list(self.consumers.keys())
            
        except Exception as e:
            self.log_error(e, {"operation": "list_topics"})
            return []
