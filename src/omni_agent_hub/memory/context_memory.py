"""
Advanced Context-Aware Memory System for Omni-Agent Hub.

This module implements sophisticated memory management with hierarchical storage,
semantic search, and intelligent context retrieval for enhanced agent performance.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
from enum import Enum

from ..core.logging import LoggerMixin
from ..services.redis_manager import RedisManager
from ..services.vector_db import VectorDatabaseManager


class MemoryType(str, Enum):
    """Types of memory for different use cases."""
    EPISODIC = "episodic"  # Specific interactions and events
    SEMANTIC = "semantic"  # General knowledge and facts
    PROCEDURAL = "procedural"  # How-to knowledge and processes
    WORKING = "working"  # Temporary, session-based memory
    LONG_TERM = "long_term"  # Persistent, important information


class MemoryImportance(str, Enum):
    """Importance levels for memory prioritization."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TEMPORARY = "temporary"


@dataclass
class MemoryItem:
    """Represents a single memory item with metadata."""
    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    context: Dict[str, Any]
    embedding: Optional[List[float]]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    decay_factor: float
    tags: Set[str]
    related_memories: Set[str]
    confidence: float
    source: str


@dataclass
class ContextWindow:
    """Represents the current context window for memory retrieval."""
    session_id: str
    user_id: str
    current_task: str
    domain: str
    complexity_level: str
    recent_memories: List[str]
    relevant_memories: List[str]
    working_memory: Dict[str, Any]
    attention_weights: Dict[str, float]


class ContextAwareMemorySystem(LoggerMixin):
    """Advanced memory system with context-aware retrieval and hierarchical storage."""
    
    def __init__(
        self,
        redis_manager: RedisManager,
        vector_db: VectorDatabaseManager,
        max_working_memory: int = 10,
        max_context_window: int = 50
    ):
        self.redis_manager = redis_manager
        self.vector_db = vector_db
        self.max_working_memory = max_working_memory
        self.max_context_window = max_context_window
        
        # Memory storage
        self.working_memory: Dict[str, MemoryItem] = {}
        self.context_windows: Dict[str, ContextWindow] = {}
        
        # Memory management parameters
        self.decay_rate = 0.95  # Daily decay factor
        self.importance_boost = 1.2  # Boost for important memories
        self.recency_weight = 0.3
        self.relevance_weight = 0.4
        self.importance_weight = 0.3
        
        # Semantic clustering
        self.memory_clusters: Dict[str, List[str]] = defaultdict(list)
        self.cluster_embeddings: Dict[str, np.ndarray] = {}
        
        self.logger.info("Context-aware memory system initialized")
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: MemoryImportance,
        context: Dict[str, Any],
        session_id: str,
        tags: Set[str] = None,
        embedding: List[float] = None
    ) -> str:
        """Store a new memory item with context awareness."""
        
        memory_id = f"{memory_type}_{session_id}_{datetime.utcnow().timestamp()}"
        
        # Generate embedding if not provided
        if embedding is None and self.vector_db:
            try:
                embeddings = await self.vector_db.generate_embeddings([content])
                embedding = embeddings[0] if embeddings else None
            except Exception as e:
                self.logger.warning("Failed to generate embedding", error=str(e))
                embedding = None
        
        # Create memory item
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            context=context,
            embedding=embedding,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            decay_factor=1.0,
            tags=tags or set(),
            related_memories=set(),
            confidence=1.0,
            source=context.get("source", "user_interaction")
        )
        
        # Store in appropriate memory layer
        await self._store_by_type(memory_item, session_id)
        
        # Update context window
        await self._update_context_window(session_id, memory_id, context)
        
        # Find and link related memories
        await self._link_related_memories(memory_item)
        
        # Update semantic clusters
        await self._update_clusters(memory_item)
        
        self.logger.info(
            "Stored memory",
            memory_id=memory_id,
            type=memory_type,
            importance=importance
        )
        
        return memory_id
    
    async def retrieve_memories(
        self,
        query: str,
        session_id: str,
        memory_types: List[MemoryType] = None,
        max_results: int = 10,
        context_filter: Dict[str, Any] = None
    ) -> List[MemoryItem]:
        """Retrieve relevant memories using context-aware search."""
        
        # Get current context window
        context_window = self.context_windows.get(session_id)
        
        # Generate query embedding
        query_embedding = None
        if self.vector_db:
            try:
                embeddings = await self.vector_db.generate_embeddings([query])
                query_embedding = embeddings[0] if embeddings else None
            except Exception as e:
                self.logger.warning("Failed to generate query embedding", error=str(e))
        
        # Search strategies
        candidates = []
        
        # 1. Semantic search using embeddings
        if query_embedding:
            semantic_candidates = await self._semantic_search(
                query_embedding, memory_types, max_results * 2
            )
            candidates.extend(semantic_candidates)
        
        # 2. Keyword-based search
        keyword_candidates = await self._keyword_search(
            query, memory_types, max_results
        )
        candidates.extend(keyword_candidates)
        
        # 3. Context-based retrieval
        if context_window:
            context_candidates = await self._context_based_search(
                context_window, query, max_results
            )
            candidates.extend(context_candidates)
        
        # Remove duplicates and score
        unique_candidates = {item.id: item for item in candidates}
        scored_memories = []
        
        for memory_item in unique_candidates.values():
            score = await self._calculate_relevance_score(
                memory_item, query, query_embedding, context_window, context_filter
            )
            scored_memories.append((memory_item, score))
        
        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        results = [item for item, score in scored_memories[:max_results]]
        
        # Update access patterns
        for memory_item in results:
            await self._update_access_pattern(memory_item)
        
        self.logger.info(
            "Retrieved memories",
            query_length=len(query),
            results_count=len(results),
            session_id=session_id
        )
        
        return results
    
    async def _store_by_type(self, memory_item: MemoryItem, session_id: str):
        """Store memory item based on its type."""
        
        if memory_item.memory_type == MemoryType.WORKING:
            # Store in working memory (Redis)
            self.working_memory[memory_item.id] = memory_item
            
            # Limit working memory size
            if len(self.working_memory) > self.max_working_memory:
                # Remove least recently used
                oldest_id = min(
                    self.working_memory.keys(),
                    key=lambda k: self.working_memory[k].last_accessed
                )
                del self.working_memory[oldest_id]
            
            # Store in Redis for session persistence
            await self.redis_manager.set(
                f"working_memory:{session_id}:{memory_item.id}",
                json.dumps(asdict(memory_item), default=str),
                expire=3600  # 1 hour
            )
        
        elif memory_item.memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
            # Store in vector database for semantic search
            if self.vector_db and memory_item.embedding:
                await self.vector_db.insert_vectors(
                    vectors=[memory_item.embedding],
                    metadata=[{
                        "id": memory_item.id,
                        "content": memory_item.content,
                        "type": memory_item.memory_type,
                        "importance": memory_item.importance,
                        "session_id": session_id,
                        "created_at": memory_item.created_at.isoformat(),
                        "tags": list(memory_item.tags)
                    }]
                )
        
        elif memory_item.memory_type == MemoryType.LONG_TERM:
            # Store in both Redis (for fast access) and vector DB (for search)
            await self.redis_manager.set(
                f"long_term_memory:{memory_item.id}",
                json.dumps(asdict(memory_item), default=str),
                expire=86400 * 30  # 30 days
            )
            
            if self.vector_db and memory_item.embedding:
                await self.vector_db.insert_vectors(
                    vectors=[memory_item.embedding],
                    metadata=[{
                        "id": memory_item.id,
                        "content": memory_item.content,
                        "type": memory_item.memory_type,
                        "importance": memory_item.importance,
                        "created_at": memory_item.created_at.isoformat()
                    }]
                )
    
    async def _update_context_window(
        self,
        session_id: str,
        memory_id: str,
        context: Dict[str, Any]
    ):
        """Update the context window for a session."""
        
        if session_id not in self.context_windows:
            self.context_windows[session_id] = ContextWindow(
                session_id=session_id,
                user_id=context.get("user_id", "unknown"),
                current_task=context.get("task", "general"),
                domain=context.get("domain", "general"),
                complexity_level=context.get("complexity", "medium"),
                recent_memories=[],
                relevant_memories=[],
                working_memory={},
                attention_weights={}
            )
        
        context_window = self.context_windows[session_id]
        
        # Update recent memories
        context_window.recent_memories.append(memory_id)
        if len(context_window.recent_memories) > self.max_context_window:
            context_window.recent_memories = context_window.recent_memories[-self.max_context_window:]
        
        # Update current task and domain if provided
        if "task" in context:
            context_window.current_task = context["task"]
        if "domain" in context:
            context_window.domain = context["domain"]
        if "complexity" in context:
            context_window.complexity_level = context["complexity"]
    
    async def _link_related_memories(self, memory_item: MemoryItem):
        """Find and link related memories using semantic similarity."""
        
        if not memory_item.embedding:
            return
        
        # Search for similar memories
        similar_memories = await self._semantic_search(
            memory_item.embedding,
            memory_types=None,
            max_results=5,
            similarity_threshold=0.8
        )
        
        # Link bidirectionally
        for similar_memory in similar_memories:
            if similar_memory.id != memory_item.id:
                memory_item.related_memories.add(similar_memory.id)
                similar_memory.related_memories.add(memory_item.id)
    
    async def _update_clusters(self, memory_item: MemoryItem):
        """Update semantic clusters with new memory."""
        
        if not memory_item.embedding:
            return
        
        embedding = np.array(memory_item.embedding)
        best_cluster = None
        best_similarity = 0.0
        
        # Find best matching cluster
        for cluster_id, cluster_embedding in self.cluster_embeddings.items():
            similarity = np.dot(embedding, cluster_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(cluster_embedding)
            )
            
            if similarity > best_similarity and similarity > 0.7:
                best_similarity = similarity
                best_cluster = cluster_id
        
        if best_cluster:
            # Add to existing cluster
            self.memory_clusters[best_cluster].append(memory_item.id)
            
            # Update cluster centroid
            cluster_memories = self.memory_clusters[best_cluster]
            if len(cluster_memories) > 1:
                # Recalculate centroid (simplified)
                self.cluster_embeddings[best_cluster] = (
                    self.cluster_embeddings[best_cluster] * 0.9 + embedding * 0.1
                )
        else:
            # Create new cluster
            cluster_id = f"cluster_{len(self.memory_clusters)}"
            self.memory_clusters[cluster_id] = [memory_item.id]
            self.cluster_embeddings[cluster_id] = embedding
    
    async def _semantic_search(
        self,
        query_embedding: List[float],
        memory_types: List[MemoryType] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[MemoryItem]:
        """Perform semantic search using vector similarity."""
        
        if not self.vector_db:
            return []
        
        try:
            # Build filter for memory types
            filter_conditions = {}
            if memory_types:
                filter_conditions["type"] = {"$in": [mt.value for mt in memory_types]}
            
            # Search vector database
            search_results = await self.vector_db.search_vectors(
                query_vector=query_embedding,
                limit=max_results,
                filter_conditions=filter_conditions
            )
            
            # Convert to MemoryItem objects
            memory_items = []
            for result in search_results:
                if result.get("score", 0) >= similarity_threshold:
                    # Reconstruct MemoryItem from metadata
                    metadata = result.get("metadata", {})
                    memory_item = MemoryItem(
                        id=metadata.get("id", ""),
                        content=metadata.get("content", ""),
                        memory_type=MemoryType(metadata.get("type", "semantic")),
                        importance=MemoryImportance(metadata.get("importance", "medium")),
                        context={},
                        embedding=query_embedding,  # Approximate
                        created_at=datetime.fromisoformat(metadata.get("created_at", datetime.utcnow().isoformat())),
                        last_accessed=datetime.utcnow(),
                        access_count=1,
                        decay_factor=1.0,
                        tags=set(metadata.get("tags", [])),
                        related_memories=set(),
                        confidence=result.get("score", 0.5),
                        source="vector_search"
                    )
                    memory_items.append(memory_item)
            
            return memory_items
            
        except Exception as e:
            self.logger.error("Semantic search failed", error=str(e))
            return []
    
    async def _keyword_search(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        max_results: int = 10
    ) -> List[MemoryItem]:
        """Perform keyword-based search in memory content."""
        
        # Simple keyword matching - can be enhanced with fuzzy search
        query_words = set(query.lower().split())
        matching_memories = []
        
        # Search working memory
        for memory_item in self.working_memory.values():
            if memory_types and memory_item.memory_type not in memory_types:
                continue
            
            content_words = set(memory_item.content.lower().split())
            overlap = len(query_words.intersection(content_words))
            
            if overlap > 0:
                memory_item.confidence = overlap / len(query_words)
                matching_memories.append(memory_item)
        
        # Sort by keyword overlap and return top results
        matching_memories.sort(key=lambda x: x.confidence, reverse=True)
        return matching_memories[:max_results]
    
    async def _context_based_search(
        self,
        context_window: ContextWindow,
        query: str,
        max_results: int = 10
    ) -> List[MemoryItem]:
        """Search based on current context and attention weights."""
        
        context_memories = []
        
        # Get memories from recent context
        for memory_id in context_window.recent_memories[-10:]:  # Last 10 memories
            if memory_id in self.working_memory:
                memory_item = self.working_memory[memory_id]
                
                # Calculate context relevance
                relevance = self._calculate_context_relevance(memory_item, context_window)
                memory_item.confidence = relevance
                
                if relevance > 0.3:  # Threshold for context relevance
                    context_memories.append(memory_item)
        
        # Sort by context relevance
        context_memories.sort(key=lambda x: x.confidence, reverse=True)
        return context_memories[:max_results]
    
    def _calculate_context_relevance(
        self,
        memory_item: MemoryItem,
        context_window: ContextWindow
    ) -> float:
        """Calculate how relevant a memory is to the current context."""
        
        relevance = 0.0
        
        # Domain similarity
        memory_domain = memory_item.context.get("domain", "general")
        if memory_domain == context_window.domain:
            relevance += 0.3
        
        # Task similarity
        memory_task = memory_item.context.get("task", "general")
        if memory_task == context_window.current_task:
            relevance += 0.3
        
        # Complexity match
        memory_complexity = memory_item.context.get("complexity", "medium")
        if memory_complexity == context_window.complexity_level:
            relevance += 0.2
        
        # Recency boost
        time_diff = datetime.utcnow() - memory_item.last_accessed
        recency_score = max(0, 1 - (time_diff.total_seconds() / 3600))  # Decay over 1 hour
        relevance += recency_score * 0.2
        
        return min(1.0, relevance)
    
    async def _calculate_relevance_score(
        self,
        memory_item: MemoryItem,
        query: str,
        query_embedding: Optional[List[float]],
        context_window: Optional[ContextWindow],
        context_filter: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate comprehensive relevance score for memory retrieval."""
        
        score = 0.0
        
        # Semantic similarity (if embeddings available)
        if query_embedding and memory_item.embedding:
            semantic_sim = np.dot(query_embedding, memory_item.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory_item.embedding)
            )
            score += semantic_sim * self.relevance_weight
        
        # Keyword relevance
        query_words = set(query.lower().split())
        content_words = set(memory_item.content.lower().split())
        keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
        score += keyword_overlap * 0.3
        
        # Recency score
        time_diff = datetime.utcnow() - memory_item.last_accessed
        recency_score = max(0, 1 - (time_diff.total_seconds() / 86400))  # Decay over 1 day
        score += recency_score * self.recency_weight
        
        # Importance boost
        importance_multiplier = {
            MemoryImportance.CRITICAL: 1.5,
            MemoryImportance.HIGH: 1.2,
            MemoryImportance.MEDIUM: 1.0,
            MemoryImportance.LOW: 0.8,
            MemoryImportance.TEMPORARY: 0.5
        }
        score *= importance_multiplier.get(memory_item.importance, 1.0)
        
        # Context relevance
        if context_window:
            context_relevance = self._calculate_context_relevance(memory_item, context_window)
            score += context_relevance * 0.2
        
        # Access frequency boost
        access_boost = min(0.2, memory_item.access_count / 100.0)
        score += access_boost
        
        # Apply decay factor
        score *= memory_item.decay_factor
        
        # Context filter
        if context_filter:
            for key, value in context_filter.items():
                if key in memory_item.context and memory_item.context[key] != value:
                    score *= 0.5  # Penalty for context mismatch
        
        return min(1.0, max(0.0, score))
    
    async def _update_access_pattern(self, memory_item: MemoryItem):
        """Update access patterns for a memory item."""
        
        memory_item.last_accessed = datetime.utcnow()
        memory_item.access_count += 1
        
        # Boost importance based on access frequency
        if memory_item.access_count > 10 and memory_item.importance == MemoryImportance.MEDIUM:
            memory_item.importance = MemoryImportance.HIGH
        elif memory_item.access_count > 50 and memory_item.importance == MemoryImportance.HIGH:
            memory_item.importance = MemoryImportance.CRITICAL
    
    async def decay_memories(self):
        """Apply decay to memories based on time and usage patterns."""
        
        current_time = datetime.utcnow()
        
        for memory_item in self.working_memory.values():
            # Calculate time-based decay
            time_diff = current_time - memory_item.last_accessed
            days_since_access = time_diff.total_seconds() / 86400
            
            # Apply decay
            memory_item.decay_factor *= (self.decay_rate ** days_since_access)
            
            # Remove very low relevance memories
            if memory_item.decay_factor < 0.1 and memory_item.importance == MemoryImportance.TEMPORARY:
                # Mark for removal
                memory_item.decay_factor = 0.0
        
        # Remove decayed memories
        self.working_memory = {
            k: v for k, v in self.working_memory.items()
            if v.decay_factor > 0.0
        }
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        
        stats = {
            "working_memory_size": len(self.working_memory),
            "total_clusters": len(self.memory_clusters),
            "active_context_windows": len(self.context_windows),
            "memory_type_distribution": defaultdict(int),
            "importance_distribution": defaultdict(int),
            "avg_access_count": 0,
            "avg_decay_factor": 0,
            "most_accessed_memories": [],
            "cluster_sizes": {}
        }
        
        if self.working_memory:
            total_access = sum(m.access_count for m in self.working_memory.values())
            total_decay = sum(m.decay_factor for m in self.working_memory.values())
            
            stats["avg_access_count"] = total_access / len(self.working_memory)
            stats["avg_decay_factor"] = total_decay / len(self.working_memory)
            
            # Distribution stats
            for memory_item in self.working_memory.values():
                stats["memory_type_distribution"][memory_item.memory_type.value] += 1
                stats["importance_distribution"][memory_item.importance.value] += 1
            
            # Most accessed memories
            sorted_memories = sorted(
                self.working_memory.values(),
                key=lambda x: x.access_count,
                reverse=True
            )
            stats["most_accessed_memories"] = [
                {"id": m.id, "content": m.content[:100], "access_count": m.access_count}
                for m in sorted_memories[:5]
            ]
        
        # Cluster statistics
        for cluster_id, memory_ids in self.memory_clusters.items():
            stats["cluster_sizes"][cluster_id] = len(memory_ids)
        
        return stats
