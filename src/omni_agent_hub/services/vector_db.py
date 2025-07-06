"""
Vector database manager for Milvus operations.

This module provides async vector database operations for embedding storage,
similarity search, and knowledge retrieval.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)

from ..core.logging import LoggerMixin
from ..core.exceptions import VectorDatabaseError


class VectorDatabaseManager(LoggerMixin):
    """Async vector database manager for Milvus operations."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        collection_name: str = "omni_embeddings"
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None
        self._initialized = False
        
        # Collection schema configuration
        self.dimension = 1536  # OpenAI text-embedding-3-small dimension
        self.index_type = "IVF_FLAT"
        self.metric_type = "COSINE"
    
    async def initialize(self) -> None:
        """Initialize Milvus connection and collection."""
        try:
            self.logger.info("Initializing Milvus connection")
            
            # Connect to Milvus
            await self._connect()
            
            # Create or load collection
            await self._setup_collection()
            
            self._initialized = True
            self.logger.info("Vector database initialized successfully")
            
        except Exception as e:
            self.log_error(e, {"operation": "vector_db_initialization"})
            raise VectorDatabaseError(f"Failed to initialize vector database: {str(e)}")
    
    async def close(self) -> None:
        """Close Milvus connections."""
        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
                self.logger.info("Vector database connections closed")
        except Exception as e:
            self.log_error(e, {"operation": "vector_db_close"})
    
    async def _connect(self) -> None:
        """Connect to Milvus server."""
        try:
            # Run connection in thread pool since pymilvus is sync
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password
                )
            )
            
            self.logger.info("Connected to Milvus server")
            
        except Exception as e:
            raise VectorDatabaseError(f"Failed to connect to Milvus: {str(e)}")
    
    async def _setup_collection(self) -> None:
        """Setup collection schema and indexes."""
        try:
            # Check if collection exists
            has_collection = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: utility.has_collection(self.collection_name)
            )
            
            if not has_collection:
                await self._create_collection()
            else:
                await self._load_collection()
                
        except Exception as e:
            raise VectorDatabaseError(f"Failed to setup collection: {str(e)}")
    
    async def _create_collection(self) -> None:
        """Create new collection with schema."""
        try:
            # Define collection schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    max_length=255,
                    is_primary=True,
                    auto_id=False
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dimension
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=65535
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON
                ),
                FieldSchema(
                    name="timestamp",
                    dtype=DataType.INT64
                ),
                FieldSchema(
                    name="session_id",
                    dtype=DataType.VARCHAR,
                    max_length=255
                )
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Omni-Agent Hub embeddings collection"
            )
            
            # Create collection
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: Collection(
                    name=self.collection_name,
                    schema=schema
                )
            )
            
            # Load collection and create index
            await self._load_collection()
            await self._create_index()

            # Load collection again after creating index
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.load
            )
            
            self.logger.info("Collection created successfully", collection=self.collection_name)
            
        except Exception as e:
            raise VectorDatabaseError(f"Failed to create collection: {str(e)}")
    
    async def _load_collection(self) -> None:
        """Load existing collection."""
        try:
            self.collection = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: Collection(self.collection_name)
            )

            self.logger.info("Collection loaded successfully", collection=self.collection_name)

        except Exception as e:
            raise VectorDatabaseError(f"Failed to load collection: {str(e)}")
    
    async def _create_index(self) -> None:
        """Create vector index for similarity search."""
        try:
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": 1024}
            }
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
            )
            
            self.logger.info("Vector index created successfully")
            
        except Exception as e:
            raise VectorDatabaseError(f"Failed to create index: {str(e)}")
    
    def _ensure_initialized(self) -> None:
        """Ensure vector database is initialized."""
        if not self._initialized or not self.collection:
            raise VectorDatabaseError("Vector database not initialized")
    
    async def insert_embeddings(
        self,
        embeddings_data: List[Dict[str, Any]]
    ) -> List[str]:
        """Insert embeddings into the collection."""
        self._ensure_initialized()
        
        try:
            # Prepare data for insertion
            ids = [item["id"] for item in embeddings_data]
            embeddings = [item["embedding"] for item in embeddings_data]
            contents = [item["content"] for item in embeddings_data]
            metadata = [item.get("metadata", {}) for item in embeddings_data]
            timestamps = [item.get("timestamp", 0) for item in embeddings_data]
            session_ids = [item.get("session_id", "") for item in embeddings_data]
            
            # Insert data
            insert_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.insert([
                    ids, embeddings, contents, metadata, timestamps, session_ids
                ])
            )
            
            # Flush to ensure data is written
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.flush
            )
            
            self.logger.info(
                "Embeddings inserted successfully",
                count=len(embeddings_data),
                collection=self.collection_name
            )
            
            return insert_result.primary_keys
            
        except Exception as e:
            self.log_error(e, {"operation": "insert_embeddings", "count": len(embeddings_data)})
            raise VectorDatabaseError(f"Failed to insert embeddings: {str(e)}")
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        session_id: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        self._ensure_initialized()
        
        try:
            # Prepare search parameters
            search_params = {
                "metric_type": self.metric_type,
                "params": {"nprobe": 10}
            }
            
            # Build filter expression
            filter_expr = []
            if session_id:
                filter_expr.append(f'session_id == "{session_id}"')
            
            if metadata_filter:
                for key, value in metadata_filter.items():
                    if isinstance(value, str):
                        filter_expr.append(f'metadata["{key}"] == "{value}"')
                    else:
                        filter_expr.append(f'metadata["{key}"] == {value}')
            
            expr = " and ".join(filter_expr) if filter_expr else None
            
            # Perform search
            search_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=limit,
                    expr=expr,
                    output_fields=["content", "metadata", "timestamp", "session_id"]
                )
            )
            
            # Format results
            results = []
            for hits in search_results:
                for hit in hits:
                    results.append({
                        "id": hit.id,
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata"),
                        "timestamp": hit.entity.get("timestamp"),
                        "session_id": hit.entity.get("session_id"),
                        "similarity": float(hit.score)
                    })
            
            self.logger.info(
                "Similarity search completed",
                results_count=len(results),
                limit=limit
            )
            
            return results
            
        except Exception as e:
            self.log_error(e, {"operation": "search_similar", "limit": limit})
            raise VectorDatabaseError(f"Failed to search embeddings: {str(e)}")
    
    async def delete_embeddings(
        self,
        ids: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> int:
        """Delete embeddings by IDs or session ID."""
        self._ensure_initialized()
        
        try:
            # Build filter expression
            if ids:
                id_list = ", ".join([f'"{id_}"' for id_ in ids])
                expr = f"id in [{id_list}]"
            elif session_id:
                expr = f'session_id == "{session_id}"'
            else:
                raise VectorDatabaseError("Either ids or session_id must be provided")
            
            # Delete entities
            delete_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.delete(expr)
            )
            
            # Flush to ensure deletion is applied
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.collection.flush
            )
            
            deleted_count = delete_result.delete_count
            self.logger.info(
                "Embeddings deleted successfully",
                deleted_count=deleted_count
            )
            
            return deleted_count
            
        except Exception as e:
            self.log_error(e, {"operation": "delete_embeddings"})
            raise VectorDatabaseError(f"Failed to delete embeddings: {str(e)}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        self._ensure_initialized()
        
        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.num_entities
            )
            
            return {
                "collection_name": self.collection_name,
                "total_entities": stats,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "metric_type": self.metric_type
            }
            
        except Exception as e:
            self.log_error(e, {"operation": "get_collection_stats"})
            raise VectorDatabaseError(f"Failed to get collection stats: {str(e)}")

    async def list_collections(self) -> List[str]:
        """List all collections in Milvus."""
        self._ensure_initialized()

        try:
            collections = await asyncio.get_event_loop().run_in_executor(
                None,
                utility.list_collections
            )
            self.logger.debug("Listed collections", count=len(collections))
            return collections

        except Exception as e:
            self.log_error(e, {"operation": "list_collections"})
            raise VectorDatabaseError(f"Failed to list collections: {str(e)}")
