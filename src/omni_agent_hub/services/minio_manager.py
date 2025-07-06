"""
MinIO Manager for object storage and file management.

This module provides async MinIO client functionality for storing
and retrieving files, documents, and artifacts.
"""

import io
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, BinaryIO
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

from minio import Minio
from minio.error import S3Error
from urllib3.poolmanager import PoolManager

from ..core.logging import LoggerMixin
from ..core.exceptions import ServiceError


@dataclass
class FileMetadata:
    """File metadata structure."""
    bucket: str
    object_name: str
    size: int
    content_type: str
    etag: str
    last_modified: datetime
    metadata: Dict[str, str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class UploadResult:
    """Upload operation result."""
    success: bool
    bucket: str
    object_name: str
    etag: Optional[str] = None
    size: Optional[int] = None
    error: Optional[str] = None


class MinIOManager(LoggerMixin):
    """Async MinIO manager for object storage."""
    
    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        secure: bool = False,
        region: str = "us-east-1"
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.region = region
        
        self.client: Optional[Minio] = None
        self._initialized = False
        
        # Default buckets
        self.default_buckets = [
            "agent-artifacts",
            "user-uploads", 
            "system-logs",
            "knowledge-base",
            "temp-files"
        ]
        
        self.logger.info("MinIO manager initialized", endpoint=endpoint)
    
    async def initialize(self) -> None:
        """Initialize MinIO client and create default buckets."""
        try:
            self.logger.info("Initializing MinIO connection")
            
            # Create MinIO client
            self.client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                region=self.region
            )
            
            # Test connection
            await self._test_connection()
            
            # Create default buckets
            await self._create_default_buckets()
            
            self._initialized = True
            self.logger.info("MinIO initialized successfully")
            
        except Exception as e:
            self.log_error(e, {"operation": "minio_initialization"})
            raise ServiceError(f"Failed to initialize MinIO: {str(e)}")
    
    async def close(self) -> None:
        """Close MinIO connections."""
        try:
            # MinIO client doesn't need explicit closing
            self.logger.info("MinIO connections closed")
            
        except Exception as e:
            self.log_error(e, {"operation": "minio_close"})
    
    async def _test_connection(self) -> None:
        """Test MinIO connection."""
        try:
            # List buckets to test connection
            await asyncio.get_event_loop().run_in_executor(
                None, list, self.client.list_buckets()
            )
            self.logger.info("MinIO connection test successful")
            
        except Exception as e:
            raise ServiceError(f"MinIO connection test failed: {str(e)}")
    
    async def _create_default_buckets(self) -> None:
        """Create default buckets if they don't exist."""
        for bucket_name in self.default_buckets:
            try:
                bucket_exists = await asyncio.get_event_loop().run_in_executor(
                    None, self.client.bucket_exists, bucket_name
                )
                
                if not bucket_exists:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.client.make_bucket, bucket_name
                    )
                    self.logger.info("Created bucket", bucket=bucket_name)
                else:
                    self.logger.debug("Bucket already exists", bucket=bucket_name)
                    
            except Exception as e:
                self.log_error(e, {"operation": "create_bucket", "bucket": bucket_name})
    
    def _ensure_initialized(self) -> None:
        """Ensure MinIO is initialized."""
        if not self._initialized or not self.client:
            raise ServiceError("MinIO not initialized")
    
    # Bucket Operations
    async def create_bucket(self, bucket_name: str) -> bool:
        """Create a new bucket."""
        self._ensure_initialized()
        
        try:
            bucket_exists = await asyncio.get_event_loop().run_in_executor(
                None, self.client.bucket_exists, bucket_name
            )
            
            if not bucket_exists:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.client.make_bucket, bucket_name
                )
                self.logger.info("Bucket created", bucket=bucket_name)
                return True
            else:
                self.logger.info("Bucket already exists", bucket=bucket_name)
                return True
                
        except Exception as e:
            self.log_error(e, {"operation": "create_bucket", "bucket": bucket_name})
            return False
    
    async def list_buckets(self) -> List[str]:
        """List all buckets."""
        self._ensure_initialized()
        
        try:
            buckets = await asyncio.get_event_loop().run_in_executor(
                None, list, self.client.list_buckets()
            )
            return [bucket.name for bucket in buckets]
            
        except Exception as e:
            self.log_error(e, {"operation": "list_buckets"})
            return []
    
    # File Upload Operations
    async def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Union[str, Path],
        content_type: Optional[str] = None,
        metadata: Dict[str, str] = None
    ) -> UploadResult:
        """Upload a file from local path."""
        self._ensure_initialized()
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return UploadResult(
                    success=False,
                    bucket=bucket_name,
                    object_name=object_name,
                    error="File not found"
                )
            
            # Auto-detect content type if not provided
            if content_type is None:
                content_type = self._get_content_type(file_path)
            
            # Prepare metadata
            file_metadata = metadata or {}
            file_metadata.update({
                "uploaded_at": datetime.utcnow().isoformat(),
                "original_name": file_path.name,
                "source": "omni-agent-hub"
            })
            
            # Upload file
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.fput_object,
                bucket_name,
                object_name,
                str(file_path),
                content_type,
                file_metadata
            )
            
            self.logger.info(
                "File uploaded successfully",
                bucket=bucket_name,
                object=object_name,
                size=file_path.stat().st_size
            )
            
            return UploadResult(
                success=True,
                bucket=bucket_name,
                object_name=object_name,
                etag=result.etag,
                size=file_path.stat().st_size
            )
            
        except Exception as e:
            self.log_error(e, {
                "operation": "upload_file",
                "bucket": bucket_name,
                "object": object_name
            })
            return UploadResult(
                success=False,
                bucket=bucket_name,
                object_name=object_name,
                error=str(e)
            )
    
    async def upload_data(
        self,
        bucket_name: str,
        object_name: str,
        data: Union[str, bytes, BinaryIO],
        content_type: str = "application/octet-stream",
        metadata: Dict[str, str] = None
    ) -> UploadResult:
        """Upload data directly to MinIO."""
        self._ensure_initialized()
        
        try:
            # Convert data to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
                if content_type == "application/octet-stream":
                    content_type = "text/plain"
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                # Assume it's a file-like object
                data_bytes = data.read()
                if hasattr(data, 'seek'):
                    data.seek(0)
            
            # Create BytesIO object
            data_stream = io.BytesIO(data_bytes)
            data_size = len(data_bytes)
            
            # Prepare metadata
            file_metadata = metadata or {}
            file_metadata.update({
                "uploaded_at": datetime.utcnow().isoformat(),
                "source": "omni-agent-hub",
                "size": str(data_size)
            })
            
            # Upload data
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.put_object,
                bucket_name,
                object_name,
                data_stream,
                data_size,
                content_type,
                file_metadata
            )
            
            self.logger.info(
                "Data uploaded successfully",
                bucket=bucket_name,
                object=object_name,
                size=data_size
            )
            
            return UploadResult(
                success=True,
                bucket=bucket_name,
                object_name=object_name,
                etag=result.etag,
                size=data_size
            )
            
        except Exception as e:
            self.log_error(e, {
                "operation": "upload_data",
                "bucket": bucket_name,
                "object": object_name
            })
            return UploadResult(
                success=False,
                bucket=bucket_name,
                object_name=object_name,
                error=str(e)
            )
    
    # File Download Operations
    async def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: Union[str, Path]
    ) -> bool:
        """Download a file to local path."""
        self._ensure_initialized()
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.fget_object,
                bucket_name,
                object_name,
                str(file_path)
            )
            
            self.logger.info(
                "File downloaded successfully",
                bucket=bucket_name,
                object=object_name,
                path=str(file_path)
            )
            
            return True
            
        except Exception as e:
            self.log_error(e, {
                "operation": "download_file",
                "bucket": bucket_name,
                "object": object_name
            })
            return False
    
    async def get_object_data(
        self,
        bucket_name: str,
        object_name: str
    ) -> Optional[bytes]:
        """Get object data as bytes."""
        self._ensure_initialized()
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.get_object,
                bucket_name,
                object_name
            )
            
            data = response.read()
            response.close()
            response.release_conn()
            
            self.logger.debug(
                "Object data retrieved",
                bucket=bucket_name,
                object=object_name,
                size=len(data)
            )
            
            return data
            
        except Exception as e:
            self.log_error(e, {
                "operation": "get_object_data",
                "bucket": bucket_name,
                "object": object_name
            })
            return None
    
    # File Management Operations
    async def delete_object(self, bucket_name: str, object_name: str) -> bool:
        """Delete an object."""
        self._ensure_initialized()
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.remove_object,
                bucket_name,
                object_name
            )
            
            self.logger.info(
                "Object deleted",
                bucket=bucket_name,
                object=object_name
            )
            
            return True
            
        except Exception as e:
            self.log_error(e, {
                "operation": "delete_object",
                "bucket": bucket_name,
                "object": object_name
            })
            return False
    
    async def list_objects(
        self,
        bucket_name: str,
        prefix: str = "",
        recursive: bool = True
    ) -> List[FileMetadata]:
        """List objects in a bucket."""
        self._ensure_initialized()
        
        try:
            objects = await asyncio.get_event_loop().run_in_executor(
                None,
                list,
                self.client.list_objects(bucket_name, prefix=prefix, recursive=recursive)
            )
            
            result = []
            for obj in objects:
                metadata = FileMetadata(
                    bucket=bucket_name,
                    object_name=obj.object_name,
                    size=obj.size,
                    content_type=obj.content_type or "application/octet-stream",
                    etag=obj.etag,
                    last_modified=obj.last_modified,
                    metadata=obj.metadata or {}
                )
                result.append(metadata)
            
            self.logger.debug(
                "Objects listed",
                bucket=bucket_name,
                count=len(result),
                prefix=prefix
            )
            
            return result
            
        except Exception as e:
            self.log_error(e, {
                "operation": "list_objects",
                "bucket": bucket_name,
                "prefix": prefix
            })
            return []
    
    async def get_object_info(
        self,
        bucket_name: str,
        object_name: str
    ) -> Optional[FileMetadata]:
        """Get object metadata."""
        self._ensure_initialized()
        
        try:
            stat = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.stat_object,
                bucket_name,
                object_name
            )
            
            return FileMetadata(
                bucket=bucket_name,
                object_name=object_name,
                size=stat.size,
                content_type=stat.content_type,
                etag=stat.etag,
                last_modified=stat.last_modified,
                metadata=stat.metadata or {}
            )
            
        except Exception as e:
            self.log_error(e, {
                "operation": "get_object_info",
                "bucket": bucket_name,
                "object": object_name
            })
            return None
    
    # Utility Methods
    def _get_content_type(self, file_path: Path) -> str:
        """Get content type based on file extension."""
        extension = file_path.suffix.lower()
        
        content_types = {
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.mp4': 'video/mp4',
            '.mp3': 'audio/mpeg',
            '.zip': 'application/zip',
            '.csv': 'text/csv',
            '.xml': 'application/xml',
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript'
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    async def generate_presigned_url(
        self,
        bucket_name: str,
        object_name: str,
        expires: timedelta = timedelta(hours=1)
    ) -> Optional[str]:
        """Generate presigned URL for object access."""
        self._ensure_initialized()
        
        try:
            url = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.presigned_get_object,
                bucket_name,
                object_name,
                expires
            )
            
            self.logger.debug(
                "Presigned URL generated",
                bucket=bucket_name,
                object=object_name,
                expires=expires
            )
            
            return url
            
        except Exception as e:
            self.log_error(e, {
                "operation": "generate_presigned_url",
                "bucket": bucket_name,
                "object": object_name
            })
            return None
