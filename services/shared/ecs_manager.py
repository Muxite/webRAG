import logging
import os
import aiohttp
from typing import Optional, Dict, Any


class EcsManager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        ecs_enabled_env = os.environ.get("ECS_ENABLED", "").lower()
        self.ecs_enabled = ecs_enabled_env in ("1", "true", "yes")
        
        self.aws_region = os.environ.get("AWS_REGION")
        self.ecs_cluster = os.environ.get("ECS_CLUSTER")
        self.ecs_task_arn = os.environ.get("ECS_TASK_ARN")
        
        self._metadata: Optional[Dict[str, Any]] = None
        self._metadata_loaded = False
        self._client = None
        self._task_arn: Optional[str] = None
        self._cluster: Optional[str] = None
        self._initialized = False
        
        if self.ecs_enabled:
            if not self.aws_region:
                self.logger.warning("ECS_ENABLED is true but AWS_REGION is not set")
            if not self.ecs_cluster and not self.ecs_task_arn:
                self.logger.debug("ECS enabled but cluster/task ARN not provided via env, will try metadata endpoint")
            self._init_client()
    
    def _init_client(self):
        if not self.aws_region:
            return
        
        try:
            import boto3
            self._client = boto3.client('ecs', region_name=self.aws_region)
        except ImportError:
            self.logger.warning("boto3 not available, ECS task protection disabled")
        except Exception as e:
            self.logger.warning(f"Failed to initialize ECS client: {e}")
    
    async def load_metadata(self) -> bool:
        if self._metadata_loaded:
            return self._metadata is not None
        
        self._metadata_loaded = True
        
        if not self.ecs_enabled:
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'http://169.254.170.2/v2/metadata',
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        self._metadata = await resp.json()
                        self.logger.info("ECS metadata loaded successfully")
                        return True
        except Exception as e:
            self.logger.debug(f"ECS metadata endpoint not available (not on ECS): {e}")
        
        return False
    
    def _get_task_arn(self) -> Optional[str]:
        if self._task_arn:
            return self._task_arn
        if self.ecs_task_arn:
            return self.ecs_task_arn
        if self._metadata:
            return self._metadata.get('TaskARN')
        return None
    
    def _get_cluster(self) -> Optional[str]:
        if self._cluster:
            return self._cluster
        if self.ecs_cluster:
            return self.ecs_cluster
        if self._metadata:
            cluster_arn = self._metadata.get('Cluster')
            if cluster_arn:
                return cluster_arn.split('/')[-1]
        return None
    
    async def initialize(self) -> bool:
        if not self.ecs_enabled:
            return False
        
        if self._initialized:
            return True
        
        if not self._metadata_loaded:
            await self.load_metadata()
        
        self._task_arn = self._get_task_arn()
        self._cluster = self._get_cluster()
        
        if not self._task_arn or not self._cluster:
            return False
        
        self._initialized = True
        self.logger.info(f"ECS manager initialized: task={self._task_arn}, cluster={self._cluster}")
        return True
    
    async def update_protection(self, protection_enabled: bool) -> None:
        if not self.ecs_enabled or not self._initialized:
            return
        
        if not self._client or not self._task_arn or not self._cluster:
            return
        
        try:
            from botocore.exceptions import ClientError
            self._client.update_task_protection(
                cluster=self._cluster,
                tasks=[self._task_arn],
                protectionEnabled=protection_enabled,
                expiresInMinutes=60
            )
            self.logger.info(f"Task protection {'enabled' if protection_enabled else 'disabled'}")
        except ImportError:
            pass
        except ClientError as e:
            self.logger.warning(f"Failed to update task protection: {e}")
        except Exception as e:
            self.logger.warning(f"Error updating task protection: {e}")

