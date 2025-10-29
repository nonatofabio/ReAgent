"""
ReactiveSharedMemory - Enhanced memory system for reactive swarm orchestration.

Extends Strands SharedMemory with tiered storage and reactive capabilities.
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class MemoryTier(Enum):
    """Memory storage tiers for different access patterns."""
    LOCAL = "local"        # Fast in-memory cache
    PERSISTENT = "persistent"  # Local persistent storage
    SHARED = "shared"      # Shared across swarm instances
    ARCHIVE = "archive"    # Long-term archival storage


@dataclass
class MemoryEntry:
    """Entry in reactive shared memory."""
    key: str
    value: Any
    tier: MemoryTier
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReactiveSharedMemory:
    """
    Enhanced shared memory system for reactive swarm orchestration.
    
    Features:
    - Tiered storage (local, persistent, shared, archive)
    - Automatic tier management based on access patterns
    - Thread-safe operations for concurrent swarm access
    - Reactive adaptation of storage strategies
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.ReactiveSharedMemory")
        
        # Storage tiers
        self.local_cache: Dict[str, MemoryEntry] = {}
        self.persistent_store: Dict[str, MemoryEntry] = {}
        self.shared_store: Dict[str, MemoryEntry] = {}
        self.archive_store: Dict[str, MemoryEntry] = {}
        
        # Storage configuration
        self.storage_path = Path(storage_path) if storage_path else Path("./reagent_memory")
        self.storage_path.mkdir(exist_ok=True)
        
        # Access patterns for tier optimization
        self.access_patterns: Dict[str, List[float]] = {}
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Configuration
        self.max_local_entries = 1000
        self.max_persistent_entries = 10000
        self.tier_promotion_threshold = 5  # Access count to promote to higher tier
        self.tier_demotion_age = 3600  # Seconds before considering demotion
        
        self.logger.info(f"ReactiveSharedMemory initialized with storage_path: {self.storage_path}")
    
    async def store_with_tier(
        self, 
        key: str, 
        value: Any, 
        tier: Union[str, MemoryTier] = "auto",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store value in specified or automatically determined tier.
        
        Args:
            key: Storage key
            value: Value to store
            tier: Storage tier or "auto" for automatic selection
            metadata: Optional metadata
            
        Returns:
            Storage result with tier information
        """
        async with self._lock:
            if isinstance(tier, str):
                if tier == "auto":
                    tier = self._determine_optimal_tier(key, value)
                else:
                    tier = MemoryTier(tier)
            
            entry = MemoryEntry(
                key=key,
                value=value,
                tier=tier,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            # Store in appropriate tier
            if tier == MemoryTier.LOCAL:
                self.local_cache[key] = entry
                await self._manage_local_cache_size()
            elif tier == MemoryTier.PERSISTENT:
                self.persistent_store[key] = entry
                await self._persist_to_disk(key, entry)
            elif tier == MemoryTier.SHARED:
                self.shared_store[key] = entry
                await self._sync_to_shared_storage(key, entry)
            elif tier == MemoryTier.ARCHIVE:
                self.archive_store[key] = entry
                await self._archive_entry(key, entry)
            
            self.logger.debug(f"Stored in memory: key={key}, tier={tier.value}")
            
            return {
                'success': True,
                'key': key,
                'tier': tier.value,
                'timestamp': entry.timestamp
            }
    
    async def retrieve_with_context(
        self, 
        key: str, 
        include_history: bool = False,
        promote_tier: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve value with access context and optional tier promotion.
        
        Args:
            key: Storage key
            include_history: Include access history in response
            promote_tier: Whether to promote frequently accessed items
            
        Returns:
            Retrieved value with context information
        """
        async with self._lock:
            entry = await self._find_entry_across_tiers(key)
            
            if entry is None:
                return {
                    'found': False,
                    'key': key,
                    'message': 'Key not found in any tier'
                }
            
            # Update access patterns
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Track access pattern
            if key not in self.access_patterns:
                self.access_patterns[key] = []
            self.access_patterns[key].append(time.time())
            
            # Consider tier promotion
            if promote_tier and entry.access_count >= self.tier_promotion_threshold:
                await self._consider_tier_promotion(key, entry)
            
            result = {
                'found': True,
                'key': key,
                'value': entry.value,
                'tier': entry.tier.value,
                'access_count': entry.access_count,
                'last_accessed': entry.last_accessed,
                'metadata': entry.metadata
            }
            
            if include_history:
                result['access_history'] = self.access_patterns.get(key, [])
            
            self.logger.debug(f"Retrieved from memory: key={key}, tier={entry.tier.value}")
            
            return result
    
    async def store_execution_result(self, task: str, result: Any) -> Dict[str, Any]:
        """Store swarm execution result for future reference."""
        key = f"execution:{hash(task)}:{int(time.time())}"
        
        return await self.store_with_tier(
            key=key,
            value=result,
            tier=MemoryTier.PERSISTENT,
            metadata={
                'type': 'execution_result',
                'task': task,
                'timestamp': time.time()
            }
        )
    
    async def get_similar_executions(self, task: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar past executions for learning."""
        # Simple similarity based on task keywords
        task_keywords = set(task.lower().split())
        similar_executions = []
        
        # Search across all tiers
        all_entries = []
        all_entries.extend(self.local_cache.values())
        all_entries.extend(self.persistent_store.values())
        all_entries.extend(self.shared_store.values())
        
        for entry in all_entries:
            if entry.metadata.get('type') == 'execution_result':
                stored_task = entry.metadata.get('task', '')
                stored_keywords = set(stored_task.lower().split())
                
                # Calculate simple similarity score
                intersection = task_keywords.intersection(stored_keywords)
                union = task_keywords.union(stored_keywords)
                similarity = len(intersection) / len(union) if union else 0
                
                if similarity > 0.3:  # Threshold for similarity
                    similar_executions.append({
                        'task': stored_task,
                        'result': entry.value,
                        'similarity': similarity,
                        'timestamp': entry.timestamp
                    })
        
        # Sort by similarity and return top results
        similar_executions.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_executions[:limit]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        async with self._lock:
            stats = {
                'local_cache': {
                    'count': len(self.local_cache),
                    'keys': list(self.local_cache.keys())
                },
                'persistent_store': {
                    'count': len(self.persistent_store),
                    'keys': list(self.persistent_store.keys())
                },
                'shared_store': {
                    'count': len(self.shared_store),
                    'keys': list(self.shared_store.keys())
                },
                'archive_store': {
                    'count': len(self.archive_store),
                    'keys': list(self.archive_store.keys())
                },
                'storage_path': str(self.storage_path),
                'storage_path_exists': self.storage_path.exists()
            }
            
            # Add file system statistics
            if self.storage_path.exists():
                json_files = list(self.storage_path.glob("*.json"))
                stats['disk_files'] = {
                    'count': len(json_files),
                    'files': [f.name for f in json_files],
                    'total_size_bytes': sum(f.stat().st_size for f in json_files)
                }
            else:
                stats['disk_files'] = {
                    'count': 0,
                    'files': [],
                    'total_size_bytes': 0
                }
            
            return stats

    async def optimize_memory_tiers(self) -> Dict[str, Any]:
        """Optimize memory tier assignments based on access patterns."""
        async with self._lock:
            optimizations = {
                'promoted': [],
                'demoted': [],
                'archived': []
            }
            
            current_time = time.time()
            
            # Check all entries for optimization opportunities
            all_stores = [
                (self.local_cache, MemoryTier.LOCAL),
                (self.persistent_store, MemoryTier.PERSISTENT),
                (self.shared_store, MemoryTier.SHARED)
            ]
            
            for store, current_tier in all_stores:
                for key, entry in list(store.items()):
                    age = current_time - entry.timestamp
                    
                    # Consider archival for old, rarely accessed items
                    if (age > self.tier_demotion_age * 24 and  # 24 hours
                        entry.access_count < 2):
                        await self._move_to_archive(key, entry)
                        optimizations['archived'].append(key)
                    
                    # Consider promotion for frequently accessed items
                    elif (entry.access_count >= self.tier_promotion_threshold and
                          current_tier != MemoryTier.LOCAL):
                        await self._promote_entry(key, entry)
                        optimizations['promoted'].append(key)
                    
                    # Consider demotion for old items in high tiers
                    elif (age > self.tier_demotion_age and
                          entry.access_count < 3 and
                          current_tier == MemoryTier.LOCAL):
                        await self._demote_entry(key, entry)
                        optimizations['demoted'].append(key)
            
            self.logger.info(f"Memory tier optimization completed: optimizations={optimizations}")
            return optimizations
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        return {
            'tier_counts': {
                'local': len(self.local_cache),
                'persistent': len(self.persistent_store),
                'shared': len(self.shared_store),
                'archive': len(self.archive_store)
            },
            'total_entries': (
                len(self.local_cache) + len(self.persistent_store) + 
                len(self.shared_store) + len(self.archive_store)
            ),
            'access_patterns_tracked': len(self.access_patterns),
            'storage_path': str(self.storage_path),
            'configuration': {
                'max_local_entries': self.max_local_entries,
                'max_persistent_entries': self.max_persistent_entries,
                'promotion_threshold': self.tier_promotion_threshold,
                'demotion_age': self.tier_demotion_age
            }
        }
    
    # Private helper methods
    
    def _determine_optimal_tier(self, key: str, value: Any) -> MemoryTier:
        """Determine optimal storage tier using LLM analysis."""
        try:
            from strands import Agent
            from strands.models import BedrockModel
            import json
            
            # Create a simple agent for tier analysis
            model = BedrockModel(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                region_name="us-west-2"
            )
            
            agent = Agent(
                model=model,
                system_prompt="You are a memory management optimizer. Analyze data to determine optimal storage tiers."
            )
            
            # Prepare data characteristics for analysis
            value_size = len(str(value))
            value_type = type(value).__name__
            
            analysis_prompt = f"""
            Analyze this data entry to determine the optimal storage tier:
            
            Key: {key}
            Value type: {value_type}
            Value size: {value_size} characters
            
            Available tiers:
            - LOCAL: Fast access, temporary data, lost on restart
            - PERSISTENT: Disk storage, survives restarts, moderate access speed
            - SHARED: Cross-process sharing, network access, slower but distributed
            - ARCHIVE: Long-term storage, slowest access, for historical data
            
            Consider:
            - Access frequency (how often will this be retrieved?)
            - Persistence needs (should it survive restarts?)
            - Sharing requirements (multiple processes need access?)
            - Data importance (critical vs temporary?)
            - Size implications (large data should avoid fast tiers?)
            
            Respond with only one word: LOCAL, PERSISTENT, SHARED, or ARCHIVE
            """
            
            response = str(agent(analysis_prompt)).strip().upper()
            
            # Validate response and map to enum
            tier_mapping = {
                'LOCAL': MemoryTier.LOCAL,
                'PERSISTENT': MemoryTier.PERSISTENT,
                'SHARED': MemoryTier.SHARED,
                'ARCHIVE': MemoryTier.ARCHIVE
            }
            
            if response in tier_mapping:
                return tier_mapping[response]
            else:
                # If LLM response is invalid, use conservative fallback
                return self._fallback_tier_selection(key, value)
                
        except Exception as e:
            # If LLM analysis fails, use fallback logic
            return self._fallback_tier_selection(key, value)
    
    def _fallback_tier_selection(self, key: str, value: Any) -> MemoryTier:
        """Conservative fallback tier selection when LLM analysis fails."""
        value_size = len(str(value))
        
        # Conservative rules without heuristics - err on side of persistence
        if key.startswith('execution:') or key.startswith('result:'):
            return MemoryTier.PERSISTENT
        elif key.startswith('temp:') or key.startswith('work:'):
            return MemoryTier.LOCAL
        elif value_size > 50000:  # Very large data
            return MemoryTier.ARCHIVE
        else:
            return MemoryTier.PERSISTENT  # Default to persistent for safety
        return MemoryTier.LOCAL
        
        # TODO: Replace with Strands agent-based analysis for better accuracy
        # This will be implemented in Phase 2 of the Strands-first approach
    
    async def _find_entry_across_tiers(self, key: str) -> Optional[MemoryEntry]:
        """Find entry across all storage tiers."""
        # Search in order of access speed
        for store in [self.local_cache, self.persistent_store, self.shared_store, self.archive_store]:
            if key in store:
                return store[key]
        return None
    
    async def _consider_tier_promotion(self, key: str, entry: MemoryEntry) -> None:
        """Consider promoting entry to higher tier."""
        if entry.tier == MemoryTier.PERSISTENT and len(self.local_cache) < self.max_local_entries:
            # Promote to local cache
            self.local_cache[key] = entry
            entry.tier = MemoryTier.LOCAL
            del self.persistent_store[key]
    
    async def _manage_local_cache_size(self) -> None:
        """Manage local cache size by evicting old entries."""
        if len(self.local_cache) > self.max_local_entries:
            # Evict least recently used entries
            sorted_entries = sorted(
                self.local_cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Move oldest entries to persistent storage
            for key, entry in sorted_entries[:len(self.local_cache) - self.max_local_entries]:
                entry.tier = MemoryTier.PERSISTENT
                self.persistent_store[key] = entry
                del self.local_cache[key]
                await self._persist_to_disk(key, entry)
    
    async def _persist_to_disk(self, key: str, entry: MemoryEntry) -> None:
        """Persist entry to disk storage."""
        try:
            self.logger.debug(f"Persisting entry to disk: key='{key}', tier={entry.tier.value}")
            
            file_path = self.storage_path / f"{key}.json"
            self.logger.debug(f"Writing to file: {file_path}")
            
            # Convert value to JSON-serializable format
            serializable_value = self._make_serializable(entry.value)
            
            data_to_write = {
                'key': entry.key,
                'value': serializable_value,
                'tier': entry.tier.value,
                'timestamp': entry.timestamp,
                'access_count': entry.access_count,
                'last_accessed': entry.last_accessed,
                'metadata': entry.metadata
            }
            
            with open(file_path, 'w') as f:
                json.dump(data_to_write, f, indent=2)
            
            # Verify the file was written
            if file_path.exists():
                file_size = file_path.stat().st_size
                self.logger.info(f"Successfully persisted '{key}' to disk ({file_size} bytes)")
            else:
                self.logger.error(f"File was not created: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to persist entry '{key}' to disk: {str(e)}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, '__dict__'):
            # Convert objects with attributes to dictionaries
            result = {'__class__': obj.__class__.__name__}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):  # Skip private attributes
                    result[key] = self._make_serializable(value)
            return result
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For other types, convert to string representation
            return str(obj)
    
    async def _sync_to_shared_storage(self, key: str, entry: MemoryEntry) -> None:
        """Sync entry to shared storage (placeholder for distributed storage)."""
        # In a real implementation, this would sync to distributed storage
        # For now, just log the operation
        #TOTO: investigate distributed storage options like Redis, etc.
        self.logger.debug(f"[NOOP - mock operation] Synced to shared storage: key={key}")
    
    async def _archive_entry(self, key: str, entry: MemoryEntry) -> None:
        """Archive entry for long-term storage."""
        # Move to archive tier and compress if needed
        entry.tier = MemoryTier.ARCHIVE
        self.archive_store[key] = entry
        self.logger.debug(f"Archived entry: key={key}")
    
    async def _move_to_archive(self, key: str, entry: MemoryEntry) -> None:
        """Move entry from current tier to archive."""
        # Remove from current tier
        if key in self.local_cache:
            del self.local_cache[key]
        elif key in self.persistent_store:
            del self.persistent_store[key]
        elif key in self.shared_store:
            del self.shared_store[key]
        
        # Add to archive
        await self._archive_entry(key, entry)
    
    async def _promote_entry(self, key: str, entry: MemoryEntry) -> None:
        """Promote entry to higher tier."""
        if entry.tier == MemoryTier.PERSISTENT and len(self.local_cache) < self.max_local_entries:
            del self.persistent_store[key]
            entry.tier = MemoryTier.LOCAL
            self.local_cache[key] = entry
    
    async def _demote_entry(self, key: str, entry: MemoryEntry) -> None:
        """Demote entry to lower tier."""
        if entry.tier == MemoryTier.LOCAL:
            del self.local_cache[key]
            entry.tier = MemoryTier.PERSISTENT
            self.persistent_store[key] = entry
            await self._persist_to_disk(key, entry)
