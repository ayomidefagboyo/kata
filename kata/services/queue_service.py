"""
Queue Service for Token Discovery

Manages background processing queues for token discovery,
enabling scalable processing for 10K+ users.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)

class QueuePriority(Enum):
    """Queue priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class QueueStatus(Enum):
    """Queue item status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class QueueItem:
    """Queue item for token discovery processing."""
    id: str
    task_type: str
    priority: QueuePriority
    status: QueueStatus
    data: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class QueueService:
    """High-performance queue service for token discovery."""
    
    def __init__(self):
        self.queues = {
            QueuePriority.LOW: asyncio.Queue(maxsize=1000),
            QueuePriority.NORMAL: asyncio.Queue(maxsize=2000),
            QueuePriority.HIGH: asyncio.Queue(maxsize=1000),
            QueuePriority.URGENT: asyncio.Queue(maxsize=500)
        }
        
        self.processing_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_failed': 0,
            'total_queued': 0,
            'current_queue_size': 0,
            'processing_count': 0
        }
        
        # Background workers
        self.workers = []
        self.is_running = False
        
        logger.info("🚀 Queue service initialized")
    
    async def start_workers(self, worker_count: int = 5):
        """Start background worker processes."""
        if self.is_running:
            logger.warning("Workers already running")
            return
        
        self.is_running = True
        
        # Start workers for each priority level
        for priority in QueuePriority:
            for i in range(worker_count):
                worker = asyncio.create_task(
                    self._worker_loop(priority, f"worker-{priority.name}-{i}")
                )
                self.workers.append(worker)
        
        logger.info(f"✅ Started {worker_count * len(QueuePriority)} queue workers")
    
    async def stop_workers(self):
        """Stop all background workers."""
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("⏹️ Stopped all queue workers")
    
    async def enqueue_task(
        self, 
        task_type: str, 
        data: Dict[str, Any], 
        priority: QueuePriority = QueuePriority.NORMAL
    ) -> str:
        """Add a task to the queue."""
        try:
            # Create queue item
            item = QueueItem(
                id=str(uuid.uuid4()),
                task_type=task_type,
                priority=priority,
                status=QueueStatus.PENDING,
                data=data,
                created_at=datetime.now()
            )
            
            # Add to appropriate queue
            await self.queues[priority].put(item)
            
            # Update statistics
            self.stats['total_queued'] += 1
            self.stats['current_queue_size'] += 1
            
            logger.info(f"📥 Enqueued {task_type} task (priority: {priority.name}, id: {item.id})")
            return item.id
            
        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            raise
    
    async def get_queue_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check processing tasks
        if task_id in self.processing_tasks:
            item = self.processing_tasks[task_id]
            return {
                'id': item.id,
                'status': item.status.value,
                'progress': 'processing',
                'created_at': item.created_at.isoformat(),
                'started_at': item.started_at.isoformat() if item.started_at else None
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            item = self.completed_tasks[task_id]
            return {
                'id': item.id,
                'status': item.status.value,
                'progress': 'completed',
                'created_at': item.created_at.isoformat(),
                'completed_at': item.completed_at.isoformat() if item.completed_at else None,
                'result': item.data.get('result')
            }
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            item = self.failed_tasks[task_id]
            return {
                'id': item.id,
                'status': item.status.value,
                'progress': 'failed',
                'created_at': item.created_at.isoformat(),
                'error': item.error,
                'retry_count': item.retry_count
            }
        
        return None
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        queue_sizes = {}
        for priority in QueuePriority:
            queue_sizes[priority.name] = self.queues[priority].qsize()
        
        return {
            'queues': queue_sizes,
            'processing': len(self.processing_tasks),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks),
            'total_processed': self.stats['total_processed'],
            'total_failed': self.stats['total_failed'],
            'total_queued': self.stats['total_queued'],
            'current_queue_size': self.stats['current_queue_size'],
            'worker_count': len(self.workers),
            'is_running': self.is_running
        }
    
    async def _worker_loop(self, priority: QueuePriority, worker_name: str):
        """Background worker loop for processing queue items."""
        logger.info(f"🔄 Worker {worker_name} started for {priority.name} priority")
        
        while self.is_running:
            try:
                # Get item from queue (with timeout)
                try:
                    item = await asyncio.wait_for(
                        self.queues[priority].get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the item
                await self._process_item(item, worker_name)
                
                # Mark as done
                self.queues[priority].task_done()
                
            except asyncio.CancelledError:
                logger.info(f"🛑 Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.info(f"⏹️ Worker {worker_name} stopped")
    
    async def _process_item(self, item: QueueItem, worker_name: str):
        """Process a single queue item."""
        try:
            # Update status
            item.status = QueueStatus.PROCESSING
            item.started_at = datetime.now()
            self.processing_tasks[item.id] = item
            
            logger.info(f"🔄 Processing {item.task_type} task (id: {item.id}) with {worker_name}")
            
            # Process based on task type
            result = await self._execute_task(item)
            
            # Mark as completed
            item.status = QueueStatus.COMPLETED
            item.completed_at = datetime.now()
            item.data['result'] = result
            
            # Move to completed
            del self.processing_tasks[item.id]
            self.completed_tasks[item.id] = item
            
            # Update statistics
            self.stats['total_processed'] += 1
            self.stats['current_queue_size'] -= 1
            
            logger.info(f"✅ Completed {item.task_type} task (id: {item.id})")
            
        except Exception as e:
            # Handle failure
            await self._handle_task_failure(item, e, worker_name)
    
    async def _execute_task(self, item: QueueItem) -> Any:
        """Execute the actual task based on type."""
        task_type = item.task_type
        
        if task_type == "token_discovery":
            return await self._execute_token_discovery(item.data)
        elif task_type == "market_data_update":
            return await self._execute_market_data_update(item.data)
        elif task_type == "token_analysis":
            return await self._execute_token_analysis(item.data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _execute_token_discovery(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute token discovery task."""
        # This would integrate with your existing token discovery service
        # For now, return mock result
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            'tokens_discovered': 25,
            'quality_tokens': 18,
            'processing_time': 2.0,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_market_data_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute market data update task."""
        await asyncio.sleep(1)  # Simulate processing time
        
        return {
            'tokens_updated': 15,
            'processing_time': 1.0,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_token_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute token analysis task."""
        await asyncio.sleep(3)  # Simulate processing time
        
        return {
            'token_analyzed': data.get('token_id'),
            'analysis_score': 0.75,
            'processing_time': 3.0,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _handle_task_failure(self, item: QueueItem, error: Exception, worker_name: str):
        """Handle task failure with retry logic."""
        item.retry_count += 1
        item.error = str(error)
        
        if item.retry_count < item.max_retries:
            # Retry the task
            item.status = QueueStatus.PENDING
            item.started_at = None
            
            # Re-queue with lower priority
            new_priority = max(QueuePriority.LOW, QueuePriority(item.priority.value - 1))
            await self.queues[new_priority].put(item)
            
            logger.warning(f"🔄 Retrying {item.task_type} task (id: {item.id}, attempt: {item.retry_count})")
            
        else:
            # Mark as failed
            item.status = QueueStatus.FAILED
            item.completed_at = datetime.now()
            
            # Move to failed
            del self.processing_tasks[item.id]
            self.failed_tasks[item.id] = item
            
            # Update statistics
            self.stats['total_failed'] += 1
            self.stats['current_queue_size'] -= 1
            
            logger.error(f"❌ Failed {item.task_type} task (id: {item.id}) after {item.retry_count} attempts: {error}")

# Global queue service instance
queue_service = QueueService()

async def get_queue_service() -> QueueService:
    """Get queue service dependency."""
    return queue_service
