"""
性能优化模块
实现智能体池管理、任务队列优化和并发控制
"""

import asyncio
import time
import weakref
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
from threading import Lock
import heapq


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TaskItem:
    """任务项"""
    task_id: str
    priority: TaskPriority
    created_at: float
    estimated_duration: float
    task_data: Any
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        # 优先级高的先执行，时间早的先执行
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at


@dataclass
class AgentPoolStats:
    """智能体池统计信息"""
    total_agents: int = 0
    active_agents: int = 0
    idle_agents: int = 0
    busy_agents: int = 0
    avg_task_duration: float = 0.0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    success_rate: float = 0.0
    last_activity: float = 0.0


class TaskQueue:
    """优化的任务队列"""
    
    def __init__(self, max_size: int = 1000):
        """
        初始化任务队列
        
        Args:
            max_size: 最大队列大小
        """
        self.max_size = max_size
        self._queue: List[TaskItem] = []
        self._lock = Lock()
        self._stats = {
            "total_added": 0,
            "total_processed": 0,
            "total_dropped": 0,
            "avg_wait_time": 0.0
        }
    
    def add_task(
        self, 
        task_id: str, 
        priority: TaskPriority, 
        task_data: Any,
        estimated_duration: float = 30.0,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        添加任务到队列
        
        Args:
            task_id: 任务ID
            priority: 任务优先级
            task_data: 任务数据
            estimated_duration: 预估执行时间
            callback: 完成回调
            
        Returns:
            是否成功添加
        """
        with self._lock:
            if len(self._queue) >= self.max_size:
                # 队列满了，尝试移除低优先级任务
                if not self._remove_low_priority_task():
                    self._stats["total_dropped"] += 1
                    return False
            
            task_item = TaskItem(
                task_id=task_id,
                priority=priority,
                created_at=time.time(),
                estimated_duration=estimated_duration,
                task_data=task_data,
                callback=callback
            )
            
            heapq.heappush(self._queue, task_item)
            self._stats["total_added"] += 1
            return True
    
    def get_next_task(self) -> Optional[TaskItem]:
        """
        获取下一个任务
        
        Returns:
            任务项或None
        """
        with self._lock:
            if not self._queue:
                return None
            
            task_item = heapq.heappop(self._queue)
            self._stats["total_processed"] += 1
            
            # 更新平均等待时间
            wait_time = time.time() - task_item.created_at
            current_avg = self._stats["avg_wait_time"]
            processed_count = self._stats["total_processed"]
            self._stats["avg_wait_time"] = (current_avg * (processed_count - 1) + wait_time) / processed_count
            
            return task_item
    
    def _remove_low_priority_task(self) -> bool:
        """移除低优先级任务"""
        if not self._queue:
            return False
        
        # 找到优先级最低的任务
        lowest_priority_task = None
        lowest_priority_idx = -1
        
        for i, task in enumerate(self._queue):
            if (lowest_priority_task is None or 
                task.priority.value < lowest_priority_task.priority.value):
                lowest_priority_task = task
                lowest_priority_idx = i
        
        if lowest_priority_idx >= 0:
            del self._queue[lowest_priority_idx]
            heapq.heapify(self._queue)  # 重新堆化
            return True
        
        return False
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        with self._lock:
            return len(self._queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        with self._lock:
            return self._stats.copy()
    
    def clear(self) -> int:
        """清空队列"""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count


class AgentPool:
    """智能体池管理器"""
    
    def __init__(self, max_agents: int = 10, min_agents: int = 2):
        """
        初始化智能体池
        
        Args:
            max_agents: 最大智能体数量
            min_agents: 最小智能体数量
        """
        self.max_agents = max_agents
        self.min_agents = min_agents
        self.agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, str] = {}  # idle, busy, error
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._stats = AgentPoolStats()
    
    def add_agent(self, agent_id: str, agent: Any) -> bool:
        """
        添加智能体到池中
        
        Args:
            agent_id: 智能体ID
            agent: 智能体实例
            
        Returns:
            是否成功添加
        """
        with self._lock:
            if len(self.agents) >= self.max_agents:
                return False
            
            self.agents[agent_id] = agent
            self.agent_status[agent_id] = "idle"
            self.agent_stats[agent_id] = {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "total_execution_time": 0.0,
                "last_activity": time.time()
            }
            
            self._update_stats()
            return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        从池中移除智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if agent_id not in self.agents:
                return False
            
            if len(self.agents) <= self.min_agents:
                return False
            
            del self.agents[agent_id]
            del self.agent_status[agent_id]
            del self.agent_stats[agent_id]
            
            self._update_stats()
            return True
    
    def get_idle_agent(self) -> Optional[Any]:
        """获取空闲的智能体"""
        with self._lock:
            for agent_id, status in self.agent_status.items():
                if status == "idle":
                    return self.agents[agent_id]
            return None
    
    def get_best_agent(self, task_requirements: Dict[str, Any] = None) -> Optional[Any]:
        """
        获取最适合的智能体
        
        Args:
            task_requirements: 任务需求
            
        Returns:
            最适合的智能体
        """
        with self._lock:
            idle_agents = [
                (agent_id, self.agents[agent_id]) 
                for agent_id, status in self.agent_status.items() 
                if status == "idle"
            ]
            
            if not idle_agents:
                return None
            
            # 简单的选择策略：选择成功率最高的智能体
            best_agent = None
            best_score = -1
            
            for agent_id, agent in idle_agents:
                stats = self.agent_stats[agent_id]
                total_tasks = stats["tasks_completed"] + stats["tasks_failed"]
                
                if total_tasks > 0:
                    success_rate = stats["tasks_completed"] / total_tasks
                    avg_time = stats["total_execution_time"] / max(stats["tasks_completed"], 1)
                    # 综合评分：成功率 + 速度
                    score = success_rate - (avg_time / 100.0)  # 时间越短分数越高
                else:
                    score = 0.5  # 新智能体的默认分数
                
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            return best_agent
    
    def set_agent_busy(self, agent_id: str) -> bool:
        """设置智能体为忙碌状态"""
        with self._lock:
            if agent_id in self.agent_status:
                self.agent_status[agent_id] = "busy"
                self._update_stats()
                return True
            return False
    
    def set_agent_idle(self, agent_id: str) -> bool:
        """设置智能体为空闲状态"""
        with self._lock:
            if agent_id in self.agent_status:
                self.agent_status[agent_id] = "idle"
                self.agent_stats[agent_id]["last_activity"] = time.time()
                self._update_stats()
                return True
            return False
    
    def update_agent_stats(
        self, 
        agent_id: str, 
        success: bool, 
        execution_time: float
    ) -> None:
        """更新智能体统计信息"""
        with self._lock:
            if agent_id not in self.agent_stats:
                return
            
            stats = self.agent_stats[agent_id]
            if success:
                stats["tasks_completed"] += 1
            else:
                stats["tasks_failed"] += 1
            
            stats["total_execution_time"] += execution_time
            stats["last_activity"] = time.time()
            
            self._update_stats()
    
    def _update_stats(self) -> None:
        """更新池统计信息"""
        self._stats.total_agents = len(self.agents)
        self._stats.active_agents = len([s for s in self.agent_status.values() if s != "error"])
        self._stats.idle_agents = len([s for s in self.agent_status.values() if s == "idle"])
        self._stats.busy_agents = len([s for s in self.agent_status.values() if s == "busy"])
        
        # 计算平均执行时间和成功率
        total_completed = sum(stats["tasks_completed"] for stats in self.agent_stats.values())
        total_failed = sum(stats["tasks_failed"] for stats in self.agent_stats.values())
        total_time = sum(stats["total_execution_time"] for stats in self.agent_stats.values())
        
        self._stats.total_tasks_completed = total_completed
        self._stats.total_tasks_failed = total_failed
        
        if total_completed > 0:
            self._stats.avg_task_duration = total_time / total_completed
            self._stats.success_rate = total_completed / (total_completed + total_failed)
        
        self._stats.last_activity = time.time()
    
    def get_stats(self) -> AgentPoolStats:
        """获取池统计信息"""
        with self._lock:
            return AgentPoolStats(
                total_agents=self._stats.total_agents,
                active_agents=self._stats.active_agents,
                idle_agents=self._stats.idle_agents,
                busy_agents=self._stats.busy_agents,
                avg_task_duration=self._stats.avg_task_duration,
                total_tasks_completed=self._stats.total_tasks_completed,
                total_tasks_failed=self._stats.total_tasks_failed,
                success_rate=self._stats.success_rate,
                last_activity=self._stats.last_activity
            )
    
    def cleanup_inactive_agents(self, inactive_threshold: float = 300.0) -> int:
        """
        清理不活跃的智能体
        
        Args:
            inactive_threshold: 不活跃阈值（秒）
            
        Returns:
            清理的智能体数量
        """
        with self._lock:
            current_time = time.time()
            inactive_agents = []
            
            for agent_id, stats in self.agent_stats.items():
                if (current_time - stats["last_activity"] > inactive_threshold and 
                    self.agent_status[agent_id] == "idle" and
                    len(self.agents) > self.min_agents):
                    inactive_agents.append(agent_id)
            
            for agent_id in inactive_agents:
                del self.agents[agent_id]
                del self.agent_status[agent_id]
                del self.agent_stats[agent_id]
            
            if inactive_agents:
                self._update_stats()
            
            return len(inactive_agents)


class ConcurrencyController:
    """并发控制器"""
    
    def __init__(self, max_concurrent_tasks: int = 5):
        """
        初始化并发控制器
        
        Args:
            max_concurrent_tasks: 最大并发任务数
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks: Set[str] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._lock = asyncio.Lock()
        self._stats = {
            "total_tasks_submitted": 0,
            "total_tasks_completed": 0,
            "total_tasks_rejected": 0,
            "avg_concurrent_tasks": 0.0,
            "max_concurrent_reached": 0
        }
    
    async def acquire_slot(self, task_id: str) -> bool:
        """
        获取执行槽位
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否成功获取槽位
        """
        async with self._lock:
            if len(self.current_tasks) >= self.max_concurrent_tasks:
                self._stats["total_tasks_rejected"] += 1
                return False
            
            self.current_tasks.add(task_id)
            self._stats["total_tasks_submitted"] += 1
            self._stats["max_concurrent_reached"] = max(
                self._stats["max_concurrent_reached"], 
                len(self.current_tasks)
            )
            
            # 更新平均并发数
            total_submitted = self._stats["total_tasks_submitted"]
            current_avg = self._stats["avg_concurrent_tasks"]
            self._stats["avg_concurrent_tasks"] = (
                (current_avg * (total_submitted - 1) + len(self.current_tasks)) / total_submitted
            )
            
            return True
    
    async def release_slot(self, task_id: str) -> None:
        """
        释放执行槽位
        
        Args:
            task_id: 任务ID
        """
        async with self._lock:
            if task_id in self.current_tasks:
                self.current_tasks.remove(task_id)
                self._stats["total_tasks_completed"] += 1
    
    def get_current_concurrency(self) -> int:
        """获取当前并发数"""
        return len(self.current_tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()
    
    def set_max_concurrent_tasks(self, max_tasks: int) -> None:
        """设置最大并发任务数"""
        self.max_concurrent_tasks = max_tasks
        # 更新信号量
        self._semaphore = asyncio.Semaphore(max_tasks)


class PerformanceOptimizer:
    """性能优化器主类"""
    
    def __init__(
        self,
        max_agents: int = 10,
        min_agents: int = 2,
        max_queue_size: int = 1000,
        max_concurrent_tasks: int = 5
    ):
        """
        初始化性能优化器
        
        Args:
            max_agents: 最大智能体数量
            min_agents: 最小智能体数量
            max_queue_size: 最大队列大小
            max_concurrent_tasks: 最大并发任务数
        """
        self.agent_pool = AgentPool(max_agents, min_agents)
        self.task_queue = TaskQueue(max_queue_size)
        self.concurrency_controller = ConcurrencyController(max_concurrent_tasks)
        
        # 性能监控
        self._performance_metrics = {
            "start_time": time.time(),
            "total_throughput": 0.0,
            "avg_response_time": 0.0,
            "peak_memory_usage": 0.0
        }
    
    async def submit_task(
        self,
        task_id: str,
        priority: TaskPriority,
        task_data: Any,
        estimated_duration: float = 30.0,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        提交任务
        
        Args:
            task_id: 任务ID
            priority: 任务优先级
            task_data: 任务数据
            estimated_duration: 预估执行时间
            callback: 完成回调
            
        Returns:
            是否成功提交
        """
        # 尝试获取执行槽位
        if not await self.concurrency_controller.acquire_slot(task_id):
            return False
        
        # 添加到任务队列
        if not self.task_queue.add_task(task_id, priority, task_data, estimated_duration, callback):
            await self.concurrency_controller.release_slot(task_id)
            return False
        
        return True
    
    async def process_next_task(self) -> Optional[Any]:
        """处理下一个任务"""
        # 获取下一个任务
        task_item = self.task_queue.get_next_task()
        if not task_item:
            return None
        
        # 获取最佳智能体
        agent = self.agent_pool.get_best_agent()
        if not agent:
            # 没有可用智能体，将任务放回队列
            self.task_queue.add_task(
                task_item.task_id,
                task_item.priority,
                task_item.task_data,
                task_item.estimated_duration,
                task_item.callback
            )
            return None
        
        # 设置智能体为忙碌状态
        agent_id = getattr(agent, 'agent_id', task_item.task_id)
        self.agent_pool.set_agent_busy(agent_id)
        
        try:
            # 执行任务
            start_time = time.time()
            result = await self._execute_task(agent, task_item.task_data)
            execution_time = time.time() - start_time
            
            # 更新统计信息
            self.agent_pool.update_agent_stats(agent_id, True, execution_time)
            
            # 调用回调
            if task_item.callback:
                await task_item.callback(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.agent_pool.update_agent_stats(agent_id, False, execution_time)
            raise e
            
        finally:
            # 释放资源
            self.agent_pool.set_agent_idle(agent_id)
            await self.concurrency_controller.release_slot(task_item.task_id)
    
    async def _execute_task(self, agent: Any, task_data: Any) -> Any:
        """执行任务"""
        # 这里应该调用智能体的执行方法
        if hasattr(agent, 'act'):
            result = agent.act(task_data)
            if asyncio.iscoroutine(result):
                return await result
            return result
        else:
            raise ValueError("智能体没有act方法")
    
    def add_agent(self, agent_id: str, agent: Any) -> bool:
        """添加智能体到池中"""
        return self.agent_pool.add_agent(agent_id, agent)
    
    def remove_agent(self, agent_id: str) -> bool:
        """从池中移除智能体"""
        return self.agent_pool.remove_agent(agent_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        current_time = time.time()
        uptime = current_time - self._performance_metrics["start_time"]
        
        queue_stats = self.task_queue.get_stats()
        pool_stats = self.agent_pool.get_stats()
        concurrency_stats = self.concurrency_controller.get_stats()
        
        return {
            "uptime": uptime,
            "queue_size": self.task_queue.get_queue_size(),
            "current_concurrency": self.concurrency_controller.get_current_concurrency(),
            "agent_pool_stats": {
                "total_agents": pool_stats.total_agents,
                "idle_agents": pool_stats.idle_agents,
                "busy_agents": pool_stats.busy_agents,
                "success_rate": pool_stats.success_rate,
                "avg_task_duration": pool_stats.avg_task_duration
            },
            "queue_stats": queue_stats,
            "concurrency_stats": concurrency_stats,
            "throughput": pool_stats.total_tasks_completed / max(uptime, 1),
            "efficiency": pool_stats.success_rate * (pool_stats.idle_agents / max(pool_stats.total_agents, 1))
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """执行性能优化"""
        optimizations = []
        
        # 清理不活跃的智能体
        cleaned_agents = self.agent_pool.cleanup_inactive_agents()
        if cleaned_agents > 0:
            optimizations.append(f"清理了 {cleaned_agents} 个不活跃的智能体")
        
        # 调整并发数（基于队列大小和智能体数量）
        queue_size = self.task_queue.get_queue_size()
        idle_agents = self.agent_pool.get_stats().idle_agents
        
        if queue_size > idle_agents * 2:
            # 队列积压严重，增加并发数
            new_concurrency = min(self.concurrency_controller.max_concurrent_tasks + 1, 10)
            self.concurrency_controller.set_max_concurrent_tasks(new_concurrency)
            optimizations.append(f"增加并发数到 {new_concurrency}")
        elif queue_size < idle_agents and self.concurrency_controller.max_concurrent_tasks > 2:
            # 队列空闲，减少并发数
            new_concurrency = max(self.concurrency_controller.max_concurrent_tasks - 1, 2)
            self.concurrency_controller.set_max_concurrent_tasks(new_concurrency)
            optimizations.append(f"减少并发数到 {new_concurrency}")
        
        return {
            "optimizations_applied": optimizations,
            "current_metrics": self.get_performance_metrics()
        }
