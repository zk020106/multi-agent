"""
多智能体系统的异步工具。
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class TaskStatus(Enum):
    """异步任务状态。"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncTask:
    """表示一个异步任务。"""
    id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
    
    @property
    def duration(self) -> Optional[float]:
        """获取任务持续时间（秒）。"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class AsyncTaskManager:
    """管理异步任务执行，具有并发控制。"""
    
    def __init__(self, max_workers: int = 5, timeout: Optional[float] = None):
        self.max_workers = max_workers
        self.timeout = timeout
        self.tasks: Dict[str, AsyncTask] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def submit_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        提交任务进行异步执行。
        
        参数:
            task_id: 唯一任务标识符
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        返回:
            任务ID
        """
        task = AsyncTask(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs
        )
        self.tasks[task_id] = task
        return task_id
    
    async def execute_task(self, task_id: str) -> Any:
        """
        异步执行单个任务。
        
        参数:
            task_id: 任务标识符
            
        返回:
            任务结果
        """
        if task_id not in self.tasks:
            raise ValueError(f"未找到任务 {task_id}")
        
        task = self.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        
        try:
            # 在线程池中运行
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: task.func(*task.args, **task.kwargs)
            )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = time.time()
            
            return result
            
        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            task.end_time = time.time()
            raise
    
    async def execute_tasks(
        self,
        task_ids: List[str],
        return_when: str = "ALL_COMPLETED"
    ) -> Dict[str, Any]:
        """
        并发执行多个任务。
        
        参数:
            task_ids: 任务标识符列表
            return_when: 何时返回 ("FIRST_COMPLETED", "FIRST_EXCEPTION", "ALL_COMPLETED")
            
        返回:
            将任务ID映射到结果的字典
        """
        # 验证任务ID
        for task_id in task_ids:
            if task_id not in self.tasks:
                raise ValueError(f"未找到任务 {task_id}")
        
        # 为所有任务创建协程
        coroutines = [self.execute_task(task_id) for task_id in task_ids]
        
        # 如果指定了超时，则执行超时
        if self.timeout:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*coroutines, return_exceptions=True),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                # 将剩余任务标记为已取消
                for task_id in task_ids:
                    if self.tasks[task_id].status == TaskStatus.RUNNING:
                        self.tasks[task_id].status = TaskStatus.CANCELLED
                raise
        else:
            results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # 构建结果字典
        result_dict = {}
        for task_id, result in zip(task_ids, results):
            if isinstance(result, Exception):
                result_dict[task_id] = None
            else:
                result_dict[task_id] = result
        
        return result_dict
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        """获取任务状态。"""
        if task_id not in self.tasks:
            raise ValueError(f"未找到任务 {task_id}")
        return self.tasks[task_id].status
    
    def get_task_result(self, task_id: str) -> Any:
        """获取任务结果。"""
        if task_id not in self.tasks:
            raise ValueError(f"未找到任务 {task_id}")
        return self.tasks[task_id].result
    
    def get_task_error(self, task_id: str) -> Optional[Exception]:
        """获取任务错误。"""
        if task_id not in self.tasks:
            raise ValueError(f"未找到任务 {task_id}")
        return self.tasks[task_id].error
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务。"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        
        return False
    
    def get_all_tasks(self) -> Dict[str, AsyncTask]:
        """获取所有任务。"""
        return self.tasks.copy()
    
    def clear_completed_tasks(self) -> int:
        """清除已完成的任务并返回计数。"""
        to_remove = [
            task_id for task_id, task in self.tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        return len(to_remove)
    
    def shutdown(self, wait: bool = True):
        """关闭任务管理器。"""
        self.executor.shutdown(wait=wait)


async def run_async_tasks(
    tasks: List[Tuple[Callable, tuple, dict]],
    max_workers: int = 5,
    timeout: Optional[float] = None,
    return_exceptions: bool = False
) -> List[Any]:
    """
    并发运行多个异步任务。
    
    参数:
        tasks: (函数, 参数, 关键字参数) 元组列表
        max_workers: 最大并发工作器数量
        timeout: 超时时间（秒）
        return_exceptions: 是否返回异常而不是抛出
        
    返回:
        结果列表
    """
    manager = AsyncTaskManager(max_workers=max_workers, timeout=timeout)
    
    try:
        # 提交所有任务
        task_ids = []
        for i, (func, args, kwargs) in enumerate(tasks):
            task_id = f"task_{i}"
            manager.submit_task(task_id, func, *args, **kwargs)
            task_ids.append(task_id)
        
        # 执行所有任务
        results = await manager.execute_tasks(task_ids)
        
        # 按顺序返回结果
        return [results[task_id] for task_id in task_ids]
        
    finally:
        manager.shutdown()


async def run_with_retry(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    *args,
    **kwargs
) -> Any:
    """
    使用重试逻辑运行函数。
    
    参数:
        func: 要执行的函数
        max_retries: 最大重试次数
        delay: 重试之间的初始延迟
        backoff_factor: 延迟的回退因子
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    返回:
        函数结果
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                await asyncio.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                break
    
    raise last_exception


async def run_with_timeout(
    func: Callable,
    timeout: float,
    *args,
    **kwargs
) -> Any:
    """
    使用超时运行函数。
    
    参数:
        func: 要执行的函数
        timeout: 超时时间（秒）
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    返回:
        函数结果
    """
    if asyncio.iscoroutinefunction(func):
        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
    else:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: func(*args, **kwargs)),
            timeout=timeout
        )


class RateLimiter:
    """用于控制请求频率的速率限制器。"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """获取发出请求的权限。"""
        async with self._lock:
            now = time.time()
            
            # 移除时间窗口外的旧请求
            self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            # 检查是否可以发出请求
            if len(self.requests) >= self.max_requests:
                # 计算等待时间
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # 等待后再次清理
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            # 记录此请求
            self.requests.append(now)
