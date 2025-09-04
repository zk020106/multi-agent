"""
顺序执行协调器
按顺序将任务分配给智能体执行
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from agents import BaseAgent
from schema import Task, Result, ResultStatus
from .base_coordinator import BaseCoordinator


class SequentialCoordinator(BaseCoordinator):
    """
    顺序执行协调器
    
    特点：
    - 按顺序执行任务
    - 一个任务完成后才开始下一个
    - 适合有依赖关系的任务序列
    - 资源使用效率高
    """
    
    def __init__(
        self,
        coordinator_id: str,
        name: str = "顺序执行协调器",
        description: str = "按顺序执行任务的协调器",
        max_concurrent_tasks: int = 1
    ):
        """
        初始化顺序执行协调器
        
        Args:
            coordinator_id: 协调器ID
            name: 协调器名称
            description: 协调器描述
            max_concurrent_tasks: 最大并发任务数
        """
        super().__init__(coordinator_id, name, description)
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue: List[Task] = []
        self.execution_lock = asyncio.Lock()
        
        self.logger.info(f"顺序执行协调器初始化完成，最大并发任务数: {max_concurrent_tasks}")
    
    async def coordinate(self, task: Task) -> Result:
        """
        协调执行任务（顺序模式）
        
        Args:
            task: 要执行的任务
            
        Returns:
            执行结果
        """
        async with self.execution_lock:
            return await self._execute_task_sequential(task)
    
    async def _execute_task_sequential(self, task: Task) -> Result:
        """
        顺序执行任务
        
        Args:
            task: 要执行的任务
            
        Returns:
            执行结果
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"开始顺序执行任务: {task.title}")
            
            # 检查任务依赖
            if not self._check_dependencies(task):
                return Result(
                    id=str(uuid.uuid4()),
                    task_id=task.id,
                    agent_id=self.coordinator_id,
                    status=ResultStatus.ERROR,
                    error_message="任务依赖未满足"
                )
            
            # 选择智能体
            agent = self._select_agent_for_task(task)
            if not agent:
                return Result(
                    id=str(uuid.uuid4()),
                    task_id=task.id,
                    agent_id=self.coordinator_id,
                    status=ResultStatus.ERROR,
                    error_message="没有可用的智能体"
                )
            
            # 分配任务
            self.assign_task_to_agent(task, agent)
            
            # 执行任务
            result = await self._execute_with_agent(task, agent)
            
            # 完成任务
            self.complete_task(task, result)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            self.logger.info(f"任务 {task.title} 顺序执行完成，耗时: {execution_time:.2f}秒")
            
            return result
            
        except Exception as e:
            error_result = Result(
                id=str(uuid.uuid4()),
                task_id=task.id,
                agent_id=self.coordinator_id,
                status=ResultStatus.ERROR,
                error_message=str(e),
                execution_time=(time.time() - start_time)
            )
            
            self.complete_task(task, error_result)
            self.logger.error(f"任务 {task.title} 执行失败: {str(e)}")
            
            return error_result
    
    async def coordinate_multiple(self, tasks: List[Task]) -> List[Result]:
        """
        协调执行多个任务（顺序模式）
        
        Args:
            tasks: 任务列表
            
        Returns:
            执行结果列表
        """
        results = []
        
        for task in tasks:
            result = await self.coordinate(task)
            results.append(result)
            
            # 如果任务失败且设置了停止条件，可以在这里处理
            if result.status == ResultStatus.ERROR:
                self.logger.warning(f"任务 {task.title} 失败，继续执行下一个任务")
        
        return results
    
    def _check_dependencies(self, task: Task) -> bool:
        """
        检查任务依赖是否满足
        
        Args:
            task: 任务对象
            
        Returns:
            依赖是否满足
        """
        if not task.dependencies:
            return True
        
        # 检查所有依赖任务是否已完成
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                self.logger.warning(f"任务 {task.title} 的依赖 {dep_id} 未完成")
                return False
        
        return True
    
    def _select_agent_for_task(self, task: Task) -> Optional[BaseAgent]:
        """
        为任务选择合适的智能体
        
        Args:
            task: 任务对象
            
        Returns:
            选中的智能体或None
        """
        available_agents = self.get_available_agents()
        
        if not available_agents:
            return None
        
        # 简单的选择策略：选择第一个可用的智能体
        # 可以根据任务类型、智能体能力等进行更复杂的选择
        selected_agent = available_agents[0]
        
        # 根据任务优先级调整选择
        if task.priority.value == "urgent":
            # 优先选择处理速度快的智能体
            selected_agent = min(available_agents, key=lambda a: a.total_execution_time)
        elif task.priority.value == "high":
            # 选择成功率高的智能体
            selected_agent = max(available_agents, key=lambda a: a.tasks_completed / max(1, a.tasks_completed + a.tasks_failed))
        
        return selected_agent

    async def _execute_with_agent(self, task: Task, agent: BaseAgent) -> Result:
        """
        使用智能体执行任务

        Args:
            task: 任务对象
            agent: 智能体实例

        Returns:
            执行结果
        """
        try:
            # 执行任务
            result = agent.act(task)

            # 等待执行完成（如果是异步的）
            if asyncio.iscoroutine(result):
                result = await result

            return result

        except Exception as e:
            return Result(
                id=str(uuid.uuid4()),
                task_id=task.id,
                agent_id=agent.agent_id,
                status=ResultStatus.ERROR,
                error_message=f"智能体执行失败: {str(e)}"
            )

    def add_task_to_queue(self, task: Task) -> None:
        """
        添加任务到队列
        
        Args:
            task: 任务对象
        """
        self.task_queue.append(task)
        self.logger.info(f"任务 {task.title} 已添加到队列，队列长度: {len(self.task_queue)}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        获取队列状态
        
        Returns:
            队列状态信息
        """
        return {
            "queue_length": len(self.task_queue),
            "queued_tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "priority": task.priority.value,
                    "status": task.status.value
                }
                for task in self.task_queue
            ]
        }
    
    def clear_queue(self) -> int:
        """
        清空任务队列
        
        Returns:
            清空的任务数量
        """
        count = len(self.task_queue)
        self.task_queue.clear()
        self.logger.info(f"清空任务队列，共 {count} 个任务")
        return count
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取协调器状态
        
        Returns:
            状态信息字典
        """
        base_status = super().get_status()
        base_status.update({
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "queue_length": len(self.task_queue),
            "execution_mode": "sequential"
        })
        return base_status
    
    def __str__(self) -> str:
        return f"顺序执行协调器[{self.coordinator_id}]: {self.name} - 队列: {len(self.task_queue)}个任务"
