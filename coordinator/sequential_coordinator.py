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
from .smart_agent_selector import SmartAgentSelector, AgentCapability, SelectionStrategy
from utils.error_handler import ErrorHandler, ExponentialBackoffStrategy, handle_errors


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
        max_concurrent_tasks: int = 1,
        selection_strategy: SelectionStrategy = SelectionStrategy.HYBRID
    ):
        """
        初始化顺序执行协调器
        
        Args:
            coordinator_id: 协调器ID
            name: 协调器名称
            description: 协调器描述
            max_concurrent_tasks: 最大并发任务数
            selection_strategy: 智能体选择策略
        """
        super().__init__(coordinator_id, name, description)
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue: List[Task] = []
        self.execution_lock = asyncio.Lock()
        
        # 初始化智能选择器
        self.agent_selector = SmartAgentSelector(strategy=selection_strategy)
        
        # 初始化错误处理器
        self.error_handler = ErrorHandler(
            retry_strategy=ExponentialBackoffStrategy(),
            enable_logging=True
        )
        
        self.logger.info(f"顺序执行协调器初始化完成，最大并发任务数: {max_concurrent_tasks}, 选择策略: {selection_strategy.value}")
    
    def add_agent(self, agent: BaseAgent, role: str = "general") -> None:
        """
        添加智能体到协调器
        
        Args:
            agent: 智能体实例
            role: 智能体角色
        """
        super().add_agent(agent, role)
        
        # 为智能体注册能力信息
        capability = AgentCapability(
            agent_id=agent.agent_id,
            capabilities=self._extract_agent_capabilities(agent),
            specializations=[role],
            performance_score=0.8,  # 初始性能分数
            success_rate=0.9,  # 初始成功率
            avg_execution_time=30.0,  # 初始平均执行时间
            current_load=0.0,  # 初始负载
            last_activity=time.time()
        )
        self.agent_selector.register_agent_capability(capability)
    
    def _extract_agent_capabilities(self, agent: BaseAgent) -> List[str]:
        """从智能体提取能力信息"""
        capabilities = ["通用"]  # 默认能力
        
        # 基于智能体类型添加特定能力
        agent_type = type(agent).__name__.lower()
        if "react" in agent_type:
            capabilities.extend(["推理", "分析", "问题解决"])
        elif "plan" in agent_type:
            capabilities.extend(["计划", "组织", "项目管理"])
        elif "tool" in agent_type:
            capabilities.extend(["工具使用", "执行", "操作"])
        
        # 基于可用工具添加能力
        if hasattr(agent, 'tools') and agent.tools:
            for tool in agent.tools:
                tool_name = tool.name.lower()
                if "search" in tool_name or "query" in tool_name:
                    capabilities.append("搜索")
                elif "calculate" in tool_name or "math" in tool_name:
                    capabilities.append("计算")
                elif "code" in tool_name or "python" in tool_name:
                    capabilities.append("编程")
                elif "translate" in tool_name:
                    capabilities.append("翻译")
        
        return list(set(capabilities))  # 去重
    
    async def coordinate(self, task: Task, callbacks: List[Any] = None) -> Result:
        """
        协调执行任务（顺序模式）
        
        Args:
            task: 要执行的任务
            
        Returns:
            执行结果
        """
        async with self.execution_lock:
            return await self._execute_task_sequential(task)
    
    async def _execute_task_sequential(self, task: Task, callbacks: List[Any] = None) -> Result:
        """
        顺序执行任务
        
        Args:
            task: 要执行的任务
            
        Returns:
            执行结果
        """
        start_time = time.time()
        
        try:
            # plan phase start
            if callbacks:
                for cb in callbacks:
                    try:
                        await cb.emit("plan_start", {"task_id": task.id, "title": task.title})
                    except Exception:
                        pass
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
            
            # plan phase end
            if callbacks:
                for cb in callbacks:
                    try:
                        await cb.emit("plan_end", {"task_id": task.id})
                    except Exception:
                        pass

            # 执行任务
            if callbacks:
                for cb in callbacks:
                    try:
                        await cb.emit("execute_start", {"task_id": task.id, "agent_id": agent.agent_id})
                    except Exception:
                        pass
            # 将回调注入到智能体（若支持），并重建执行器以便回调生效
            try:
                if callbacks is not None and hasattr(agent, 'callbacks'):
                    agent.callbacks = callbacks
                    if hasattr(agent, '_create_agent_executor'):
                        agent.agent_executor = agent._create_agent_executor()
            except Exception:
                pass

            result = await self._execute_with_agent(task, agent)
            
            # 完成任务
            self.complete_task(task, result)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            # 更新智能体性能信息
            if agent:
                success = result.status == ResultStatus.SUCCESS
                self.agent_selector.update_agent_performance(
                    agent.agent_id, success, execution_time
                )
                # 更新负载信息
                self.agent_selector.update_agent_load(agent.agent_id, 0.0)
            
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
        
        # 使用智能选择器选择最佳智能体
        selected_agent = self.agent_selector.select_agent(task, available_agents)
        
        if selected_agent:
            self.logger.info(f"智能选择器为任务 '{task.title}' 选择了智能体: {selected_agent.name}")
        else:
            self.logger.warning(f"智能选择器未能为任务 '{task.title}' 找到合适的智能体")
        
        return selected_agent

    @handle_errors
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
            # 使用错误处理器处理异常
            error_info = self.error_handler.error_classifier.classify_error(
                e, 
                context={
                    "task_id": task.id,
                    "agent_id": agent.agent_id,
                    "task_title": task.title
                }
            )
            
            return Result(
                id=str(uuid.uuid4()),
                task_id=task.id,
                agent_id=agent.agent_id,
                status=ResultStatus.ERROR,
                error_message=f"智能体执行失败: {error_info.message}",
                metadata={"error_info": error_info.__dict__}
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
            "execution_mode": "sequential",
            "selection_strategy": self.agent_selector.strategy.value,
            "agent_selector_stats": self.agent_selector.get_selection_stats(),
            "error_stats": self.error_handler.get_error_stats()
        })
        return base_status
    
    def __str__(self) -> str:
        return f"顺序执行协调器[{self.coordinator_id}]: {self.name} - 队列: {len(self.task_queue)}个任务"
