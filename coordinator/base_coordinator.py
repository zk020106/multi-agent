"""
多智能体协调器基类
定义协调器的统一接口和基础功能
"""
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from schema import Task, Result, TaskStatus, ResultStatus
from utils.logger import get_logger


class BaseCoordinator(ABC):
    """
    协调器基类
    
    负责管理多个智能体的任务分配和结果聚合
    """
    
    def __init__(self, coordinator_id: str, name: str, description: str = ""):
        """
        初始化协调器
        
        Args:
            coordinator_id: 协调器ID
            name: 协调器名称
            description: 协调器描述
        """
        self.coordinator_id = coordinator_id
        self.name = name
        self.description = description
        self.logger = get_logger(f"coordinator.{coordinator_id}")
        
        # 智能体管理
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_roles: Dict[str, str] = {}  # agent_id -> role
        
        # 任务管理
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # 统计信息
        self.tasks_assigned = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        
        self.created_at = datetime.now()
        
        self.logger.info(f"协调器 {self.name} 初始化完成")
    
    @abstractmethod
    async def coordinate(self, task: Task) -> Result:
        """
        协调执行任务
        
        Args:
            task: 要执行的任务
            
        Returns:
            执行结果
        """
        pass
    
    def add_agent(self, agent: BaseAgent, role: str = "general") -> None:
        """
        添加智能体到协调器
        
        Args:
            agent: 智能体实例
            role: 智能体角色
        """
        self.agents[agent.agent_id] = agent
        self.agent_roles[agent.agent_id] = role
        self.logger.info(f"添加智能体 {agent.name} (角色: {role})")
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        移除智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            是否成功移除
        """
        if agent_id in self.agents:
            agent_name = self.agents[agent_id].name
            del self.agents[agent_id]
            if agent_id in self.agent_roles:
                del self.agent_roles[agent_id]
            self.logger.info(f"移除智能体 {agent_name}")
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        获取智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            智能体实例或None
        """
        return self.agents.get(agent_id)
    
    def get_agents_by_role(self, role: str) -> List[BaseAgent]:
        """
        根据角色获取智能体列表
        
        Args:
            role: 角色名称
            
        Returns:
            智能体列表
        """
        return [
            agent for agent_id, agent in self.agents.items()
            if self.agent_roles.get(agent_id) == role
        ]
    
    def get_available_agents(self) -> List[BaseAgent]:
        """
        获取可用的智能体列表
        
        Returns:
            可用智能体列表
        """
        return [agent for agent in self.agents.values() if not agent.is_busy]
    
    def get_busy_agents(self) -> List[BaseAgent]:
        """
        获取忙碌的智能体列表
        
        Returns:
            忙碌的智能体列表
        """
        return [agent for agent in self.agents.values() if agent.is_busy]
    
    def assign_task_to_agent(self, task: Task, agent: BaseAgent) -> None:
        """
        将任务分配给智能体
        
        Args:
            task: 任务对象
            agent: 智能体实例
        """
        task.assigned_agent = agent.agent_id
        task.update_status(TaskStatus.IN_PROGRESS)
        self.active_tasks[task.id] = task
        self.tasks_assigned += 1
        
        self.logger.info(f"任务 {task.title} 分配给智能体 {agent.name}")
    
    def complete_task(self, task: Task, result: Result) -> None:
        """
        完成任务
        
        Args:
            task: 任务对象
            result: 执行结果
        """
        if task.id in self.active_tasks:
            del self.active_tasks[task.id]
        
        if result.status == ResultStatus.SUCCESS:
            self.completed_tasks[task.id] = task
            self.tasks_completed += 1
            task.update_status(TaskStatus.COMPLETED)
        else:
            self.failed_tasks[task.id] = task
            self.tasks_failed += 1
            task.update_status(TaskStatus.FAILED)
        
        self.total_execution_time += result.execution_time or 0.0
        
        self.logger.info(f"任务 {task.title} 完成，状态: {result.status.value}")
    
    def create_subtask(
        self,
        parent_task: Task,
        title: str,
        description: str,
        priority: str = "normal"
    ) -> Task:
        """
        创建子任务
        
        Args:
            parent_task: 父任务
            title: 子任务标题
            description: 子任务描述
            priority: 优先级
            
        Returns:
            子任务对象
        """
        from schema import TaskPriority
        
        subtask = Task(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            priority=TaskPriority(priority),
            metadata={
                "parent_task_id": parent_task.id,
                "created_by": self.coordinator_id
            }
        )
        
        # 添加依赖关系
        subtask.add_dependency(parent_task.id)
        
        self.logger.info(f"创建子任务: {title}")
        return subtask
    
    def aggregate_results(self, results: List[Result]) -> Result:
        """
        聚合多个结果
        
        Args:
            results: 结果列表
            
        Returns:
            聚合后的结果
        """
        if not results:
            return Result(
                id=str(uuid.uuid4()),
                task_id="",
                agent_id=self.coordinator_id,
                status=ResultStatus.ERROR,
                error_message="没有结果可聚合"
            )
        
        # 统计成功和失败的结果
        successful_results = [r for r in results if r.status == ResultStatus.SUCCESS]
        failed_results = [r for r in results if r.status != ResultStatus.SUCCESS]
        
        # 计算总执行时间
        total_time = sum(r.execution_time or 0.0 for r in results)
        
        # 确定整体状态
        if len(successful_results) == len(results):
            overall_status = ResultStatus.SUCCESS
        elif len(successful_results) > 0:
            overall_status = ResultStatus.PARTIAL
        else:
            overall_status = ResultStatus.FAILURE
        
        # 聚合数据
        aggregated_data = {
            "total_results": len(results),
            "successful_results": len(successful_results),
            "failed_results": len(failed_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "individual_results": [
                {
                    "agent_id": r.agent_id,
                    "status": r.status.value,
                    "execution_time": r.execution_time,
                    "data": r.data
                }
                for r in results
            ]
        }
        
        # 收集错误信息
        error_messages = [r.error_message for r in failed_results if r.error_message]
        
        return Result(
            id=str(uuid.uuid4()),
            task_id=results[0].task_id if results else "",
            agent_id=self.coordinator_id,
            status=overall_status,
            data=aggregated_data,
            error_message="; ".join(error_messages) if error_messages else None,
            execution_time=total_time
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取协调器状态
        
        Returns:
            状态信息字典
        """
        return {
            "coordinator_id": self.coordinator_id,
            "name": self.name,
            "description": self.description,
            "total_agents": len(self.agents),
            "available_agents": len(self.get_available_agents()),
            "busy_agents": len(self.get_busy_agents()),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "tasks_assigned": self.tasks_assigned,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_execution_time": self.total_execution_time,
            "created_at": self.created_at.isoformat()
        }
    
    def get_agent_roles(self) -> Dict[str, str]:
        """
        获取智能体角色映射
        
        Returns:
            智能体ID到角色的映射
        """
        return self.agent_roles.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.tasks_assigned = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        self.logger.info("统计信息已重置")
    
    def __str__(self) -> str:
        return f"协调器[{self.coordinator_id}]: {self.name} - {len(self.agents)}个智能体"
