"""
智能体基类，基于LangChain框架
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseLanguageModel
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool

from schema import Message, Task, Result, MessageType, TaskStatus, ResultStatus
from utils.logger import AgentLogger


class BaseAgent(ABC):
    """
    智能体基类，基于LangChain框架
    
    所有智能体都需要实现以下核心方法：
    - receive_message: 接收消息
    - act: 执行动作
    - return_result: 返回结果
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: List[BaseTool] = None,
        memory: BaseMemory = None,
        system_prompt: str = "",
        callbacks: List[BaseCallbackHandler] = None
    ):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体唯一标识
            name: 智能体名称
            description: 智能体描述
            llm: 大语言模型实例
            tools: 可用工具列表
            memory: 记忆组件
            system_prompt: 系统提示词
            callbacks: 回调处理器列表
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools or []
        self.memory = memory
        self.system_prompt = system_prompt
        self.callbacks = callbacks or []
        
        # 日志记录器
        self.logger = AgentLogger(agent_id)
        
        # 状态管理
        self.is_busy = False
        self.current_task: Optional[Task] = None
        self.created_at = datetime.now()
        
        # 统计信息
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        
        self.logger.info(f"智能体 {self.name} 初始化完成")
    
    @abstractmethod
    def receive_message(self, message: Message) -> None:
        """
        接收消息
        
        Args:
            message: 接收到的消息对象
        """
        pass
    
    @abstractmethod
    def act(self, task: Task) -> Result:
        """
        执行任务动作
        
        Args:
            task: 要执行的任务
            
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    def return_result(self, result: Result) -> Message:
        """
        返回执行结果
        
        Args:
            result: 执行结果
            
        Returns:
            结果消息
        """
        pass
    
    def add_tool(self, tool: BaseTool) -> None:
        """
        添加工具
        
        Args:
            tool: 要添加的工具
        """
        self.tools.append(tool)
        self.logger.info(f"添加工具: {tool.name}")
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        移除工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            是否成功移除
        """
        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                del self.tools[i]
                self.logger.info(f"移除工具: {tool_name}")
                return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        获取指定工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具实例或None
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def list_tools(self) -> List[str]:
        """
        列出所有可用工具名称
        
        Returns:
            工具名称列表
        """
        return [tool.name for tool in self.tools]
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """
        更新系统提示词
        
        Args:
            new_prompt: 新的系统提示词
        """
        self.system_prompt = new_prompt
        self.logger.info("系统提示词已更新")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取智能体状态信息
        
        Returns:
            状态信息字典
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "is_busy": self.is_busy,
            "current_task": self.current_task.id if self.current_task else None,
            "tools_count": len(self.tools),
            "tools": self.list_tools(),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_execution_time": self.total_execution_time,
            "created_at": self.created_at.isoformat(),
            "memory_type": type(self.memory).__name__ if self.memory else None
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        self.logger.info("统计信息已重置")
    
    def _create_result(
        self,
        task: Task,
        status: ResultStatus,
        data: Any = None,
        error_message: str = None,
        execution_time: float = 0.0
    ) -> Result:
        """
        创建结果对象
        
        Args:
            task: 任务对象
            status: 结果状态
            data: 结果数据
            error_message: 错误信息
            execution_time: 执行时间
            
        Returns:
            结果对象
        """
        result = Result(
            id=str(uuid.uuid4()),
            task_id=task.id,
            agent_id=self.agent_id,
            status=status,
            data=data,
            error_message=error_message,
            execution_time=execution_time
        )
        
        # 添加日志
        result.add_log(f"智能体 {self.name} 开始执行任务: {task.title}")
        if error_message:
            result.add_log(f"错误: {error_message}")
        else:
            result.add_log(f"任务执行完成，状态: {status.value}")
        
        return result
    
    def _create_message(
        self,
        content: str,
        receiver: str,
        message_type: MessageType = MessageType.AGENT_RESPONSE,
        conversation_id: str = None
    ) -> Message:
        """
        创建消息对象
        
        Args:
            content: 消息内容
            receiver: 接收者ID
            message_type: 消息类型
            conversation_id: 对话ID
            
        Returns:
            消息对象
        """
        return Message(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            message_type=message_type,
            conversation_id=conversation_id
        )
    
    def _update_task_status(self, task: Task, status: TaskStatus) -> None:
        """
        更新任务状态
        
        Args:
            task: 任务对象
            status: 新状态
        """
        task.update_status(status)
        self.logger.info(f"任务 {task.id} 状态更新为: {status.value}")
    
    def __str__(self) -> str:
        return f"智能体[{self.agent_id}]: {self.name} - {self.description}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.agent_id}', name='{self.name}')>"
