"""
智能体工厂类
用于创建不同类型的智能体实例
"""

from enum import Enum
from typing import Dict, List, Optional, Type, Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool

from .base_agent import BaseAgent
from .plan_execute_agent import LangChainPlanExecuteAgent
from .react_agent import ReActAgent
from .tool_agent import ToolAgent


class AgentType(Enum):
    """智能体类型枚举"""
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    TOOL = "tool"


class AgentFactory:
    """
    智能体工厂类
    
    负责创建和管理不同类型的智能体实例
    """
    
    # 智能体类型映射
    AGENT_CLASSES: Dict[AgentType, Type[BaseAgent]] = {
        AgentType.REACT: ReActAgent,
        AgentType.PLAN_EXECUTE: LangChainPlanExecuteAgent,
        AgentType.TOOL: ToolAgent
    }
    
    def __init__(self):
        """初始化智能体工厂"""
        self._created_agents: Dict[str, BaseAgent] = {}
    
    def create_agent(
        self,
        agent_type: AgentType,
        agent_id: str,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: List[BaseTool] = None,
        memory: BaseMemory = None,
        system_prompt: str = "",
        **kwargs
    ) -> BaseAgent:
        """
        创建智能体实例
        
        Args:
            agent_type: 智能体类型
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            llm: 大语言模型
            tools: 工具列表
            memory: 记忆组件
            system_prompt: 系统提示词
            **kwargs: 其他参数
            
        Returns:
            智能体实例
            
        Raises:
            ValueError: 如果智能体类型不支持
        """
        if agent_type not in self.AGENT_CLASSES:
            raise ValueError(f"不支持的智能体类型: {agent_type}")
        
        # 检查ID是否已存在
        if agent_id in self._created_agents:
            raise ValueError(f"智能体ID {agent_id} 已存在")
        
        # 获取智能体类
        agent_class = self.AGENT_CLASSES[agent_type]
        
        # 创建智能体实例
        agent = agent_class(
            agent_id=agent_id,
            name=name,
            description=description,
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # 注册到工厂
        self._created_agents[agent_id] = agent
        
        return agent
    
    def create_react_agent(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: List[BaseTool] = None,
        memory: BaseMemory = None,
        system_prompt: str = "",
        max_iterations: int = 10,
        verbose: bool = True,
        **kwargs
    ) -> ReActAgent:
        """
        创建ReAct智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            llm: 大语言模型
            tools: 工具列表
            memory: 记忆组件
            system_prompt: 系统提示词
            max_iterations: 最大迭代次数
            verbose: 是否详细输出
            **kwargs: 其他参数
            
        Returns:
            ReAct智能体实例
        """
        return self.create_agent(
            agent_type=AgentType.REACT,
            agent_id=agent_id,
            name=name,
            description=description,
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
            **kwargs
        )
    
    def create_plan_execute_agent(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: List[BaseTool] = None,
        memory: BaseMemory = None,
        system_prompt: str = "",
        max_plan_steps: int = 10,
        **kwargs
    ) -> LangChainPlanExecuteAgent:
        """
        创建计划执行智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            llm: 大语言模型
            tools: 工具列表
            memory: 记忆组件
            system_prompt: 系统提示词
            max_plan_steps: 最大计划步骤数
            **kwargs: 其他参数
            
        Returns:
            计划执行智能体实例
        """
        return self.create_agent(
            agent_type=AgentType.PLAN_EXECUTE,
            agent_id=agent_id,
            name=name,
            description=description,
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            max_plan_steps=max_plan_steps,
            **kwargs
        )
    
    def create_tool_agent(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: List[BaseTool] = None,
        memory: BaseMemory = None,
        system_prompt: str = "",
        max_iterations: int = 5,
        verbose: bool = True,
        **kwargs
    ) -> ToolAgent:
        """
        创建工具驱动智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            llm: 大语言模型
            tools: 工具列表
            memory: 记忆组件
            system_prompt: 系统提示词
            max_iterations: 最大迭代次数
            verbose: 是否详细输出
            **kwargs: 其他参数
            
        Returns:
            工具驱动智能体实例
        """
        return self.create_agent(
            agent_type=AgentType.TOOL,
            agent_id=agent_id,
            name=name,
            description=description,
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            verbose=verbose,
            **kwargs
        )
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        获取智能体实例
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            智能体实例或None
        """
        return self._created_agents.get(agent_id)
    
    def remove_agent(self, agent_id: str) -> bool:
        """
        移除智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            是否成功移除
        """
        if agent_id in self._created_agents:
            del self._created_agents[agent_id]
            return True
        return False
    
    def list_agents(self) -> List[str]:
        """
        列出所有智能体ID
        
        Returns:
            智能体ID列表
        """
        return list(self._created_agents.keys())
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        获取智能体信息
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            智能体信息字典或None
        """
        agent = self.get_agent(agent_id)
        if agent:
            return agent.get_status()
        return None
    
    def get_all_agents_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有智能体信息
        
        Returns:
            智能体信息字典
        """
        return {
            agent_id: agent.get_status()
            for agent_id, agent in self._created_agents.items()
        }
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """
        根据类型获取智能体列表
        
        Args:
            agent_type: 智能体类型
            
        Returns:
            智能体实例列表
        """
        return [
            agent for agent in self._created_agents.values()
            if isinstance(agent, self.AGENT_CLASSES[agent_type])
        ]
    
    def get_busy_agents(self) -> List[BaseAgent]:
        """
        获取忙碌的智能体列表
        
        Returns:
            忙碌的智能体列表
        """
        return [agent for agent in self._created_agents.values() if agent.is_busy]
    
    def get_available_agents(self) -> List[BaseAgent]:
        """
        获取可用的智能体列表
        
        Returns:
            可用的智能体列表
        """
        return [agent for agent in self._created_agents.values() if not agent.is_busy]
    
    def clear_all_agents(self) -> int:
        """
        清除所有智能体
        
        Returns:
            清除的智能体数量
        """
        count = len(self._created_agents)
        self._created_agents.clear()
        return count
    
    def get_agent_count(self) -> int:
        """
        获取智能体总数
        
        Returns:
            智能体数量
        """
        return len(self._created_agents)
    
    def get_agent_type_count(self) -> Dict[AgentType, int]:
        """
        获取各类型智能体数量
        
        Returns:
            各类型智能体数量字典
        """
        type_count = {}
        for agent_type in AgentType:
            type_count[agent_type] = len(self.get_agents_by_type(agent_type))
        return type_count


# 全局智能体工厂实例
_global_factory = AgentFactory()


def get_agent_factory() -> AgentFactory:
    """获取全局智能体工厂实例"""
    return _global_factory


def create_agent(agent_type: AgentType, **kwargs) -> BaseAgent:
    """使用全局工厂创建智能体"""
    return _global_factory.create_agent(agent_type, **kwargs)


def get_agent(agent_id: str) -> Optional[BaseAgent]:
    """从全局工厂获取智能体"""
    return _global_factory.get_agent(agent_id)
