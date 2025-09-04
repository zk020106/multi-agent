"""
消息路由基类
定义路由器的统一接口和基础功能
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from schema import Message, Task
from utils.logger import get_logger


class BaseRouter(ABC):
    """
    路由器基类
    
    负责智能体之间的消息分发和路由决策
    """
    
    def __init__(self, router_id: str, name: str, description: str = ""):
        """
        初始化路由器
        
        Args:
            router_id: 路由器ID
            name: 路由器名称
            description: 路由器描述
        """
        self.router_id = router_id
        self.name = name
        self.description = description
        self.logger = get_logger(f"router.{router_id}")
        
        # 智能体注册表
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}  # agent_id -> capabilities
        self.agent_roles: Dict[str, str] = {}  # agent_id -> role
        
        # 路由规则和策略
        self.routing_rules: List[Dict[str, Any]] = []
        self.routing_history: List[Dict[str, Any]] = []
        
        # 统计信息
        self.messages_routed = 0
        self.routing_errors = 0
        self.total_routing_time = 0.0
        
        self.created_at = datetime.now()
        
        self.logger.info(f"路由器 {self.name} 初始化完成")
    
    @abstractmethod
    def route_message(self, message: Message) -> List[str]:
        """
        路由消息到合适的智能体
        
        Args:
            message: 要路由的消息
            
        Returns:
            目标智能体ID列表
        """
        pass
    
    @abstractmethod
    def route_task(self, task: Task) -> Optional[str]:
        """
        路由任务到合适的智能体
        
        Args:
            task: 要路由的任务
            
        Returns:
            目标智能体ID或None
        """
        pass
    
    def register_agent(
        self,
        agent: BaseAgent,
        capabilities: List[str] = None,
        role: str = "general"
    ) -> None:
        """
        注册智能体到路由器
        
        Args:
            agent: 智能体实例
            capabilities: 智能体能力列表
            role: 智能体角色
        """
        self.registered_agents[agent.agent_id] = agent
        self.agent_capabilities[agent.agent_id] = capabilities or []
        self.agent_roles[agent.agent_id] = role
        
        self.logger.info(f"注册智能体 {agent.name} (角色: {role}, 能力: {capabilities})")
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        注销智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            是否成功注销
        """
        if agent_id in self.registered_agents:
            agent_name = self.registered_agents[agent_id].name
            del self.registered_agents[agent_id]
            if agent_id in self.agent_capabilities:
                del self.agent_capabilities[agent_id]
            if agent_id in self.agent_roles:
                del self.agent_roles[agent_id]
            
            self.logger.info(f"注销智能体 {agent_name}")
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
        return self.registered_agents.get(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """
        根据能力获取智能体列表
        
        Args:
            capability: 能力名称
            
        Returns:
            具有该能力的智能体列表
        """
        agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if capability in capabilities:
                agents.append(self.registered_agents[agent_id])
        return agents
    
    def get_agents_by_role(self, role: str) -> List[BaseAgent]:
        """
        根据角色获取智能体列表
        
        Args:
            role: 角色名称
            
        Returns:
            具有该角色的智能体列表
        """
        agents = []
        for agent_id, agent_role in self.agent_roles.items():
            if agent_role == role:
                agents.append(self.registered_agents[agent_id])
        return agents
    
    def get_available_agents(self) -> List[BaseAgent]:
        """
        获取可用的智能体列表
        
        Returns:
            可用的智能体列表
        """
        return [agent for agent in self.registered_agents.values() if not agent.is_busy]
    
    def add_routing_rule(self, rule: Dict[str, Any]) -> None:
        """
        添加路由规则
        
        Args:
            rule: 路由规则字典
        """
        rule["id"] = str(uuid.uuid4())
        rule["created_at"] = datetime.now().isoformat()
        self.routing_rules.append(rule)
        
        self.logger.info(f"添加路由规则: {rule.get('name', '未命名规则')}")
    
    def remove_routing_rule(self, rule_id: str) -> bool:
        """
        移除路由规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            是否成功移除
        """
        for i, rule in enumerate(self.routing_rules):
            if rule.get("id") == rule_id:
                del self.routing_rules[i]
                self.logger.info(f"移除路由规则: {rule.get('name', '未命名规则')}")
                return True
        return False
    
    def _record_routing_decision(
        self,
        message: Message,
        target_agents: List[str],
        routing_time: float,
        success: bool = True
    ) -> None:
        """
        记录路由决策
        
        Args:
            message: 原始消息
            target_agents: 目标智能体列表
            routing_time: 路由耗时
            success: 是否成功
        """
        decision = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "message_id": message.id,
            "message_type": message.message_type.value,
            "sender": message.sender,
            "target_agents": target_agents,
            "routing_time": routing_time,
            "success": success
        }
        
        self.routing_history.append(decision)
        
        # 限制历史记录数量
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-500:]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        获取路由统计信息
        
        Returns:
            统计信息字典
        """
        successful_routes = len([h for h in self.routing_history if h["success"]])
        failed_routes = len([h for h in self.routing_history if not h["success"]])
        
        return {
            "router_id": self.router_id,
            "name": self.name,
            "registered_agents": len(self.registered_agents),
            "routing_rules": len(self.routing_rules),
            "total_routes": len(self.routing_history),
            "successful_routes": successful_routes,
            "failed_routes": failed_routes,
            "success_rate": successful_routes / max(1, len(self.routing_history)) * 100,
            "average_routing_time": sum(h["routing_time"] for h in self.routing_history) / max(1, len(self.routing_history)),
            "created_at": self.created_at.isoformat()
        }
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """
        获取所有智能体的能力映射
        
        Returns:
            智能体ID到能力列表的映射
        """
        return self.agent_capabilities.copy()
    
    def get_agent_roles(self) -> Dict[str, str]:
        """
        获取所有智能体的角色映射
        
        Returns:
            智能体ID到角色的映射
        """
        return self.agent_roles.copy()
    
    def clear_routing_history(self) -> int:
        """
        清除路由历史
        
        Returns:
            清除的记录数量
        """
        count = len(self.routing_history)
        self.routing_history.clear()
        self.logger.info(f"清除路由历史，共 {count} 条记录")
        return count
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.messages_routed = 0
        self.routing_errors = 0
        self.total_routing_time = 0.0
        self.logger.info("统计信息已重置")
    
    def __str__(self) -> str:
        return f"路由器[{self.router_id}]: {self.name} - {len(self.registered_agents)}个智能体"
