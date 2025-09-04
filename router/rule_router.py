"""
基于规则的消息路由器
使用预定义规则进行消息路由
"""

import re
import time
import uuid
from typing import Any, Dict, List, Optional

from schema import Message, Task, MessageType
from .base_router import BaseRouter


class RuleRouter(BaseRouter):
    """
    基于规则的路由器
    
    特点：
    - 使用预定义规则进行路由决策
    - 支持关键词匹配、正则表达式等
    - 路由速度快，可预测性强
    - 适合规则明确的场景
    """
    
    def __init__(
        self,
        router_id: str,
        name: str = "规则路由器",
        description: str = "基于预定义规则的消息路由器"
    ):
        """
        初始化规则路由器
        
        Args:
            router_id: 路由器ID
            name: 路由器名称
            description: 路由器描述
        """
        super().__init__(router_id, name, description)
        
        # 默认路由规则
        self._setup_default_rules()
        
        self.logger.info(f"规则路由器初始化完成，默认规则数: {len(self.routing_rules)}")
    
    def _setup_default_rules(self) -> None:
        """设置默认路由规则"""
        default_rules = [
            {
                "name": "数学计算规则",
                "description": "包含数学表达式的消息路由到计算智能体",
                "conditions": {
                    "message_contains": ["计算", "数学", "+", "-", "*", "/", "="],
                    "message_type": ["user_input"]
                },
                "target_capability": "calculation",
                "priority": 1
            },
            {
                "name": "代码执行规则",
                "description": "包含代码的消息路由到代码执行智能体",
                "conditions": {
                    "message_contains": ["代码", "python", "def ", "import", "print("],
                    "message_type": ["user_input"]
                },
                "target_capability": "code_execution",
                "priority": 1
            },
            {
                "name": "API调用规则",
                "description": "包含API相关关键词的消息路由到API智能体",
                "conditions": {
                    "message_contains": ["API", "http", "请求", "接口", "调用"],
                    "message_type": ["user_input"]
                },
                "target_capability": "api_calling",
                "priority": 1
            },
            {
                "name": "文档搜索规则",
                "description": "包含搜索相关关键词的消息路由到搜索智能体",
                "conditions": {
                    "message_contains": ["搜索", "查找", "文档", "信息"],
                    "message_type": ["user_input"]
                },
                "target_capability": "document_search",
                "priority": 1
            },
            {
                "name": "高优先级任务规则",
                "description": "高优先级任务路由到专用智能体",
                "conditions": {
                    "task_priority": ["urgent", "high"],
                    "message_type": ["user_input"]
                },
                "target_role": "priority_handler",
                "priority": 0
            },
            {
                "name": "默认规则",
                "description": "默认路由到通用智能体",
                "conditions": {
                    "message_type": ["user_input"]
                },
                "target_role": "general",
                "priority": 999
            }
        ]
        
        for rule in default_rules:
            self.add_routing_rule(rule)
    
    def route_message(self, message: Message) -> List[str]:
        """
        根据规则路由消息
        
        Args:
            message: 要路由的消息
            
        Returns:
            目标智能体ID列表
        """
        start_time = time.time()
        
        try:
            # 按优先级排序规则
            sorted_rules = sorted(self.routing_rules, key=lambda x: x.get("priority", 999))
            
            for rule in sorted_rules:
                if self._match_rule(message, rule):
                    target_agents = self._get_target_agents(rule)
                    if target_agents:
                        routing_time = time.time() - start_time
                        self._record_routing_decision(message, target_agents, routing_time, True)
                        self.messages_routed += 1
                        self.total_routing_time += routing_time
                        
                        self.logger.info(f"消息 {message.id} 路由到智能体: {target_agents}")
                        return target_agents
            
            # 如果没有匹配的规则，返回空列表
            routing_time = time.time() - start_time
            self._record_routing_decision(message, [], routing_time, False)
            self.routing_errors += 1
            
            self.logger.warning(f"消息 {message.id} 没有匹配的路由规则")
            return []
            
        except Exception as e:
            routing_time = time.time() - start_time
            self._record_routing_decision(message, [], routing_time, False)
            self.routing_errors += 1
            
            self.logger.error(f"路由消息失败: {str(e)}")
            return []
    
    def route_task(self, task: Task) -> Optional[str]:
        """
        根据规则路由任务
        
        Args:
            task: 要路由的任务
            
        Returns:
            目标智能体ID或None
        """
        # 创建临时消息对象用于路由
        temp_message = Message(
            id=str(uuid.uuid4()),
            sender="system",
            receiver="",
            content=f"{task.title} {task.description}",
            message_type=MessageType.USER_INPUT,
            metadata={"task_id": task.id, "task_priority": task.priority.value}
        )
        
        target_agents = self.route_message(temp_message)
        return target_agents[0] if target_agents else None
    
    def _match_rule(self, message: Message, rule: Dict[str, Any]) -> bool:
        """
        检查消息是否匹配规则
        
        Args:
            message: 消息对象
            rule: 路由规则
            
        Returns:
            是否匹配
        """
        conditions = rule.get("conditions", {})
        
        # 检查消息类型
        if "message_type" in conditions:
            if message.message_type.value not in conditions["message_type"]:
                return False
        
        # 检查消息内容
        if "message_contains" in conditions:
            message_text = message.content.lower()
            keywords = [kw.lower() for kw in conditions["message_contains"]]
            if not any(keyword in message_text for keyword in keywords):
                return False
        
        # 检查正则表达式
        if "message_regex" in conditions:
            pattern = conditions["message_regex"]
            if not re.search(pattern, message.content, re.IGNORECASE):
                return False
        
        # 检查任务优先级
        if "task_priority" in conditions:
            task_priority = message.metadata.get("task_priority")
            if not task_priority or task_priority not in conditions["task_priority"]:
                return False
        
        # 检查发送者
        if "sender" in conditions:
            if message.sender not in conditions["sender"]:
                return False
        
        # 检查接收者
        if "receiver" in conditions:
            if message.receiver not in conditions["receiver"]:
                return False
        
        return True
    
    def _get_target_agents(self, rule: Dict[str, Any]) -> List[str]:
        """
        根据规则获取目标智能体
        
        Args:
            rule: 路由规则
            
        Returns:
            目标智能体ID列表
        """
        target_agents = []
        
        # 按能力路由
        if "target_capability" in rule:
            capability = rule["target_capability"]
            agents = self.get_agents_by_capability(capability)
            target_agents.extend([agent.agent_id for agent in agents])
        
        # 按角色路由
        if "target_role" in rule:
            role = rule["target_role"]
            agents = self.get_agents_by_role(role)
            target_agents.extend([agent.agent_id for agent in agents])
        
        # 按智能体ID路由
        if "target_agents" in rule:
            target_agents.extend(rule["target_agents"])
        
        # 过滤可用的智能体
        available_agents = [agent_id for agent_id in target_agents 
                          if agent_id in self.registered_agents and not self.registered_agents[agent_id].is_busy]
        
        return available_agents
    
    def add_keyword_rule(
        self,
        name: str,
        keywords: List[str],
        target_capability: str = None,
        target_role: str = None,
        target_agents: List[str] = None,
        priority: int = 1
    ) -> None:
        """
        添加关键词规则
        
        Args:
            name: 规则名称
            keywords: 关键词列表
            target_capability: 目标能力
            target_role: 目标角色
            target_agents: 目标智能体ID列表
            priority: 优先级
        """
        rule = {
            "name": name,
            "description": f"关键词匹配规则: {', '.join(keywords)}",
            "conditions": {
                "message_contains": keywords,
                "message_type": ["user_input"]
            },
            "priority": priority
        }
        
        if target_capability:
            rule["target_capability"] = target_capability
        if target_role:
            rule["target_role"] = target_role
        if target_agents:
            rule["target_agents"] = target_agents
        
        self.add_routing_rule(rule)
    
    def add_regex_rule(
        self,
        name: str,
        pattern: str,
        target_capability: str = None,
        target_role: str = None,
        target_agents: List[str] = None,
        priority: int = 1
    ) -> None:
        """
        添加正则表达式规则
        
        Args:
            name: 规则名称
            pattern: 正则表达式模式
            target_capability: 目标能力
            target_role: 目标角色
            target_agents: 目标智能体ID列表
            priority: 优先级
        """
        rule = {
            "name": name,
            "description": f"正则表达式规则: {pattern}",
            "conditions": {
                "message_regex": pattern,
                "message_type": ["user_input"]
            },
            "priority": priority
        }
        
        if target_capability:
            rule["target_capability"] = target_capability
        if target_role:
            rule["target_role"] = target_role
        if target_agents:
            rule["target_agents"] = target_agents
        
        self.add_routing_rule(rule)
    
    def get_rule_stats(self) -> Dict[str, Any]:
        """
        获取规则统计信息
        
        Returns:
            规则统计信息
        """
        return {
            "total_rules": len(self.routing_rules),
            "rules_by_priority": {
                str(rule.get("priority", 999)): len([r for r in self.routing_rules if r.get("priority", 999) == rule.get("priority", 999)])
                for rule in self.routing_rules
            },
            "rule_types": {
                "keyword_rules": len([r for r in self.routing_rules if "message_contains" in r.get("conditions", {})]),
                "regex_rules": len([r for r in self.routing_rules if "message_regex" in r.get("conditions", {})]),
                "capability_rules": len([r for r in self.routing_rules if "target_capability" in r]),
                "role_rules": len([r for r in self.routing_rules if "target_role" in r])
            }
        }
    
    def __str__(self) -> str:
        return f"规则路由器[{self.router_id}]: {self.name} - {len(self.routing_rules)}条规则"
