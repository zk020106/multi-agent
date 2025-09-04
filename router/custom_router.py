"""
自定义路由器
支持用户自定义路由逻辑
"""
import time
import uuid
from typing import Any, Dict, List, Optional, Callable

from schema import Message, Task, MessageType
from .base_router import BaseRouter


class CustomRouter(BaseRouter):
    """
    自定义路由器
    
    特点：
    - 支持用户自定义路由函数
    - 灵活的路由策略配置
    - 支持多种路由模式组合
    - 适合特殊业务需求
    """
    
    def __init__(
        self,
        router_id: str,
        name: str = "自定义路由器",
        description: str = "支持自定义路由逻辑的路由器",
        routing_function: Optional[Callable] = None
    ):
        """
        初始化自定义路由器
        
        Args:
            router_id: 路由器ID
            name: 路由器名称
            description: 路由器描述
            routing_function: 自定义路由函数
        """
        super().__init__(router_id, name, description)
        
        self.routing_function = routing_function
        self.routing_modes = {
            "round_robin": self._round_robin_routing,
            "load_balancing": self._load_balancing_routing,
            "capability_based": self._capability_based_routing,
            "role_based": self._role_based_routing,
            "custom": self._custom_routing
        }
        self.current_mode = "capability_based"
        
        self.logger.info(f"自定义路由器初始化完成，当前模式: {self.current_mode}")
    
    def route_message(self, message: Message) -> List[str]:
        """
        路由消息
        
        Args:
            message: 要路由的消息
            
        Returns:
            目标智能体ID列表
        """
        start_time = time.time()
        
        try:
            # 根据当前模式选择路由策略
            routing_func = self.routing_modes.get(self.current_mode, self._capability_based_routing)
            target_agents = routing_func(message)
            
            routing_time = time.time() - start_time
            self._record_routing_decision(message, target_agents, routing_time, True)
            self.messages_routed += 1
            self.total_routing_time += routing_time
            
            self.logger.info(f"消息 {message.id} 路由到智能体: {target_agents}")
            return target_agents
            
        except Exception as e:
            routing_time = time.time() - start_time
            self._record_routing_decision(message, [], routing_time, False)
            self.routing_errors += 1
            
            self.logger.error(f"路由消息失败: {str(e)}")
            return []
    
    def route_task(self, task: Task) -> Optional[str]:
        """
        路由任务
        
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
            content=f"任务: {task.title}\n描述: {task.description}",
            message_type=MessageType.USER_INPUT,
            metadata={
                "task_id": task.id,
                "task_priority": task.priority.value,
                "task_type": "task_routing"
            }
        )
        
        target_agents = self.route_message(temp_message)
        return target_agents[0] if target_agents else None
    
    def _round_robin_routing(self, message: Message) -> List[str]:
        """
        轮询路由策略
        
        Args:
            message: 消息对象
            
        Returns:
            目标智能体ID列表
        """
        available_agents = self.get_available_agents()
        if not available_agents:
            return []
        
        # 简单的轮询：选择第一个可用智能体
        return [available_agents[0].agent_id]
    
    def _load_balancing_routing(self, message: Message) -> List[str]:
        """
        负载均衡路由策略
        
        Args:
            message: 消息对象
            
        Returns:
            目标智能体ID列表
        """
        available_agents = self.get_available_agents()
        if not available_agents:
            return []
        
        # 选择负载最轻的智能体（任务完成数最少）
        lightest_agent = min(available_agents, key=lambda a: a.tasks_completed + a.tasks_failed)
        return [lightest_agent.agent_id]
    
    def _capability_based_routing(self, message: Message) -> List[str]:
        """
        基于能力的路由策略
        
        Args:
            message: 消息对象
            
        Returns:
            目标智能体ID列表
        """
        # 分析消息内容，确定所需能力
        required_capabilities = self._analyze_message_capabilities(message)
        
        # 查找具有所需能力的智能体
        target_agents = []
        for capability in required_capabilities:
            agents = self.get_agents_by_capability(capability)
            available_agents = [agent for agent in agents if not agent.is_busy]
            if available_agents:
                target_agents.extend([agent.agent_id for agent in available_agents])
        
        # 去重
        return list(set(target_agents))
    
    def _role_based_routing(self, message: Message) -> List[str]:
        """
        基于角色的路由策略
        
        Args:
            message: 消息对象
            
        Returns:
            目标智能体ID列表
        """
        # 根据消息类型确定角色
        if message.message_type == MessageType.USER_INPUT:
            # 用户输入通常需要通用角色
            agents = self.get_agents_by_role("general")
        elif message.message_type == MessageType.TOOL_CALL:
            # 工具调用需要工具执行角色
            agents = self.get_agents_by_role("execution")
        else:
            # 其他情况使用通用角色
            agents = self.get_agents_by_role("general")
        
        available_agents = [agent for agent in agents if not agent.is_busy]
        return [agent.agent_id for agent in available_agents]
    
    def _custom_routing(self, message: Message) -> List[str]:
        """
        自定义路由策略
        
        Args:
            message: 消息对象
            
        Returns:
            目标智能体ID列表
        """
        if self.routing_function:
            try:
                return self.routing_function(message, self.registered_agents, self.agent_capabilities, self.agent_roles)
            except Exception as e:
                self.logger.error(f"自定义路由函数执行失败: {str(e)}")
                return self._fallback_routing(message)
        else:
            return self._fallback_routing(message)
    
    def _analyze_message_capabilities(self, message: Message) -> List[str]:
        """
        分析消息所需的能力
        
        Args:
            message: 消息对象
            
        Returns:
            所需能力列表
        """
        capabilities = []
        content = message.content.lower()
        
        # 简单的关键词匹配
        if any(keyword in content for keyword in ["计算", "数学", "+", "-", "*", "/"]):
            capabilities.append("calculation")
        
        if any(keyword in content for keyword in ["代码", "python", "def", "import"]):
            capabilities.append("code_execution")
        
        if any(keyword in content for keyword in ["api", "http", "请求", "接口"]):
            capabilities.append("api_calling")
        
        if any(keyword in content for keyword in ["搜索", "查找", "文档", "信息"]):
            capabilities.append("document_search")
        
        if any(keyword in content for keyword in ["数据库", "sql", "查询"]):
            capabilities.append("database")
        
        # 如果没有匹配到特定能力，返回通用能力
        if not capabilities:
            capabilities.append("general")
        
        return capabilities
    
    def _fallback_routing(self, message: Message) -> List[str]:
        """
        备用路由策略
        
        Args:
            message: 消息对象
            
        Returns:
            目标智能体ID列表
        """
        # 选择第一个可用的智能体
        available_agents = self.get_available_agents()
        if available_agents:
            return [available_agents[0].agent_id]
        return []
    
    def set_routing_mode(self, mode: str) -> bool:
        """
        设置路由模式
        
        Args:
            mode: 路由模式名称
            
        Returns:
            是否设置成功
        """
        if mode in self.routing_modes:
            self.current_mode = mode
            self.logger.info(f"路由模式已设置为: {mode}")
            return True
        else:
            self.logger.error(f"不支持的路由模式: {mode}")
            return False
    
    def set_custom_routing_function(self, routing_function: Callable) -> None:
        """
        设置自定义路由函数
        
        Args:
            routing_function: 自定义路由函数
        """
        self.routing_function = routing_function
        self.current_mode = "custom"
        self.logger.info("自定义路由函数已设置")
    
    def get_routing_modes(self) -> List[str]:
        """
        获取可用的路由模式
        
        Returns:
            路由模式列表
        """
        return list(self.routing_modes.keys())
    
    def get_current_mode(self) -> str:
        """
        获取当前路由模式
        
        Returns:
            当前路由模式
        """
        return self.current_mode
    
    def add_routing_mode(self, name: str, routing_function: Callable) -> None:
        """
        添加自定义路由模式
        
        Args:
            name: 模式名称
            routing_function: 路由函数
        """
        self.routing_modes[name] = routing_function
        self.logger.info(f"添加自定义路由模式: {name}")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        获取路由统计信息
        
        Returns:
            统计信息字典
        """
        base_stats = super().get_routing_stats()
        base_stats.update({
            "current_mode": self.current_mode,
            "available_modes": list(self.routing_modes.keys()),
            "has_custom_function": self.routing_function is not None,
            "routing_method": "custom"
        })
        return base_stats
    
    def __str__(self) -> str:
        return f"自定义路由器[{self.router_id}]: {self.name} - 模式: {self.current_mode}"
