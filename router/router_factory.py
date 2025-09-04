"""
路由器工厂类
用于创建不同类型的路由器实例
"""

from enum import Enum
from typing import Dict, List, Optional, Type, Any, Callable

from .base_router import BaseRouter
from .custom_router import CustomRouter
from .llm_router import LLMRouter
from .rule_router import RuleRouter


class RouterType(Enum):
    """路由器类型枚举"""
    RULE = "rule"
    LLM = "llm"
    CUSTOM = "custom"


class RouterFactory:
    """
    路由器工厂类
    
    负责创建和管理不同类型的路由器实例
    """
    
    # 路由器类型映射
    ROUTER_CLASSES: Dict[RouterType, Type[BaseRouter]] = {
        RouterType.RULE: RuleRouter,
        RouterType.LLM: LLMRouter,
        RouterType.CUSTOM: CustomRouter
    }
    
    def __init__(self):
        """初始化路由器工厂"""
        self._created_routers: Dict[str, BaseRouter] = {}
    
    def create_router(
        self,
        router_type: RouterType,
        router_id: str,
        name: str,
        description: str = ""
    ) -> BaseRouter:
        """
        创建路由器实例
        
        Args:
            router_type: 路由器类型
            router_id: 路由器ID
            name: 路由器名称
            description: 路由器描述
            **kwargs: 其他参数
            
        Returns:
            路由器实例
            
        Raises:
            ValueError: 如果路由器类型不支持
        """
        if router_type not in self.ROUTER_CLASSES:
            raise ValueError(f"不支持的路由器类型: {router_type}")
        
        # 检查ID是否已存在
        if router_id in self._created_routers:
            raise ValueError(f"路由器ID {router_id} 已存在")
        
        # 获取路由器类
        router_class = self.ROUTER_CLASSES[router_type]
        
        # 创建路由器实例
        router = router_class(
            router_id=router_id,
            name=name,
            description=description
        )
        
        # 注册到工厂
        self._created_routers[router_id] = router
        
        return router
    
    def create_rule_router(
        self,
        router_id: str,
        name: str = "规则路由器",
        description: str = "基于预定义规则的消息路由器",
        **kwargs
    ) -> RuleRouter:
        """
        创建规则路由器
        
        Args:
            router_id: 路由器ID
            name: 路由器名称
            description: 路由器描述
            **kwargs: 其他参数
            
        Returns:
            规则路由器实例
        """
        return self.create_router(
            router_type=RouterType.RULE,
            router_id=router_id,
            name=name,
            description=description
        )
    
    def create_llm_router(
        self,
        router_id: str,
        llm,
        name: str = "LLM智能路由器",
        description: str = "基于大语言模型的智能消息路由器",
        **kwargs
    ) -> LLMRouter:
        """
        创建LLM路由器
        
        Args:
            router_id: 路由器ID
            llm: 大语言模型实例
            name: 路由器名称
            description: 路由器描述
            **kwargs: 其他参数
            
        Returns:
            LLM路由器实例
        """
        return self.create_router(
            router_type=RouterType.LLM,
            router_id=router_id,
            name=name,
            description=description
        )
    
    def create_custom_router(
        self,
        router_id: str,
        name: str = "自定义路由器",
        description: str = "支持自定义路由逻辑的路由器",
        routing_function: Optional[Callable] = None,
        **kwargs
    ) -> CustomRouter:
        """
        创建自定义路由器
        
        Args:
            router_id: 路由器ID
            name: 路由器名称
            description: 路由器描述
            routing_function: 自定义路由函数
            **kwargs: 其他参数
            
        Returns:
            自定义路由器实例
        """
        return self.create_router(
            router_type=RouterType.CUSTOM,
            router_id=router_id,
            name=name,
            description=description
        )
    
    def get_router(self, router_id: str) -> Optional[BaseRouter]:
        """
        获取路由器实例
        
        Args:
            router_id: 路由器ID
            
        Returns:
            路由器实例或None
        """
        return self._created_routers.get(router_id)
    
    def remove_router(self, router_id: str) -> bool:
        """
        移除路由器
        
        Args:
            router_id: 路由器ID
            
        Returns:
            是否成功移除
        """
        if router_id in self._created_routers:
            router_name = self._created_routers[router_id].name
            del self._created_routers[router_id]
            return True
        return False
    
    def list_routers(self) -> List[str]:
        """
        列出所有路由器ID
        
        Returns:
            路由器ID列表
        """
        return list(self._created_routers.keys())
    
    def get_router_info(self, router_id: str) -> Optional[Dict[str, Any]]:
        """
        获取路由器信息
        
        Args:
            router_id: 路由器ID
            
        Returns:
            路由器信息字典或None
        """
        router = self.get_router(router_id)
        if router:
            return router.get_routing_stats()
        return None
    
    def get_all_routers_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有路由器信息
        
        Returns:
            路由器信息字典
        """
        return {
            router_id: router.get_routing_stats()
            for router_id, router in self._created_routers.items()
        }
    
    def get_routers_by_type(self, router_type: RouterType) -> List[BaseRouter]:
        """
        根据类型获取路由器列表
        
        Args:
            router_type: 路由器类型
            
        Returns:
            路由器实例列表
        """
        return [
            router for router in self._created_routers.values()
            if isinstance(router, self.ROUTER_CLASSES[router_type])
        ]
    
    def get_router_count(self) -> int:
        """
        获取路由器总数
        
        Returns:
            路由器数量
        """
        return len(self._created_routers)
    
    def get_router_type_count(self) -> Dict[RouterType, int]:
        """
        获取各类型路由器数量
        
        Returns:
            各类型路由器数量字典
        """
        type_count = {}
        for router_type in RouterType:
            type_count[router_type] = len(self.get_routers_by_type(router_type))
        return type_count
    
    def clear_all_routers(self) -> int:
        """
        清除所有路由器
        
        Returns:
            清除的路由器数量
        """
        count = len(self._created_routers)
        self._created_routers.clear()
        return count
    
    def register_agent_to_all_routers(
        self,
        agent,
        capabilities: List[str] = None,
        role: str = "general"
    ) -> int:
        """
        将智能体注册到所有路由器
        
        Args:
            agent: 智能体实例
            capabilities: 智能体能力列表
            role: 智能体角色
            
        Returns:
            注册的路由器数量
        """
        count = 0
        for router in self._created_routers.values():
            router.register_agent(agent, capabilities, role)
            count += 1
        return count
    
    def unregister_agent_from_all_routers(self, agent_id: str) -> int:
        """
        从所有路由器注销智能体
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            注销的路由器数量
        """
        count = 0
        for router in self._created_routers.values():
            if router.unregister_agent(agent_id):
                count += 1
        return count
    
    def route_message_through_all_routers(self, message) -> Dict[str, List[str]]:
        """
        通过所有路由器路由消息
        
        Args:
            message: 消息对象
            
        Returns:
            各路由器的路由结果
        """
        results = {}
        for router_id, router in self._created_routers.items():
            try:
                target_agents = router.route_message(message)
                results[router_id] = target_agents
            except Exception as e:
                results[router_id] = []
        return results
    
    def get_combined_routing_stats(self) -> Dict[str, Any]:
        """
        获取所有路由器的综合统计信息
        
        Returns:
            综合统计信息
        """
        total_routes = 0
        total_successful = 0
        total_failed = 0
        total_time = 0.0
        
        for router in self._created_routers.values():
            stats = router.get_routing_stats()
            total_routes += stats.get("total_routes", 0)
            total_successful += stats.get("successful_routes", 0)
            total_failed += stats.get("failed_routes", 0)
            total_time += stats.get("average_routing_time", 0.0) * stats.get("total_routes", 0)
        
        return {
            "total_routers": len(self._created_routers),
            "total_routes": total_routes,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": total_successful / max(1, total_routes) * 100,
            "average_routing_time": total_time / max(1, total_routes),
            "router_types": self.get_router_type_count()
        }


# 全局路由器工厂实例
_global_factory = RouterFactory()


def get_router_factory() -> RouterFactory:
    """获取全局路由器工厂实例"""
    return _global_factory


def create_router(router_type: RouterType, **kwargs) -> BaseRouter:
    """使用全局工厂创建路由器"""
    return _global_factory.create_router(router_type)


def get_router(router_id: str) -> Optional[BaseRouter]:
    """从全局工厂获取路由器"""
    return _global_factory.get_router(router_id)
