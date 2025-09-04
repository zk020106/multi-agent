"""
多智能体系统的基础工具类。
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ToolStatus(Enum):
    """工具执行状态。"""
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


@dataclass
class ToolResult:
    """工具执行结果。"""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """
    多智能体系统中所有工具的抽象基类。
    
    所有工具必须实现execute方法并定义其
    输入模式和输出模式。
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.input_schema: Dict[str, Any] = {}
        self.output_schema: Dict[str, Any] = {}
        self.required_params: List[str] = []
        self.optional_params: List[str] = []
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        使用给定参数执行工具。
        
        参数:
            **kwargs: 工具特定参数
            
        返回:
            包含执行结果的ToolResult对象
        """
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """
        验证输入参数。
        
        参数:
            **kwargs: 要验证的输入参数
            
        返回:
            如果有效则返回True，否则返回False
        """
        # 检查必需参数
        for param in self.required_params:
            if param not in kwargs:
                return False
        
        return True
    
    def get_schema(self) -> Dict[str, Any]:
        """
        获取工具模式信息。
        
        返回:
            包含工具模式的字典
        """
        return {
            'name': self.name,
            'description': self.description,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'required_params': self.required_params,
            'optional_params': self.optional_params
        }
    
    def run_with_metrics(self, **kwargs) -> ToolResult:
        """
        使用执行指标运行工具。
        
        参数:
            **kwargs: 工具参数
            
        返回:
            包含执行指标的ToolResult
        """
        start_time = time.time()
        
        try:
            # 验证输入
            if not self.validate_input(**kwargs):
                return ToolResult(
                    success=False,
                    error_message=f"无效的输入参数。必需参数: {self.required_params}"
                )
            
            # 执行工具
            result = self.execute(**kwargs)
            result.execution_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def __str__(self) -> str:
        return f"工具: {self.name} - {self.description}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class ToolRegistry:
    """管理可用工具的注册表。"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """
        注册工具。
        
        参数:
            tool: 要注册的工具实例
        """
        self._tools[tool.name] = tool
    
    def unregister(self, tool_name: str) -> bool:
        """
        注销工具。
        
        参数:
            tool_name: 要注销的工具名称
            
        返回:
            如果工具被注销则返回True，如果未找到则返回False
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        按名称获取工具。
        
        参数:
            tool_name: 工具名称
            
        返回:
            工具实例，如果未找到则返回None
        """
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        列出所有已注册的工具名称。
        
        返回:
            工具名称列表
        """
        return list(self._tools.keys())
    
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有已注册工具的模式。
        
        返回:
            将工具名称映射到其模式的字典
        """
        return {name: tool.get_schema() for name, tool in self._tools.items()}
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        按名称执行工具。
        
        参数:
            tool_name: 要执行的工具名称
            **kwargs: 工具参数
            
        返回:
            ToolResult对象
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error_message=f"未找到工具 '{tool_name}'"
            )
        
        return tool.run_with_metrics(**kwargs)


# 全局工具注册表
_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表。"""
    return _global_registry


def register_tool(tool: BaseTool) -> None:
    """在全局注册表中注册工具。"""
    _global_registry.register(tool)


def get_tool(tool_name: str) -> Optional[BaseTool]:
    """从全局注册表获取工具。"""
    return _global_registry.get_tool(tool_name)


def execute_tool(tool_name: str, **kwargs) -> ToolResult:
    """从全局注册表执行工具。"""
    return _global_registry.execute_tool(tool_name, **kwargs)
