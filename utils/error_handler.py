"""
增强的错误处理机制
提供细粒度错误分类、重试策略和降级机制
"""

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"  # 低严重程度，可以忽略或简单重试
    MEDIUM = "medium"  # 中等严重程度，需要重试或降级
    HIGH = "high"  # 高严重程度，需要特殊处理
    CRITICAL = "critical"  # 严重错误，需要立即停止


class ErrorCategory(Enum):
    """错误分类"""
    NETWORK = "network"  # 网络相关错误
    TIMEOUT = "timeout"  # 超时错误
    RESOURCE = "resource"  # 资源相关错误
    VALIDATION = "validation"  # 验证错误
    PERMISSION = "permission"  # 权限错误
    CONFIGURATION = "configuration"  # 配置错误
    EXTERNAL_SERVICE = "external_service"  # 外部服务错误
    INTERNAL = "internal"  # 内部错误
    UNKNOWN = "unknown"  # 未知错误


@dataclass
class ErrorInfo:
    """错误信息"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    context: Dict[str, Any] = None
    timestamp: float = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.context is None:
            self.context = {}


class RetryStrategy(ABC):
    """重试策略基类"""
    
    @abstractmethod
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """判断是否应该重试"""
        pass
    
    @abstractmethod
    def get_retry_delay(self, error_info: ErrorInfo) -> float:
        """获取重试延迟时间"""
        pass
    
    @abstractmethod
    def get_max_retries(self, error_info: ErrorInfo) -> int:
        """获取最大重试次数"""
        pass


class ExponentialBackoffStrategy(RetryStrategy):
    """指数退避重试策略"""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, multiplier: float = 2.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """判断是否应该重试"""
        # 严重错误不重试
        if error_info.severity == ErrorSeverity.CRITICAL:
            return False
        
        # 验证错误通常不重试
        if error_info.category == ErrorCategory.VALIDATION:
            return False
        
        # 权限错误不重试
        if error_info.category == ErrorCategory.PERMISSION:
            return False
        
        return error_info.retry_count < self.get_max_retries(error_info)
    
    def get_retry_delay(self, error_info: ErrorInfo) -> float:
        """获取重试延迟时间"""
        delay = self.base_delay * (self.multiplier ** error_info.retry_count)
        return min(delay, self.max_delay)
    
    def get_max_retries(self, error_info: ErrorInfo) -> int:
        """获取最大重试次数"""
        if error_info.severity == ErrorSeverity.LOW:
            return 5
        elif error_info.severity == ErrorSeverity.MEDIUM:
            return 3
        elif error_info.severity == ErrorSeverity.HIGH:
            return 1
        else:
            return 0


class LinearBackoffStrategy(RetryStrategy):
    """线性退避重试策略"""
    
    def __init__(self, base_delay: float = 2.0, max_retries: int = 3):
        self.base_delay = base_delay
        self.max_retries = max_retries
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """判断是否应该重试"""
        if error_info.severity == ErrorSeverity.CRITICAL:
            return False
        return error_info.retry_count < self.max_retries
    
    def get_retry_delay(self, error_info: ErrorInfo) -> float:
        """获取重试延迟时间"""
        return self.base_delay * (error_info.retry_count + 1)
    
    def get_max_retries(self, error_info: ErrorInfo) -> int:
        """获取最大重试次数"""
        return self.max_retries


class FixedDelayStrategy(RetryStrategy):
    """固定延迟重试策略"""
    
    def __init__(self, delay: float = 1.0, max_retries: int = 3):
        self.delay = delay
        self.max_retries = max_retries
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """判断是否应该重试"""
        if error_info.severity == ErrorSeverity.CRITICAL:
            return False
        return error_info.retry_count < self.max_retries
    
    def get_retry_delay(self, error_info: ErrorInfo) -> float:
        """获取重试延迟时间"""
        return self.delay
    
    def get_max_retries(self, error_info: ErrorInfo) -> int:
        """获取最大重试次数"""
        return self.max_retries


class ErrorClassifier:
    """错误分类器"""
    
    def __init__(self):
        self.error_mappings = {
            # 网络相关错误
            ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            TimeoutError: (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
            asyncio.TimeoutError: (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
            
            # 资源相关错误
            MemoryError: (ErrorCategory.RESOURCE, ErrorSeverity.HIGH),
            OSError: (ErrorCategory.RESOURCE, ErrorSeverity.MEDIUM),
            
            # 验证错误
            ValueError: (ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            TypeError: (ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            AttributeError: (ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            
            # 权限错误
            PermissionError: (ErrorCategory.PERMISSION, ErrorSeverity.HIGH),
            
            # 配置错误
            KeyError: (ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM),
            FileNotFoundError: (ErrorCategory.CONFIGURATION, ErrorSeverity.MEDIUM),
            
            # 外部服务错误
            Exception: (ErrorCategory.EXTERNAL_SERVICE, ErrorSeverity.MEDIUM),
        }
    
    def classify_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """分类错误"""
        error_id = f"err_{int(time.time() * 1000)}"
        
        # 查找匹配的错误类型
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        
        for error_type, (cat, sev) in self.error_mappings.items():
            if isinstance(exception, error_type):
                category = cat
                severity = sev
                break
        
        # 根据错误消息调整严重程度
        error_message = str(exception)
        if "critical" in error_message.lower() or "fatal" in error_message.lower():
            severity = ErrorSeverity.CRITICAL
        elif "warning" in error_message.lower() or "minor" in error_message.lower():
            severity = ErrorSeverity.LOW
        
        return ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=error_message,
            original_exception=exception,
            context=context or {}
        )


class FallbackHandler(ABC):
    """降级处理器基类"""
    
    @abstractmethod
    async def handle_fallback(self, error_info: ErrorInfo, original_func: Callable, *args, **kwargs) -> Any:
        """处理降级逻辑"""
        pass


class DefaultFallbackHandler(FallbackHandler):
    """默认降级处理器"""
    
    async def handle_fallback(self, error_info: ErrorInfo, original_func: Callable, *args, **kwargs) -> Any:
        """处理降级逻辑"""
        # 记录降级信息
        print(f"执行降级处理: {error_info.message}")
        
        # 返回默认值或抛出降级后的异常
        if error_info.category == ErrorCategory.NETWORK:
            return {"error": "网络不可用，使用缓存数据", "fallback": True}
        elif error_info.category == ErrorCategory.TIMEOUT:
            return {"error": "操作超时，返回部分结果", "fallback": True}
        else:
            raise Exception(f"降级处理失败: {error_info.message}")


class ErrorHandler:
    """错误处理器主类"""
    
    def __init__(
        self,
        retry_strategy: RetryStrategy = None,
        fallback_handler: FallbackHandler = None,
        enable_logging: bool = True
    ):
        """
        初始化错误处理器
        
        Args:
            retry_strategy: 重试策略
            fallback_handler: 降级处理器
            enable_logging: 是否启用日志记录
        """
        self.retry_strategy = retry_strategy or ExponentialBackoffStrategy()
        self.fallback_handler = fallback_handler or DefaultFallbackHandler()
        self.enable_logging = enable_logging
        self.error_classifier = ErrorClassifier()
        self.error_history: List[ErrorInfo] = []
    
    def handle_error(
        self,
        func: Callable,
        *args,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> Callable:
        """错误处理装饰器"""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._handle_async_error(func, *args, context=context, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._handle_sync_error(func, *args, context=context, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def _handle_async_error(
        self,
        func: Callable,
        *args,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> Any:
        """处理异步函数错误"""
        error_info = None
        
        for attempt in range(10):  # 最大尝试次数
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error_info = self.error_classifier.classify_error(e, context)
                error_info.retry_count = attempt
                
                if self.enable_logging:
                    self._log_error(error_info)
                
                # 检查是否应该重试
                if not self.retry_strategy.should_retry(error_info):
                    break
                
                # 等待重试
                delay = self.retry_strategy.get_retry_delay(error_info)
                await asyncio.sleep(delay)
        
        # 所有重试都失败了，尝试降级处理
        if error_info:
            self.error_history.append(error_info)
            return await self.fallback_handler.handle_fallback(error_info, func, *args, **kwargs)
        
        raise Exception("未知错误")
    
    def _handle_sync_error(
        self,
        func: Callable,
        *args,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> Any:
        """处理同步函数错误"""
        error_info = None
        
        for attempt in range(10):  # 最大尝试次数
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_info = self.error_classifier.classify_error(e, context)
                error_info.retry_count = attempt
                
                if self.enable_logging:
                    self._log_error(error_info)
                
                # 检查是否应该重试
                if not self.retry_strategy.should_retry(error_info):
                    break
                
                # 等待重试
                delay = self.retry_strategy.get_retry_delay(error_info)
                time.sleep(delay)
        
        # 所有重试都失败了，尝试降级处理
        if error_info:
            self.error_history.append(error_info)
            # 对于同步函数，我们需要在事件循环中运行降级处理
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(
                    self.fallback_handler.handle_fallback(error_info, func, *args, **kwargs)
                )
            except RuntimeError:
                # 没有事件循环，直接抛出异常
                raise Exception(f"降级处理失败: {error_info.message}")
        
        raise Exception("未知错误")
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """记录错误信息"""
        log_message = (
            f"错误 [{error_info.error_id}]: {error_info.category.value} - "
            f"{error_info.severity.value} - {error_info.message} "
            f"(重试 {error_info.retry_count}/{error_info.max_retries})"
        )
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            print(f"🚨 CRITICAL: {log_message}")
        elif error_info.severity == ErrorSeverity.HIGH:
            print(f"⚠️  HIGH: {log_message}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            print(f"⚠️  MEDIUM: {log_message}")
        else:
            print(f"ℹ️  LOW: {log_message}")
        
        if error_info.original_exception:
            print(f"原始异常: {traceback.format_exc()}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        if not self.error_history:
            return {"total_errors": 0}
        
        # 按类别统计
        category_stats = {}
        severity_stats = {}
        
        for error in self.error_history:
            category = error.category.value
            severity = error.severity.value
            
            category_stats[category] = category_stats.get(category, 0) + 1
            severity_stats[severity] = severity_stats.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "category_distribution": category_stats,
            "severity_distribution": severity_stats,
            "recent_errors": [
                {
                    "id": error.error_id,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "timestamp": error.timestamp,
                    "retry_count": error.retry_count
                }
                for error in self.error_history[-10:]  # 最近10个错误
            ]
        }
    
    def clear_error_history(self) -> None:
        """清除错误历史"""
        self.error_history.clear()
    
    def set_retry_strategy(self, strategy: RetryStrategy) -> None:
        """设置重试策略"""
        self.retry_strategy = strategy
    
    def set_fallback_handler(self, handler: FallbackHandler) -> None:
        """设置降级处理器"""
        self.fallback_handler = handler


# 全局错误处理器实例
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def set_error_handler(handler: ErrorHandler) -> None:
    """设置全局错误处理器"""
    global _global_error_handler
    _global_error_handler = handler


def handle_errors(
    retry_strategy: RetryStrategy = None,
    fallback_handler: FallbackHandler = None,
    context: Dict[str, Any] = None
):
    """错误处理装饰器工厂"""
    def decorator(func: Callable) -> Callable:
        handler = ErrorHandler(retry_strategy, fallback_handler)
        return handler.handle_error(func, context=context)
    return decorator


# 便捷装饰器
def with_retry(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    strategy = FixedDelayStrategy(delay=delay, max_retries=max_retries)
    return handle_errors(retry_strategy=strategy)


def with_exponential_backoff(base_delay: float = 1.0, max_delay: float = 60.0):
    """指数退避重试装饰器"""
    strategy = ExponentialBackoffStrategy(base_delay=base_delay, max_delay=max_delay)
    return handle_errors(retry_strategy=strategy)


def with_fallback(fallback_handler: FallbackHandler):
    """降级处理装饰器"""
    return handle_errors(fallback_handler=fallback_handler)
