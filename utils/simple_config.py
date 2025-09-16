"""
简化的配置管理系统
提供更直观的配置接口和验证机制
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import yaml


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(Enum):
    """数据库类型"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    REDIS = "redis"


class MemoryType(Enum):
    """内存类型"""
    IN_MEMORY = "in_memory"
    REDIS = "redis"
    VECTOR = "vector"


class CoordinatorType(Enum):
    """协调器类型"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    VOTING = "voting"


@dataclass
class SimpleConfig:
    """简化的配置类"""
    
    # 系统基础设置
    system_name: str = "多智能体系统"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    
    # 数据库设置
    database_type: DatabaseType = DatabaseType.SQLITE
    database_url: str = "sqlite:///multi_agent.db"
    
    # LLM设置
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_api_key: str = ""
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    llm_timeout: int = 30
    
    # 内存设置
    memory_type: MemoryType = MemoryType.IN_MEMORY
    memory_redis_url: str = "redis://localhost:6379"
    memory_vector_url: str = "http://localhost:19530"
    memory_max_size: int = 1000
    
    # 协调器设置
    coordinator_type: CoordinatorType = CoordinatorType.SEQUENTIAL
    max_parallel_tasks: int = 5
    task_timeout: int = 300
    retry_attempts: int = 3
    
    # 智能体设置
    max_agents: int = 10
    agent_timeout: int = 60
    
    # 工具设置
    tool_timeout: int = 30
    max_tool_retries: int = 3
    
    # 自定义设置
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        # 验证日志级别
        if not isinstance(self.log_level, LogLevel):
            errors.append(f"无效的日志级别: {self.log_level}")
        
        # 验证数据库类型
        if not isinstance(self.database_type, DatabaseType):
            errors.append(f"无效的数据库类型: {self.database_type}")
        
        # 验证内存类型
        if not isinstance(self.memory_type, MemoryType):
            errors.append(f"无效的内存类型: {self.memory_type}")
        
        # 验证协调器类型
        if not isinstance(self.coordinator_type, CoordinatorType):
            errors.append(f"无效的协调器类型: {self.coordinator_type}")
        
        # 验证数值范围
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            errors.append(f"LLM温度必须在0-2之间: {self.llm_temperature}")
        
        if self.llm_max_tokens <= 0:
            errors.append(f"LLM最大令牌数必须大于0: {self.llm_max_tokens}")
        
        if self.max_agents <= 0:
            errors.append(f"最大智能体数必须大于0: {self.max_agents}")
        
        if self.max_parallel_tasks <= 0:
            errors.append(f"最大并行任务数必须大于0: {self.max_parallel_tasks}")
        
        # 验证URL格式
        if self.database_url and not self._is_valid_url(self.database_url):
            errors.append(f"无效的数据库URL: {self.database_url}")
        
        if self.memory_redis_url and not self._is_valid_url(self.memory_redis_url):
            errors.append(f"无效的Redis URL: {self.memory_redis_url}")
        
        return errors
    
    def _is_valid_url(self, url: str) -> bool:
        """简单的URL验证"""
        return url.startswith(('http://', 'https://', 'sqlite://', 'postgresql://', 'mysql://', 'redis://'))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleConfig':
        """从字典创建配置"""
        config = cls()
        
        # 系统设置
        config.system_name = data.get('system_name', config.system_name)
        config.debug = data.get('debug', config.debug)
        config.log_level = LogLevel(data.get('log_level', config.log_level.value))
        config.log_file = data.get('log_file', config.log_file)
        
        # 数据库设置
        config.database_type = DatabaseType(data.get('database_type', config.database_type.value))
        config.database_url = data.get('database_url', config.database_url)
        
        # LLM设置
        config.llm_provider = data.get('llm_provider', config.llm_provider)
        config.llm_model = data.get('llm_model', config.llm_model)
        config.llm_api_key = data.get('llm_api_key', config.llm_api_key)
        config.llm_base_url = data.get('llm_base_url', config.llm_base_url)
        config.llm_temperature = data.get('llm_temperature', config.llm_temperature)
        config.llm_max_tokens = data.get('llm_max_tokens', config.llm_max_tokens)
        config.llm_timeout = data.get('llm_timeout', config.llm_timeout)
        
        # 内存设置
        config.memory_type = MemoryType(data.get('memory_type', config.memory_type.value))
        config.memory_redis_url = data.get('memory_redis_url', config.memory_redis_url)
        config.memory_vector_url = data.get('memory_vector_url', config.memory_vector_url)
        config.memory_max_size = data.get('memory_max_size', config.memory_max_size)
        
        # 协调器设置
        config.coordinator_type = CoordinatorType(data.get('coordinator_type', config.coordinator_type.value))
        config.max_parallel_tasks = data.get('max_parallel_tasks', config.max_parallel_tasks)
        config.task_timeout = data.get('task_timeout', config.task_timeout)
        config.retry_attempts = data.get('retry_attempts', config.retry_attempts)
        
        # 智能体设置
        config.max_agents = data.get('max_agents', config.max_agents)
        config.agent_timeout = data.get('agent_timeout', config.agent_timeout)
        
        # 工具设置
        config.tool_timeout = data.get('tool_timeout', config.tool_timeout)
        config.max_tool_retries = data.get('max_tool_retries', config.max_tool_retries)
        
        # 自定义设置
        config.custom_settings = data.get('custom_settings', config.custom_settings)
        
        return config


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Optional[SimpleConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置"""
        if self.config_path and Path(self.config_path).exists():
            self.config = self.load_from_file(self.config_path)
        else:
            self.config = SimpleConfig()
    
    def load_from_file(self, file_path: Union[str, Path]) -> SimpleConfig:
        """
        从文件加载配置
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            配置对象
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
        
        config = SimpleConfig.from_dict(data)
        
        # 验证配置
        errors = config.validate()
        if errors:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
        
        return config
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        保存配置到文件
        
        Args:
            file_path: 保存路径
        """
        if not self.config:
            raise ValueError("没有配置可保存")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.config.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, indent=2, allow_unicode=True)
            elif file_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_path.suffix}")
    
    def load_from_env(self) -> SimpleConfig:
        """
        从环境变量加载配置
        
        Returns:
            配置对象
        """
        config = SimpleConfig()
        
        # 系统设置
        config.system_name = os.getenv('MAS_SYSTEM_NAME', config.system_name)
        config.debug = os.getenv('MAS_DEBUG', 'false').lower() == 'true'
        config.log_level = LogLevel(os.getenv('MAS_LOG_LEVEL', config.log_level.value))
        config.log_file = os.getenv('MAS_LOG_FILE', config.log_file)
        
        # 数据库设置
        config.database_type = DatabaseType(os.getenv('MAS_DB_TYPE', config.database_type.value))
        config.database_url = os.getenv('MAS_DB_URL', config.database_url)
        
        # LLM设置
        config.llm_provider = os.getenv('MAS_LLM_PROVIDER', config.llm_provider)
        config.llm_model = os.getenv('MAS_LLM_MODEL', config.llm_model)
        config.llm_api_key = os.getenv('MAS_LLM_API_KEY', config.llm_api_key)
        config.llm_base_url = os.getenv('MAS_LLM_BASE_URL', config.llm_base_url)
        config.llm_temperature = float(os.getenv('MAS_LLM_TEMPERATURE', str(config.llm_temperature)))
        config.llm_max_tokens = int(os.getenv('MAS_LLM_MAX_TOKENS', str(config.llm_max_tokens)))
        config.llm_timeout = int(os.getenv('MAS_LLM_TIMEOUT', str(config.llm_timeout)))
        
        # 内存设置
        config.memory_type = MemoryType(os.getenv('MAS_MEMORY_TYPE', config.memory_type.value))
        config.memory_redis_url = os.getenv('MAS_MEMORY_REDIS_URL', config.memory_redis_url)
        config.memory_vector_url = os.getenv('MAS_MEMORY_VECTOR_URL', config.memory_vector_url)
        config.memory_max_size = int(os.getenv('MAS_MEMORY_MAX_SIZE', str(config.memory_max_size)))
        
        # 协调器设置
        config.coordinator_type = CoordinatorType(os.getenv('MAS_COORDINATOR_TYPE', config.coordinator_type.value))
        config.max_parallel_tasks = int(os.getenv('MAS_MAX_PARALLEL_TASKS', str(config.max_parallel_tasks)))
        config.task_timeout = int(os.getenv('MAS_TASK_TIMEOUT', str(config.task_timeout)))
        config.retry_attempts = int(os.getenv('MAS_RETRY_ATTEMPTS', str(config.retry_attempts)))
        
        # 智能体设置
        config.max_agents = int(os.getenv('MAS_MAX_AGENTS', str(config.max_agents)))
        config.agent_timeout = int(os.getenv('MAS_AGENT_TIMEOUT', str(config.agent_timeout)))
        
        # 工具设置
        config.tool_timeout = int(os.getenv('MAS_TOOL_TIMEOUT', str(config.tool_timeout)))
        config.max_tool_retries = int(os.getenv('MAS_MAX_TOOL_RETRIES', str(config.max_tool_retries)))
        
        return config
    
    def get_config(self) -> SimpleConfig:
        """获取当前配置"""
        if not self.config:
            self.config = SimpleConfig()
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        更新配置
        
        Args:
            **kwargs: 要更新的配置项
        """
        if not self.config:
            self.config = SimpleConfig()
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.custom_settings[key] = value
        
        # 验证更新后的配置
        errors = self.config.validate()
        if errors:
            raise ValueError(f"配置更新后验证失败: {'; '.join(errors)}")
    
    def reset_to_defaults(self) -> None:
        """重置为默认配置"""
        self.config = SimpleConfig()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        if not self.config:
            return {"status": "no_config"}
        
        return {
            "system_name": self.config.system_name,
            "debug": self.config.debug,
            "log_level": self.config.log_level.value,
            "database_type": self.config.database_type.value,
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.llm_model,
            "memory_type": self.config.memory_type.value,
            "coordinator_type": self.config.coordinator_type.value,
            "max_agents": self.config.max_agents,
            "max_parallel_tasks": self.config.max_parallel_tasks,
            "custom_settings_count": len(self.config.custom_settings)
        }


# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def set_config_manager(manager: ConfigManager) -> None:
    """设置全局配置管理器"""
    global _global_config_manager
    _global_config_manager = manager


def get_config() -> SimpleConfig:
    """获取当前配置"""
    return get_config_manager().get_config()


def load_config_from_file(file_path: Union[str, Path]) -> SimpleConfig:
    """从文件加载配置"""
    return get_config_manager().load_from_file(file_path)


def save_config_to_file(config: SimpleConfig, file_path: Union[str, Path]) -> None:
    """保存配置到文件"""
    manager = get_config_manager()
    manager.config = config
    manager.save_to_file(file_path)


def load_config_from_env() -> SimpleConfig:
    """从环境变量加载配置"""
    return get_config_manager().load_from_env()


# 便捷函数
def create_default_config_file(file_path: Union[str, Path], format: str = "yaml") -> None:
    """
    创建默认配置文件
    
    Args:
        file_path: 文件路径
        format: 文件格式 (yaml, json)
    """
    config = SimpleConfig()
    file_path = Path(file_path)
    
    if format.lower() == "yaml":
        file_path = file_path.with_suffix('.yaml')
    elif format.lower() == "json":
        file_path = file_path.with_suffix('.json')
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    save_config_to_file(config, file_path)
    print(f"默认配置文件已创建: {file_path}")


def validate_config_file(file_path: Union[str, Path]) -> List[str]:
    """
    验证配置文件
    
    Args:
        file_path: 配置文件路径
        
    Returns:
        错误列表，空列表表示验证通过
    """
    try:
        config = load_config_from_file(file_path)
        return config.validate()
    except Exception as e:
        return [str(e)]
