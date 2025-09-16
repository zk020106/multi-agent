"""
多智能体系统的配置管理。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class DatabaseConfig:
    """数据库配置。"""
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "multi_agent"
    username: str = ""
    password: str = ""
    url: Optional[str] = None


@dataclass
class LLMConfig:
    """LLM配置。"""
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-3.5-turbo"
    api_key: str = ""
    # 建议到 /v1 为止，不要包含 /chat/completions 或 /completions
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30


@dataclass
class MemoryConfig:
    """内存存储配置。"""
    type: str = "in_memory"  # in_memory, redis, vector
    redis_url: str = "redis://localhost:6379"
    vector_db_url: str = "http://localhost:19530"
    max_memory_size: int = 1000


@dataclass
class CoordinatorConfig:
    """协调器配置。"""
    type: str = "sequential"  # sequential, parallel, voting
    max_parallel_tasks: int = 5
    timeout: int = 300
    retry_attempts: int = 3


@dataclass
class Config:
    """多智能体系统的主配置类。"""

    # 系统设置
    system_name: str = "多智能体系统"
    debug: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # 组件配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)

    # 智能体设置
    max_agents: int = 10
    agent_timeout: int = 60
    max_parallel_tasks: int = 10

    # 工具设置
    tool_timeout: int = 30
    max_tool_retries: int = 3

    # 自定义设置
    custom: Dict[str, Any] = field(default_factory=dict)
    # 第三方集成
    serper_api_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典。"""
        return {
            'system_name': self.system_name,
            'debug': self.debug,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'database': {
                'type': self.database.type,
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                'username': self.database.username,
                'password': self.database.password,
                'url': self.database.url
            },
            'llm': {
                'provider': self.llm.provider,
                'model': self.llm.model,
                'api_key': self.llm.api_key,
                'base_url': self.llm.base_url,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
                'timeout': self.llm.timeout
            },
            'memory': {
                'type': self.memory.type,
                'redis_url': self.memory.redis_url,
                'vector_db_url': self.memory.vector_db_url,
                'max_memory_size': self.memory.max_memory_size
            },
            'coordinator': {
                'type': self.coordinator.type,
                'max_parallel_tasks': self.coordinator.max_parallel_tasks,
                'timeout': self.coordinator.timeout,
                'retry_attempts': self.coordinator.retry_attempts
            },
            'max_agents': self.max_agents,
            'agent_timeout': self.agent_timeout,
            'tool_timeout': self.tool_timeout,
            'max_tool_retries': self.max_tool_retries,
            'max_parallel_tasks': self.max_parallel_tasks,
            'custom': self.custom,
            'serper_api_key': self.serper_api_key
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """从字典创建配置。"""
        config = cls()

        # 系统设置
        config.system_name = data.get('system_name', config.system_name)
        config.debug = data.get('debug', config.debug)
        config.log_level = data.get('log_level', config.log_level)
        config.log_file = data.get('log_file', config.log_file)

        # 数据库配置
        if 'database' in data:
            db_data = data['database']
            config.database = DatabaseConfig(
                type=db_data.get('type', config.database.type),
                host=db_data.get('host', config.database.host),
                port=db_data.get('port', config.database.port),
                database=db_data.get('database', config.database.database),
                username=db_data.get('username', config.database.username),
                password=db_data.get('password', config.database.password),
                url=db_data.get('url', config.database.url)
            )

        # LLM配置
        if 'llm' in data:
            llm_data = data['llm']
            config.llm = LLMConfig(
                provider=llm_data.get('provider', config.llm.provider),
                model=llm_data.get('model', config.llm.model),
                api_key=llm_data.get('api_key', config.llm.api_key),
                base_url=llm_data.get('base_url', config.llm.base_url),
                temperature=llm_data.get('temperature', config.llm.temperature),
                max_tokens=llm_data.get('max_tokens', config.llm.max_tokens),
                timeout=llm_data.get('timeout', config.llm.timeout)
            )

        # 内存配置
        if 'memory' in data:
            mem_data = data['memory']
            config.memory = MemoryConfig(
                type=mem_data.get('type', config.memory.type),
                redis_url=mem_data.get('redis_url', config.memory.redis_url),
                vector_db_url=mem_data.get('vector_db_url', config.memory.vector_db_url),
                max_memory_size=mem_data.get('max_memory_size', config.memory.max_memory_size)
            )

        # 协调器配置
        if 'coordinator' in data:
            coord_data = data['coordinator']
            config.coordinator = CoordinatorConfig(
                type=coord_data.get('type', config.coordinator.type),
                max_parallel_tasks=coord_data.get('max_parallel_tasks', config.coordinator.max_parallel_tasks),
                timeout=coord_data.get('timeout', config.coordinator.timeout),
                retry_attempts=coord_data.get('retry_attempts', config.coordinator.retry_attempts)
            )

        # 其他设置
        config.max_agents = data.get('max_agents', config.max_agents)
        config.agent_timeout = data.get('agent_timeout', config.agent_timeout)
        config.tool_timeout = data.get('tool_timeout', config.tool_timeout)
        config.max_tool_retries = data.get('max_tool_retries', config.max_tool_retries)
        config.custom = data.get('custom', config.custom)
        # 第三方集成
        config.serper_api_key = data.get('serper_api_key', config.serper_api_key)

        return config


def load_config(config_path: Union[str, Path]) -> Config:
    """
    从文件加载配置。
    
    参数:
        config_path: 配置文件路径（JSON或YAML）
        
    返回:
        配置对象
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

    return Config.from_dict(data)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    将配置保存到文件。
    
    参数:
        config: 要保存的配置对象
        config_path: 保存配置文件的路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")


def load_config_from_env() -> Config:
    """
    从环境变量加载配置。
    
    返回:
        从环境变量获取值的配置对象
    """
    config = Config()

    # 系统设置
    config.system_name = os.getenv('MAS_SYSTEM_NAME', config.system_name)
    config.debug = os.getenv('MAS_DEBUG', 'false').lower() == 'true'
    config.log_level = os.getenv('MAS_LOG_LEVEL', config.log_level)
    config.log_file = os.getenv('MAS_LOG_FILE', config.log_file)

    # 数据库设置
    config.database.type = os.getenv('MAS_DB_TYPE', config.database.type)
    config.database.host = os.getenv('MAS_DB_HOST', config.database.host)
    config.database.port = int(os.getenv('MAS_DB_PORT', str(config.database.port)))
    config.database.database = os.getenv('MAS_DB_NAME', config.database.database)
    config.database.username = os.getenv('MAS_DB_USER', config.database.username)
    config.database.password = os.getenv('MAS_DB_PASSWORD', config.database.password)
    config.database.url = os.getenv('MAS_DB_URL', config.database.url)

    # LLM设置
    config.llm.provider = os.getenv('MAS_LLM_PROVIDER', config.llm.provider)
    config.llm.model = os.getenv('MAS_LLM_MODEL', config.llm.model)
    config.llm.api_key = os.getenv('MAS_LLM_API_KEY', config.llm.api_key)
    config.llm.base_url = os.getenv('MAS_LLM_BASE_URL', config.llm.base_url)
    config.llm.temperature = float(os.getenv('MAS_LLM_TEMPERATURE', str(config.llm.temperature)))
    config.llm.max_tokens = int(os.getenv('MAS_LLM_MAX_TOKENS', str(config.llm.max_tokens)))
    config.llm.timeout = int(os.getenv('MAS_LLM_TIMEOUT', str(config.llm.timeout)))

    return config


# 全局配置实例
_global_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例。"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config) -> None:
    """设置全局配置实例。"""
    global _global_config
    _global_config = config
