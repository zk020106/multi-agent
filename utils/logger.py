"""
多智能体系统的日志记录工具。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """为不同日志级别提供颜色的自定义格式化器。"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 洋红色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    use_colors: bool = True
) -> None:
    """
    为多智能体系统设置日志配置。
    
    参数:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 可选的日志文件路径
        format_string: 自定义格式字符串
        use_colors: 是否使用彩色输出
    """
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    
    # 创建格式化器
    if use_colors and not log_file:
        formatter = ColoredFormatter(format_string)
    else:
        formatter = logging.Formatter(format_string)
    
    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取特定模块的日志记录器实例。
    
    参数:
        name: 日志记录器名称（通常是 __name__）
        
    返回:
        日志记录器实例
    """
    return logging.getLogger(name)


class AgentLogger:
    """具有额外上下文的智能体专用日志记录器。"""
    
    def __init__(self, agent_id: str, base_logger: Optional[logging.Logger] = None):
        self.agent_id = agent_id
        self.logger = base_logger or get_logger(f"agent.{agent_id}")
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """使用智能体上下文记录消息。"""
        context_message = f"[智能体: {self.agent_id}] {message}"
        self.logger.log(level, context_message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """记录调试消息。"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录信息消息。"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告消息。"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误消息。"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误消息。"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def task_start(self, task_id: str, task_title: str):
        """记录任务开始。"""
        self.info(f"开始任务: {task_id} - {task_title}")
    
    def task_complete(self, task_id: str, result_status: str):
        """记录任务完成。"""
        self.info(f"任务完成: {task_id} - 状态: {result_status}")
    
    def tool_call(self, tool_name: str, input_data: dict):
        """记录工具调用。"""
        self.debug(f"调用工具: {tool_name}，输入: {input_data}")
    
    def tool_result(self, tool_name: str, result: dict):
        """记录工具结果。"""
        self.debug(f"工具 {tool_name} 的结果: {result}")


# 初始化默认日志记录
setup_logging()
