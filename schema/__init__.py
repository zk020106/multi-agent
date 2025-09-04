"""
多智能体系统的模式模块。
定义消息、任务和结果的核心数据结构。
"""

from .message import Message, MessageType
from .task import Task, TaskStatus, TaskPriority
from .result import Result, ResultStatus

__all__ = [
    'Message', 'MessageType',
    'Task', 'TaskStatus', 'TaskPriority', 
    'Result', 'ResultStatus'
]
