"""
多智能体协调器模块
负责智能体之间的任务分配和结果聚合
"""

from .base_coordinator import BaseCoordinator
from .sequential_coordinator import SequentialCoordinator

__all__ = [
    'BaseCoordinator',
    'SequentialCoordinator',
]
