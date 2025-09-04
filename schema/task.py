"""
多智能体系统的任务模式定义。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """任务执行状态。"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级级别。"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Task:
    """
    表示智能体要执行的工作的任务对象。
    
    属性:
        id: 唯一任务标识符
        title: 任务标题/描述
        description: 详细任务描述
        status: 当前任务状态
        priority: 任务优先级级别
        assigned_agent: 分配给此任务的智能体ID
        created_at: 任务创建时间
        updated_at: 任务最后更新时间
        completed_at: 任务完成时间
        dependencies: 此任务依赖的任务ID列表
        metadata: 额外的任务元数据
        result: 任务执行结果
    """
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    
    def update_status(self, status: TaskStatus) -> None:
        """更新任务状态和时间戳。"""
        self.status = status
        self.updated_at = datetime.now()
        if status == TaskStatus.COMPLETED:
            self.completed_at = datetime.now()
    
    def assign_to_agent(self, agent_id: str) -> None:
        """将任务分配给智能体。"""
        self.assigned_agent = agent_id
        self.updated_at = datetime.now()
    
    def add_dependency(self, task_id: str) -> None:
        """添加任务依赖。"""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
            self.updated_at = datetime.now()
    
    def is_ready_to_execute(self, completed_tasks: List[str]) -> bool:
        """检查任务是否准备好执行（所有依赖都已完成）。"""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def to_dict(self) -> Dict[str, Any]:
        """将任务转换为字典。"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority.value,
            'assigned_agent': self.assigned_agent,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
            'result': self.result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建任务。"""
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            status=TaskStatus(data['status']),
            priority=TaskPriority(data['priority']),
            assigned_agent=data.get('assigned_agent'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            dependencies=data.get('dependencies', []),
            metadata=data.get('metadata', {}),
            result=data.get('result')
        )
    
    def __str__(self) -> str:
        return f"[{self.status.value}] {self.title} (优先级: {self.priority.value})"
