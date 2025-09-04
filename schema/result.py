"""
多智能体系统的结果模式定义。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ResultStatus(Enum):
    """结果状态类型。"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class Result:
    """
    表示任务执行结果的结果对象。
    
    属性:
        id: 唯一结果标识符
        task_id: 此结果所属任务的ID
        agent_id: 产生此结果的智能体ID
        status: 结果状态
        data: 结果数据/内容
        error_message: 执行失败时的错误消息
        execution_time: 执行任务所需时间
        created_at: 结果创建时间
        metadata: 额外的结果元数据
        logs: 执行日志
    """
    id: str
    task_id: str
    agent_id: str
    status: ResultStatus
    data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    
    def add_log(self, log_message: str) -> None:
        """向结果添加日志条目。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {log_message}")
    
    def set_error(self, error_message: str) -> None:
        """设置错误状态和消息。"""
        self.status = ResultStatus.ERROR
        self.error_message = error_message
        self.add_log(f"错误: {error_message}")
    
    def set_success(self, data: Any = None) -> None:
        """设置成功状态和数据。"""
        self.status = ResultStatus.SUCCESS
        self.data = data
        self.add_log("任务成功完成")
    
    def set_partial_success(self, data: Any = None, error_message: str = None) -> None:
        """设置部分成功状态。"""
        self.status = ResultStatus.PARTIAL
        self.data = data
        if error_message:
            self.error_message = error_message
            self.add_log(f"部分成功: {error_message}")
        else:
            self.add_log("任务部分成功完成")
    
    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典。"""
        return {
            'id': self.id,
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'status': self.status.value,
            'data': self.data,
            'error_message': self.error_message,
            'execution_time': self.execution_time,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'logs': self.logs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Result':
        """从字典创建结果。"""
        return cls(
            id=data['id'],
            task_id=data['task_id'],
            agent_id=data['agent_id'],
            status=ResultStatus(data['status']),
            data=data.get('data'),
            error_message=data.get('error_message'),
            execution_time=data.get('execution_time'),
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {}),
            logs=data.get('logs', [])
        )
    
    def __str__(self) -> str:
        status_emoji = {
            ResultStatus.SUCCESS: "✅",
            ResultStatus.FAILURE: "❌",
            ResultStatus.PARTIAL: "⚠️",
            ResultStatus.ERROR: "🚨"
        }
        return f"{status_emoji.get(self.status, '❓')} [{self.status.value}] 任务 {self.task_id} 由 {self.agent_id} 执行"
