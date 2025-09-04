"""
多智能体系统的消息模式定义。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(Enum):
    """系统中的消息类型。"""
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR = "error"


@dataclass
class Message:
    """
    表示智能体和组件之间通信的消息对象。
    
    属性:
        id: 唯一消息标识符
        sender: 发送者ID（智能体、用户、系统）
        receiver: 接收者ID（智能体、系统）
        content: 消息内容
        message_type: 消息类型
        timestamp: 消息创建时间
        metadata: 额外的消息元数据
        conversation_id: 此消息所属对话的ID
    """
    id: str
    sender: str
    receiver: str
    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典。"""
        return {
            'id': self.id,
            'sender': self.sender,
            'receiver': self.receiver,
            'content': self.content,
            'message_type': self.message_type.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'conversation_id': self.conversation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息。"""
        return cls(
            id=data['id'],
            sender=data['sender'],
            receiver=data['receiver'],
            content=data['content'],
            message_type=MessageType(data['message_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
            conversation_id=data.get('conversation_id')
        )
    
    def __str__(self) -> str:
        return f"[{self.message_type.value}] {self.sender} -> {self.receiver}: {self.content[:50]}..."
