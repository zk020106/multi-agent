"""
基于Redis的记忆组件
使用Redis存储对话记忆
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis
from langchain_core.memory import BaseMemory

from schema import Message, MessageType
from utils.logger import get_logger


class RedisMemory(BaseMemory):
    """
    Redis记忆组件
    
    使用Redis存储对话记忆，支持持久化和分布式访问
    """
    
    def __init__(
        self,
        agent_id: str,
        redis_url: str = "redis://localhost:6379",
        memory_key: str = "chat_history",
        ttl: int = 3600  # 1小时过期
    ):
        """
        初始化Redis记忆
        
        Args:
            agent_id: 智能体ID
            redis_url: Redis连接URL
            memory_key: 记忆键名
            ttl: 过期时间（秒）
        """
        super().__init__()
        
        self.agent_id = agent_id
        self.memory_key = memory_key
        self.ttl = ttl
        self.logger = get_logger(f"redis_memory.{agent_id}")
        
        # 连接Redis
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # 测试连接
            self.redis_client.ping()
            self.logger.info(f"Redis连接成功: {redis_url}")
        except Exception as e:
            self.logger.error(f"Redis连接失败: {str(e)}")
            raise
        
        # 构建键名
        self.chat_key = f"agent:{agent_id}:chat_history"
        self.metadata_key = f"agent:{agent_id}:metadata"
    
    @property
    def memory_variables(self) -> List[str]:
        """返回记忆变量列表"""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        加载记忆变量
        
        Args:
            inputs: 输入变量
            
        Returns:
            记忆变量字典
        """
        try:
            # 从Redis获取对话历史
            messages_data = self.redis_client.lrange(self.chat_key, 0, -1)
            
            messages = []
            for msg_data in messages_data:
                try:
                    msg_dict = json.loads(msg_data)
                    messages.append(msg_dict["content"])
                except json.JSONDecodeError:
                    continue
            
            return {self.memory_key: messages}
            
        except Exception as e:
            self.logger.error(f"加载记忆变量失败: {str(e)}")
            return {self.memory_key: []}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        保存上下文到Redis
        
        Args:
            inputs: 输入变量
            outputs: 输出变量
        """
        try:
            # 保存输入
            input_text = inputs.get("input", "")
            if input_text:
                self._add_message_to_redis("user", input_text)
            
            # 保存输出
            output_text = outputs.get("output", "")
            if output_text:
                self._add_message_to_redis("assistant", output_text)
            
            # 更新过期时间
            self.redis_client.expire(self.chat_key, self.ttl)
            self.redis_client.expire(self.metadata_key, self.ttl)
            
            self.logger.debug("上下文已保存到Redis")
            
        except Exception as e:
            self.logger.error(f"保存上下文失败: {str(e)}")
    
    def _add_message_to_redis(self, message_type: str, content: str) -> None:
        """
        添加消息到Redis
        
        Args:
            message_type: 消息类型
            content: 消息内容
        """
        message_data = {
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }
        
        # 添加到列表
        self.redis_client.lpush(self.chat_key, json.dumps(message_data, ensure_ascii=False))
        
        # 限制列表长度（保留最近1000条消息）
        self.redis_client.ltrim(self.chat_key, 0, 999)
    
    def add_message(self, message: Message) -> None:
        """
        添加消息到Redis记忆
        
        Args:
            message: 消息对象
        """
        try:
            # 确定消息类型
            if message.message_type == MessageType.USER_INPUT:
                message_type = "user"
            elif message.message_type == MessageType.AGENT_RESPONSE:
                message_type = "assistant"
            else:
                message_type = "system"
            
            # 添加到Redis
            self._add_message_to_redis(message_type, message.content)
            
            self.logger.debug(f"添加消息到Redis: {message.message_type.value}")
            
        except Exception as e:
            self.logger.error(f"添加消息失败: {str(e)}")
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取消息列表
        
        Args:
            limit: 限制返回的消息数量
            
        Returns:
            消息列表
        """
        try:
            # 从Redis获取消息
            if limit:
                messages_data = self.redis_client.lrange(self.chat_key, 0, limit - 1)
            else:
                messages_data = self.redis_client.lrange(self.chat_key, 0, -1)
            
            messages = []
            for msg_data in messages_data:
                try:
                    msg_dict = json.loads(msg_data)
                    messages.append(msg_dict)
                except json.JSONDecodeError:
                    continue
            
            return messages
            
        except Exception as e:
            self.logger.error(f"获取消息失败: {str(e)}")
            return []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        获取对话摘要
        
        Returns:
            对话摘要信息
        """
        try:
            messages = self.get_messages()
            
            user_count = len([m for m in messages if m.get("type") == "user"])
            assistant_count = len([m for m in messages if m.get("type") == "assistant"])
            system_count = len([m for m in messages if m.get("type") == "system"])
            
            return {
                "agent_id": self.agent_id,
                "total_messages": len(messages),
                "user_messages": user_count,
                "assistant_messages": assistant_count,
                "system_messages": system_count,
                "memory_type": "redis"
            }
            
        except Exception as e:
            self.logger.error(f"获取对话摘要失败: {str(e)}")
            return {"agent_id": self.agent_id, "error": str(e)}
    
    def clear(self) -> None:
        """清除所有记忆"""
        try:
            # 删除Redis键
            self.redis_client.delete(self.chat_key)
            self.redis_client.delete(self.metadata_key)
            self.logger.info("Redis记忆已清除")
            
        except Exception as e:
            self.logger.error(f"清除记忆失败: {str(e)}")
    
    def set_ttl(self, ttl: int) -> None:
        """
        设置过期时间
        
        Args:
            ttl: 过期时间（秒）
        """
        self.ttl = ttl
        try:
            self.redis_client.expire(self.chat_key, ttl)
            self.redis_client.expire(self.metadata_key, ttl)
            self.logger.info(f"设置过期时间: {ttl}秒")
            
        except Exception as e:
            self.logger.error(f"设置过期时间失败: {str(e)}")
    
    def get_ttl(self) -> int:
        """
        获取剩余过期时间
        
        Returns:
            剩余过期时间（秒）
        """
        try:
            return self.redis_client.ttl(self.chat_key)
        except Exception as e:
            self.logger.error(f"获取过期时间失败: {str(e)}")
            return -1
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        导出记忆为字典格式
        
        Returns:
            记忆数据字典
        """
        try:
            messages = self.get_messages()
            summary = self.get_conversation_summary()
            
            return {
                "agent_id": self.agent_id,
                "memory_type": "redis",
                "ttl": self.ttl,
                "messages": messages,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"导出记忆失败: {str(e)}")
            return {"agent_id": self.agent_id, "error": str(e)}
    
    def import_from_dict(self, data: Dict[str, Any]) -> bool:
        """
        从字典导入记忆
        
        Args:
            data: 记忆数据字典
            
        Returns:
            是否成功导入
        """
        try:
            # 清除现有记忆
            self.clear()
            
            # 导入消息
            for msg_data in data.get("messages", []):
                self._add_message_to_redis(
                    msg_data.get("type", "user"),
                    msg_data.get("content", "")
                )
            
            # 设置TTL
            if "ttl" in data:
                self.set_ttl(data["ttl"])
            
            self.logger.info(f"成功导入 {len(data.get('messages', []))} 条消息")
            return True
            
        except Exception as e:
            self.logger.error(f"导入记忆失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        summary = self.get_conversation_summary()
        return f"Redis记忆[{self.agent_id}]: {summary.get('total_messages', 0)}条消息"
