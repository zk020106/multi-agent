"""
基于LangChain的记忆管理器
统一管理不同类型的记忆组件
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.memory import BaseMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from schema import Message, MessageType
from utils.logger import get_logger


class MemoryManager:
    """
    记忆管理器
    
    负责管理智能体的记忆组件，包括对话历史、上下文记忆等
    """
    
    def __init__(self, memory_type: str = "buffer"):
        """
        初始化记忆管理器
        
        Args:
            memory_type: 记忆类型 (buffer, summary, vector, redis)
        """
        self.memory_type = memory_type
        self.memories: Dict[str, BaseMemory] = {}
        self.logger = get_logger(__name__)
        
        self.logger.info(f"记忆管理器初始化完成，类型: {memory_type}")
    
    def create_memory(
        self,
        agent_id: str,
        memory_type: str = None,
        **kwargs
    ) -> BaseMemory:
        """
        为智能体创建记忆组件
        
        Args:
            agent_id: 智能体ID
            memory_type: 记忆类型
            **kwargs: 记忆组件参数
            
        Returns:
            记忆组件实例
        """
        memory_type = memory_type or self.memory_type
        
        if memory_type == "buffer":
            memory = ConversationBufferMemory(
                return_messages=True,
                **kwargs
            )
        elif memory_type == "summary":
            memory = ConversationSummaryMemory(
                llm=kwargs.get('llm'),
                return_messages=True,
                **kwargs
            )
        elif memory_type == "vector":
            from .vector_memory import VectorMemory
            memory = VectorMemory(
                agent_id=agent_id,
                **kwargs
            )
        elif memory_type == "redis":
            from .redis_memory import RedisMemory
            memory = RedisMemory(
                agent_id=agent_id,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")
        
        self.memories[agent_id] = memory
        self.logger.info(f"为智能体 {agent_id} 创建了 {memory_type} 类型记忆")
        
        return memory
    
    def get_memory(self, agent_id: str) -> Optional[BaseMemory]:
        """
        获取智能体的记忆组件
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            记忆组件实例或None
        """
        return self.memories.get(agent_id)
    
    def remove_memory(self, agent_id: str) -> bool:
        """
        移除智能体的记忆组件
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            是否成功移除
        """
        if agent_id in self.memories:
            del self.memories[agent_id]
            self.logger.info(f"移除智能体 {agent_id} 的记忆")
            return True
        return False
    
    def add_message(
        self,
        agent_id: str,
        message: Message
    ) -> None:
        """
        向智能体记忆添加消息
        
        Args:
            agent_id: 智能体ID
            message: 消息对象
        """
        memory = self.get_memory(agent_id)
        if not memory:
            self.logger.warning(f"智能体 {agent_id} 没有记忆组件")
            return
        
        # 转换为LangChain消息格式
        langchain_message = self._convert_to_langchain_message(message)
        
        # 添加到记忆
        if message.message_type == MessageType.USER_INPUT:
            memory.chat_memory.add_user_message(message.content)
        elif message.message_type == MessageType.AGENT_RESPONSE:
            memory.chat_memory.add_ai_message(message.content)
        elif message.message_type == MessageType.SYSTEM_NOTIFICATION:
            memory.chat_memory.add_message(SystemMessage(content=message.content))
        
        self.logger.debug(f"向智能体 {agent_id} 记忆添加消息: {message.message_type.value}")
    
    def get_conversation_history(
        self,
        agent_id: str,
        limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        获取智能体的对话历史
        
        Args:
            agent_id: 智能体ID
            limit: 限制返回的消息数量
            
        Returns:
            对话历史消息列表
        """
        memory = self.get_memory(agent_id)
        if not memory:
            return []
        
        messages = memory.chat_memory.messages
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def clear_memory(self, agent_id: str) -> None:
        """
        清除智能体的记忆
        
        Args:
            agent_id: 智能体ID
        """
        memory = self.get_memory(agent_id)
        if memory:
            memory.clear()
            self.logger.info(f"清除智能体 {agent_id} 的记忆")
    
    def get_memory_summary(self, agent_id: str) -> Optional[str]:
        """
        获取智能体记忆摘要
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            记忆摘要或None
        """
        memory = self.get_memory(agent_id)
        if not memory:
            return None
        
        if hasattr(memory, 'predict_new_summary'):
            try:
                return memory.predict_new_summary([], "")
            except:
                pass
        
        # 如果没有摘要功能，返回对话历史统计
        messages = memory.chat_memory.messages
        return f"对话历史包含 {len(messages)} 条消息"
    
    def save_memory_to_file(
        self,
        agent_id: str,
        file_path: str
    ) -> bool:
        """
        将记忆保存到文件
        
        Args:
            agent_id: 智能体ID
            file_path: 文件路径
            
        Returns:
            是否成功保存
        """
        try:
            memory = self.get_memory(agent_id)
            if not memory:
                return False
            
            # 获取对话历史
            messages = memory.chat_memory.messages
            
            # 转换为可序列化格式
            serializable_messages = []
            for msg in messages:
                serializable_messages.append({
                    "type": msg.__class__.__name__,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                })
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_messages, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"智能体 {agent_id} 的记忆已保存到 {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存记忆失败: {str(e)}")
            return False
    
    def load_memory_from_file(
        self,
        agent_id: str,
        file_path: str
    ) -> bool:
        """
        从文件加载记忆
        
        Args:
            agent_id: 智能体ID
            file_path: 文件路径
            
        Returns:
            是否成功加载
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                serializable_messages = json.load(f)
            
            # 创建记忆组件
            memory = self.create_memory(agent_id)
            
            # 恢复消息
            for msg_data in serializable_messages:
                msg_type = msg_data["type"]
                content = msg_data["content"]
                
                if msg_type == "HumanMessage":
                    memory.chat_memory.add_user_message(content)
                elif msg_type == "AIMessage":
                    memory.chat_memory.add_ai_message(content)
                elif msg_type == "SystemMessage":
                    memory.chat_memory.add_message(SystemMessage(content=content))
            
            self.logger.info(f"从 {file_path} 加载智能体 {agent_id} 的记忆")
            return True
            
        except Exception as e:
            self.logger.error(f"加载记忆失败: {str(e)}")
            return False
    
    def _convert_to_langchain_message(self, message: Message) -> BaseMessage:
        """
        将系统消息转换为LangChain消息格式
        
        Args:
            message: 系统消息对象
            
        Returns:
            LangChain消息对象
        """
        if message.message_type == MessageType.USER_INPUT:
            return HumanMessage(content=message.content)
        elif message.message_type == MessageType.AGENT_RESPONSE:
            return AIMessage(content=message.content)
        elif message.message_type == MessageType.SYSTEM_NOTIFICATION:
            return SystemMessage(content=message.content)
        else:
            return HumanMessage(content=message.content)
    
    def get_all_agents(self) -> List[str]:
        """
        获取所有有记忆的智能体ID列表
        
        Returns:
            智能体ID列表
        """
        return list(self.memories.keys())
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_agents": len(self.memories),
            "memory_types": {},
            "total_messages": 0
        }
        
        for agent_id, memory in self.memories.items():
            memory_type = type(memory).__name__
            if memory_type not in stats["memory_types"]:
                stats["memory_types"][memory_type] = 0
            stats["memory_types"][memory_type] += 1
            
            if hasattr(memory, 'chat_memory'):
                stats["total_messages"] += len(memory.chat_memory.messages)
        
        return stats
    
    def __str__(self) -> str:
        return f"记忆管理器(类型: {self.memory_type}, 智能体数量: {len(self.memories)})"
