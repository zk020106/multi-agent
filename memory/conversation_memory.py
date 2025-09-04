"""
对话记忆组件，基于LangChain实现
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from schema import Message, MessageType
from utils.logger import get_logger


class ConversationMemory:
    """
    对话记忆组件
    
    基于LangChain的对话记忆，支持多种记忆策略
    """
    
    def __init__(
        self,
        agent_id: str,
        memory_type: str = "buffer",
        max_token_limit: int = 2000,
        llm = None
    ):
        """
        初始化对话记忆
        
        Args:
            agent_id: 智能体ID
            memory_type: 记忆类型 (buffer, summary)
            max_token_limit: 最大token限制
            llm: 用于摘要的大语言模型
        """
        self.agent_id = agent_id
        self.memory_type = memory_type
        self.max_token_limit = max_token_limit
        self.llm = llm
        self.logger = get_logger(f"conversation_memory.{agent_id}")
        
        # 创建记忆组件
        self.memory = self._create_memory()
        
        self.logger.info(f"对话记忆初始化完成，类型: {memory_type}")
    
    def _create_memory(self):
        """创建记忆组件"""
        if self.memory_type == "buffer":
            return ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        elif self.memory_type == "summary":
            if not self.llm:
                raise ValueError("摘要记忆需要提供LLM实例")
            return ConversationSummaryMemory(
                llm=self.llm,
                return_messages=True,
                memory_key="chat_history",
            )
        else:
            raise ValueError(f"不支持的记忆类型: {self.memory_type}")
    
    def add_message(self, message: Message) -> None:
        """
        添加消息到记忆
        
        Args:
            message: 消息对象
        """
        try:
            if message.message_type == MessageType.USER_INPUT:
                self.memory.chat_memory.add_user_message(message.content)
            elif message.message_type == MessageType.AGENT_RESPONSE:
                self.memory.chat_memory.add_ai_message(message.content)
            elif message.message_type == MessageType.SYSTEM_NOTIFICATION:
                self.memory.chat_memory.add_message(SystemMessage(content=message.content))
            
            self.logger.debug(f"添加消息: {message.message_type.value}")
            
        except Exception as e:
            self.logger.error(f"添加消息失败: {str(e)}")
    
    def add_langchain_message(self, message: BaseMessage) -> None:
        """
        添加LangChain消息到记忆
        
        Args:
            message: LangChain消息对象
        """
        try:
            self.memory.chat_memory.add_message(message)
            self.logger.debug(f"添加LangChain消息: {type(message).__name__}")
            
        except Exception as e:
            self.logger.error(f"添加LangChain消息失败: {str(e)}")
    
    def get_messages(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        获取对话消息
        
        Args:
            limit: 限制返回的消息数量
            
        Returns:
            消息列表
        """
        messages = self.memory.chat_memory.messages
        if limit:
            messages = messages[-limit:]
        return messages
    
    def get_conversation_summary(self) -> Optional[str]:
        """
        获取对话摘要
        
        Returns:
            对话摘要或None
        """
        if hasattr(self.memory, 'predict_new_summary'):
            try:
                return self.memory.predict_new_summary([], "")
            except Exception as e:
                self.logger.error(f"生成摘要失败: {str(e)}")
                return None
        return None
    
    def clear(self) -> None:
        """清除所有记忆"""
        self.memory.clear()
        self.logger.info("记忆已清除")
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """
        获取记忆变量
        
        Returns:
            记忆变量字典
        """
        return self.memory.load_memory_variables({})
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        保存上下文
        
        Args:
            inputs: 输入变量
            outputs: 输出变量
        """
        self.memory.save_context(inputs, outputs)
        self.logger.debug("上下文已保存")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            统计信息字典
        """
        messages = self.memory.chat_memory.messages
        
        stats = {
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if isinstance(m, HumanMessage)]),
            "ai_messages": len([m for m in messages if isinstance(m, AIMessage)]),
            "system_messages": len([m for m in messages if isinstance(m, SystemMessage)]),
            "max_token_limit": self.max_token_limit
        }
        
        # 计算总字符数
        total_chars = sum(len(m.content) for m in messages)
        stats["total_characters"] = total_chars
        
        return stats
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        导出记忆为字典格式
        
        Returns:
            记忆数据字典
        """
        messages = self.memory.chat_memory.messages
        
        return {
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
            "max_token_limit": self.max_token_limit,
            "messages": [
                {
                    "type": type(msg).__name__,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                }
                for msg in messages
            ],
            "stats": self.get_stats()
        }
    
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
                msg_type = msg_data["type"]
                content = msg_data["content"]
                
                if msg_type == "HumanMessage":
                    self.memory.chat_memory.add_user_message(content)
                elif msg_type == "AIMessage":
                    self.memory.chat_memory.add_ai_message(content)
                elif msg_type == "SystemMessage":
                    self.memory.chat_memory.add_message(SystemMessage(content=content))
            
            self.logger.info(f"成功导入 {len(data.get('messages', []))} 条消息")
            return True
            
        except Exception as e:
            self.logger.error(f"导入记忆失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        stats = self.get_stats()
        return f"对话记忆[{self.agent_id}]: {stats['total_messages']}条消息"
