"""
基于向量数据库的记忆组件
使用向量存储和检索对话记忆
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.memory import BaseMemory
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from schema import Message, MessageType
from utils.logger import get_logger


class VectorMemory(BaseMemory):
    """
    向量记忆组件
    
    使用向量数据库存储和检索对话记忆，支持语义搜索
    """
    
    def __init__(
        self,
        agent_id: str,
        embeddings: Embeddings,
        vectorstore: Optional[VectorStore] = None,
        k: int = 4,
        memory_key: str = "chat_history"
    ):
        """
        初始化向量记忆
        
        Args:
            agent_id: 智能体ID
            embeddings: 嵌入模型
            vectorstore: 向量存储实例
            k: 检索的相似文档数量
            memory_key: 记忆键名
        """
        super().__init__()
        
        self.agent_id = agent_id
        self.embeddings = embeddings
        self.k = k
        self.memory_key = memory_key
        self.logger = get_logger(f"vector_memory.{agent_id}")
        
        # 初始化向量存储
        if vectorstore is None:
            self.vectorstore = FAISS.from_texts(
                ["初始记忆"],
                embeddings,
                metadatas=[{"agent_id": agent_id, "timestamp": datetime.now().isoformat()}]
            )
        else:
            self.vectorstore = vectorstore
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.logger.info(f"向量记忆初始化完成，智能体: {agent_id}")
    
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
        # 从输入中获取查询文本
        query = inputs.get("input", "")
        if not query:
            return {self.memory_key: []}
        
        try:
            # 检索相关记忆
            relevant_docs = self.vectorstore.similarity_search(query, k=self.k)
            
            # 转换为消息格式
            messages = []
            for doc in relevant_docs:
                content = doc.page_content
                metadata = doc.metadata
                
                # 根据元数据确定消息类型
                if metadata.get("message_type") == "user":
                    messages.append(f"用户: {content}")
                elif metadata.get("message_type") == "assistant":
                    messages.append(f"助手: {content}")
                else:
                    messages.append(content)
            
            return {self.memory_key: messages}
            
        except Exception as e:
            self.logger.error(f"加载记忆变量失败: {str(e)}")
            return {self.memory_key: []}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        保存上下文到向量存储
        
        Args:
            inputs: 输入变量
            outputs: 输出变量
        """
        try:
            # 提取输入和输出文本
            input_text = inputs.get("input", "")
            output_text = outputs.get("output", "")
            
            # 保存输入
            if input_text:
                self._add_to_vectorstore(
                    input_text,
                    {"message_type": "user", "timestamp": datetime.now().isoformat()}
                )
            
            # 保存输出
            if output_text:
                self._add_to_vectorstore(
                    output_text,
                    {"message_type": "assistant", "timestamp": datetime.now().isoformat()}
                )
            
            self.logger.debug("上下文已保存到向量存储")
            
        except Exception as e:
            self.logger.error(f"保存上下文失败: {str(e)}")
    
    def _add_to_vectorstore(self, text: str, metadata: Dict[str, Any]) -> None:
        """
        添加文本到向量存储
        
        Args:
            text: 文本内容
            metadata: 元数据
        """
        # 添加智能体ID到元数据
        metadata["agent_id"] = self.agent_id
        
        # 分割文本
        texts = self.text_splitter.split_text(text)
        
        # 添加到向量存储
        metadatas = [metadata.copy() for _ in texts]
        self.vectorstore.add_texts(texts, metadatas)
    
    def add_message(self, message: Message) -> None:
        """
        添加消息到向量记忆
        
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
            
            # 添加到向量存储
            self._add_to_vectorstore(
                message.content,
                {
                    "message_type": message_type,
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": message.conversation_id
                }
            )
            
            self.logger.debug(f"添加消息到向量存储: {message.message_type.value}")
            
        except Exception as e:
            self.logger.error(f"添加消息失败: {str(e)}")
    
    def search_similar(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        搜索相似记忆
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似记忆列表
        """
        try:
            k = k or self.k
            docs = self.vectorstore.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"搜索相似记忆失败: {str(e)}")
            return []
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        获取记忆摘要
        
        Returns:
            记忆摘要信息
        """
        try:
            # 获取所有文档
            all_docs = self.vectorstore.similarity_search("", k=1000)
            
            # 统计信息
            total_docs = len(all_docs)
            user_messages = len([d for d in all_docs if d.metadata.get("message_type") == "user"])
            assistant_messages = len([d for d in all_docs if d.metadata.get("message_type") == "assistant"])
            
            return {
                "agent_id": self.agent_id,
                "total_documents": total_docs,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "memory_type": "vector"
            }
            
        except Exception as e:
            self.logger.error(f"获取记忆摘要失败: {str(e)}")
            return {"agent_id": self.agent_id, "error": str(e)}
    
    def clear(self) -> None:
        """清除所有记忆"""
        try:
            # 重新初始化向量存储
            self.vectorstore = FAISS.from_texts(
                ["记忆已清除"],
                self.embeddings,
                metadatas=[{"agent_id": self.agent_id, "timestamp": datetime.now().isoformat()}]
            )
            self.logger.info("向量记忆已清除")
            
        except Exception as e:
            self.logger.error(f"清除记忆失败: {str(e)}")
    
    def save_to_file(self, file_path: str) -> bool:
        """
        保存向量存储到文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功保存
        """
        try:
            self.vectorstore.save_local(file_path)
            self.logger.info(f"向量记忆已保存到 {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存向量记忆失败: {str(e)}")
            return False
    
    def load_from_file(self, file_path: str) -> bool:
        """
        从文件加载向量存储
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功加载
        """
        try:
            self.vectorstore = FAISS.load_local(file_path, self.embeddings)
            self.logger.info(f"从 {file_path} 加载向量记忆")
            return True
            
        except Exception as e:
            self.logger.error(f"加载向量记忆失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        summary = self.get_memory_summary()
        return f"向量记忆[{self.agent_id}]: {summary.get('total_documents', 0)}个文档"
