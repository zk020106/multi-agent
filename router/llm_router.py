"""
基于LLM的智能路由器
使用大语言模型进行智能路由决策
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from schema import Message, Task, MessageType
from .base_router import BaseRouter


class LLMRouter(BaseRouter):
    """
    基于LLM的智能路由器
    
    特点：
    - 使用大语言模型进行路由决策
    - 能够理解复杂的语义和上下文
    - 支持动态学习和适应
    - 适合复杂和不确定的路由场景
    """
    
    def __init__(
        self,
        router_id: str,
        llm,
        name: str = "LLM智能路由器",
        description: str = "基于大语言模型的智能消息路由器",
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        """
        初始化LLM路由器
        
        Args:
            router_id: 路由器ID
            llm: 大语言模型实例
            name: 路由器名称
            description: 路由器描述
            max_retries: 最大重试次数
            timeout: 超时时间
        """
        super().__init__(router_id, name, description)
        
        self.llm = llm
        self.max_retries = max_retries
        self.timeout = timeout
        
        # 路由提示模板
        self.routing_prompt = self._create_routing_prompt()
        
        self.logger.info(f"LLM路由器初始化完成，模型: {type(llm).__name__}")
    
    def _create_routing_prompt(self) -> str:
        """
        创建路由提示模板
        
        Returns:
            路由提示模板
        """
        return """你是一个智能消息路由器，负责将消息路由到最合适的智能体。

可用智能体信息：
{agent_info}

消息信息：
- 发送者: {sender}
- 消息类型: {message_type}
- 消息内容: {content}
- 消息元数据: {metadata}

请分析消息内容，选择最合适的智能体来处理这个消息。

返回格式（JSON）：
{{
    "reasoning": "选择理由",
    "target_agents": ["agent_id1", "agent_id2"],
    "confidence": 0.95,
    "alternative_agents": ["agent_id3"]
}}

请确保：
1. 选择最匹配的智能体
2. 考虑智能体的能力和当前状态
3. 提供清晰的推理过程
4. 返回有效的JSON格式"""
    
    def route_message(self, message: Message) -> List[str]:
        """
        使用LLM路由消息
        
        Args:
            message: 要路由的消息
            
        Returns:
            目标智能体ID列表
        """
        start_time = time.time()
        
        try:
            # 准备智能体信息
            agent_info = self._prepare_agent_info()
            
            # 构建提示
            prompt = self.routing_prompt.format(
                agent_info=agent_info,
                sender=message.sender,
                message_type=message.message_type.value,
                content=message.content,
                metadata=json.dumps(message.metadata, ensure_ascii=False, indent=2)
            )
            
            # 调用LLM进行路由决策
            routing_decision = self._call_llm_for_routing(prompt)
            
            if routing_decision:
                target_agents = routing_decision.get("target_agents", [])
                reasoning = routing_decision.get("reasoning", "")
                confidence = routing_decision.get("confidence", 0.0)
                
                # 验证目标智能体
                valid_agents = self._validate_target_agents(target_agents)
                
                routing_time = time.time() - start_time
                self._record_routing_decision(message, valid_agents, routing_time, True)
                self.messages_routed += 1
                self.total_routing_time += routing_time
                
                self.logger.info(f"LLM路由决策: {reasoning} (置信度: {confidence:.2f})")
                self.logger.info(f"消息 {message.id} 路由到智能体: {valid_agents}")
                
                return valid_agents
            else:
                # LLM路由失败，使用备用策略
                return self._fallback_routing(message)
                
        except Exception as e:
            routing_time = time.time() - start_time
            self._record_routing_decision(message, [], routing_time, False)
            self.routing_errors += 1
            
            self.logger.error(f"LLM路由失败: {str(e)}")
            return self._fallback_routing(message)
    
    def route_task(self, task: Task) -> Optional[str]:
        """
        使用LLM路由任务
        
        Args:
            task: 要路由的任务
            
        Returns:
            目标智能体ID或None
        """
        # 创建临时消息对象用于路由
        temp_message = Message(
            id=str(uuid.uuid4()),
            sender="system",
            receiver="",
            content=f"任务: {task.title}\n描述: {task.description}",
            message_type=MessageType.USER_INPUT,
            metadata={
                "task_id": task.id,
                "task_priority": task.priority.value,
                "task_type": "task_routing"
            }
        )
        
        target_agents = self.route_message(temp_message)
        return target_agents[0] if target_agents else None
    
    def _prepare_agent_info(self) -> str:
        """
        准备智能体信息字符串
        
        Returns:
            智能体信息字符串
        """
        agent_info_list = []
        
        for agent_id, agent in self.registered_agents.items():
            capabilities = self.agent_capabilities.get(agent_id, [])
            role = self.agent_roles.get(agent_id, "general")
            status = "可用" if not agent.is_busy else "忙碌"
            
            agent_info = f"""
智能体ID: {agent_id}
名称: {agent.name}
描述: {agent.description}
角色: {role}
能力: {', '.join(capabilities)}
状态: {status}
任务完成数: {agent.tasks_completed}
任务失败数: {agent.tasks_failed}
"""
            agent_info_list.append(agent_info)
        
        return "\n".join(agent_info_list)
    
    def _call_llm_for_routing(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        调用LLM进行路由决策
        
        Args:
            prompt: 路由提示
            
        Returns:
            路由决策字典或None
        """
        for attempt in range(self.max_retries):
            try:
                # 调用LLM
                response = self.llm.invoke(prompt)
                
                # 解析响应
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # 提取JSON部分
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    routing_decision = json.loads(json_text)
                    return routing_decision
                else:
                    self.logger.warning(f"LLM响应中未找到有效JSON，尝试 {attempt + 1}/{self.max_retries}")
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON解析失败，尝试 {attempt + 1}/{self.max_retries}: {str(e)}")
            except Exception as e:
                self.logger.warning(f"LLM调用失败，尝试 {attempt + 1}/{self.max_retries}: {str(e)}")
        
        return None
    
    def _validate_target_agents(self, target_agents: List[str]) -> List[str]:
        """
        验证目标智能体
        
        Args:
            target_agents: 目标智能体ID列表
            
        Returns:
            有效的智能体ID列表
        """
        valid_agents = []
        
        for agent_id in target_agents:
            if agent_id in self.registered_agents:
                agent = self.registered_agents[agent_id]
                if not agent.is_busy:  # 只选择可用的智能体
                    valid_agents.append(agent_id)
                else:
                    self.logger.warning(f"智能体 {agent_id} 当前忙碌，跳过")
            else:
                self.logger.warning(f"智能体 {agent_id} 不存在，跳过")
        
        return valid_agents
    
    def _fallback_routing(self, message: Message) -> List[str]:
        """
        备用路由策略
        
        Args:
            message: 消息对象
            
        Returns:
            目标智能体ID列表
        """
        # 简单的备用策略：选择第一个可用的通用智能体
        general_agents = self.get_agents_by_role("general")
        available_agents = [agent for agent in general_agents if not agent.is_busy]
        
        if available_agents:
            target_agent = available_agents[0]
            self.logger.info(f"使用备用路由策略，选择智能体: {target_agent.name}")
            return [target_agent.agent_id]
        else:
            self.logger.warning("没有可用的智能体进行备用路由")
            return []
    
    def update_routing_prompt(self, new_prompt: str) -> None:
        """
        更新路由提示模板
        
        Args:
            new_prompt: 新的提示模板
        """
        self.routing_prompt = new_prompt
        self.logger.info("路由提示模板已更新")
    
    def add_context_example(
        self,
        message_content: str,
        target_agents: List[str],
        reasoning: str
    ) -> None:
        """
        添加路由上下文示例
        
        Args:
            message_content: 消息内容
            target_agents: 目标智能体
            reasoning: 推理过程
        """
        example = {
            "message": message_content,
            "target_agents": target_agents,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        # 这里可以将示例存储到记忆中，用于改进路由决策
        self.logger.info(f"添加路由示例: {reasoning}")
    
    def get_llm_stats(self) -> Dict[str, Any]:
        """
        获取LLM路由统计信息
        
        Returns:
            LLM统计信息
        """
        base_stats = self.get_routing_stats()
        base_stats.update({
            "llm_model": type(self.llm).__name__,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "routing_method": "llm_based"
        })
        return base_stats
    
    def __str__(self) -> str:
        return f"LLM路由器[{self.router_id}]: {self.name} - {type(self.llm).__name__}"
