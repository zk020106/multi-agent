"""
基于LangChain的ReAct智能体实现
ReAct (Reasoning and Acting) 模式：推理和行动循环
"""

import time
from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from schema import Message, Task, Result, MessageType, TaskStatus, ResultStatus
from .base_agent import BaseAgent


class ReActAgent(BaseAgent):
    """
    ReAct智能体：结合推理和行动的智能体
    
    特点：
    - 在执行动作前进行推理
    - 根据观察结果调整下一步行动
    - 适合需要多步推理的复杂任务
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: List[BaseTool] = None,
        memory = None,
        system_prompt: str = "",
        max_iterations: int = 10,
        verbose: bool = True,
        **kwargs
    ):
        """
        初始化ReAct智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            llm: 大语言模型
            tools: 工具列表
            memory: 记忆组件
            system_prompt: 系统提示词
            max_iterations: 最大迭代次数
            verbose: 是否详细输出
        """
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            llm=llm,
            tools=tools,
            memory=memory,
            system_prompt=system_prompt,
            **kwargs
        )
        
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 创建ReAct提示模板
        self.react_prompt = self._create_react_prompt()
        
        # 创建智能体执行器
        self.agent_executor = self._create_agent_executor()
        
        self.logger.info(f"ReAct智能体 {self.name} 初始化完成，最大迭代次数: {max_iterations}")
    
    def _create_react_prompt(self) -> PromptTemplate:
        """
        创建ReAct提示模板
        
        Returns:
            ReAct提示模板
        """
        template = """你是一个智能助手，能够通过推理和行动来解决问题。

你有以下工具可以使用：
{tools}

使用以下格式：

Question: 需要回答的问题
Thought: 你应该总是思考要做什么
Action: 要采取的行动，应该是[{tool_names}]中的一个
Action Input: 行动的输入
Observation: 行动的结果
... (这个思考/行动/观察可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 原始问题的最终答案

开始！

Question: {input}
Thought: {agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )
    
    def _create_agent_executor(self) -> AgentExecutor:
        """
        创建智能体执行器
        
        Returns:
            智能体执行器
        """
        if not self.tools:
            raise ValueError("ReAct智能体需要至少一个工具")
        
        # 创建ReAct智能体
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.react_prompt
        )
        
        # 创建执行器
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            memory=self.memory,
            callbacks=self.callbacks,
            handle_parsing_errors=True
        )
        
        return executor
    
    def receive_message(self, message: Message) -> None:
        """
        接收消息
        
        Args:
            message: 接收到的消息
        """
        self.logger.info(f"接收到消息: {message.content[:100]}...")
        
        # 如果有记忆组件，保存消息
        if self.memory:
            if message.message_type == MessageType.USER_INPUT:
                self.memory.chat_memory.add_user_message(message.content)
            elif message.message_type == MessageType.AGENT_RESPONSE:
                self.memory.chat_memory.add_ai_message(message.content)
    
    def act(self, task: Task) -> Result:
        """
        执行任务
        
        Args:
            task: 要执行的任务
            
        Returns:
            执行结果
        """
        start_time = time.time()
        self.is_busy = True
        self.current_task = task
        
        self.logger.task_start(task.id, task.title)
        
        try:
            # 更新任务状态
            self._update_task_status(task, TaskStatus.IN_PROGRESS)
            
            # 构建输入
            input_text = f"任务: {task.title}\n描述: {task.description}"
            if task.metadata:
                input_text += f"\n额外信息: {task.metadata}"
            
            # 执行ReAct循环
            result_data = self.agent_executor.invoke({"input": input_text})
            
            execution_time = time.time() - start_time
            
            # 创建成功结果
            result = self._create_result(
                task=task,
                status=ResultStatus.SUCCESS,
                data=result_data,
                execution_time=execution_time
            )
            
            # 更新统计
            self.tasks_completed += 1
            self.total_execution_time += execution_time
            
            # 更新任务状态
            self._update_task_status(task, TaskStatus.COMPLETED)
            
            self.logger.task_complete(task.id, "成功")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            
            # 创建失败结果
            result = self._create_result(
                task=task,
                status=ResultStatus.ERROR,
                error_message=error_message,
                execution_time=execution_time
            )
            
            # 更新统计
            self.tasks_failed += 1
            self.total_execution_time += execution_time
            
            # 更新任务状态
            self._update_task_status(task, TaskStatus.FAILED)
            
            self.logger.error(f"任务执行失败: {error_message}")
            
            return result
            
        finally:
            self.is_busy = False
            self.current_task = None
    
    def return_result(self, result: Result) -> Message:
        """
        返回执行结果
        
        Args:
            result: 执行结果
            
        Returns:
            结果消息
        """
        if result.status == ResultStatus.SUCCESS:
            content = f"任务执行成功！\n结果: {result.data.get('output', '无输出')}"
        else:
            content = f"任务执行失败: {result.error_message}"
        
        message = self._create_message(
            content=content,
            receiver="system",
            message_type=MessageType.AGENT_RESPONSE,
            conversation_id=result.task_id
        )
        
        self.logger.info(f"返回结果: {result.status.value}")
        
        return message
    
    def add_tool(self, tool: BaseTool) -> None:
        """
        添加工具并重新创建执行器
        
        Args:
            tool: 要添加的工具
        """
        super().add_tool(tool)
        # 重新创建执行器以包含新工具
        self.agent_executor = self._create_agent_executor()
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        移除工具并重新创建执行器
        
        Args:
            tool_name: 工具名称
            
        Returns:
            是否成功移除
        """
        success = super().remove_tool(tool_name)
        if success:
            # 重新创建执行器
            self.agent_executor = self._create_agent_executor()
        return success
    
    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """
        获取推理轨迹
        
        Returns:
            推理步骤列表
        """
        if hasattr(self.agent_executor, 'intermediate_steps'):
            return self.agent_executor.intermediate_steps
        return []
    
    def __str__(self) -> str:
        return f"ReAct智能体[{self.agent_id}]: {self.name}"
