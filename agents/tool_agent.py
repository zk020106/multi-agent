"""
基于LangChain的工具驱动智能体
专门用于执行特定工具操作的智能体
"""
import time
from typing import Any, Dict, List

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from schema import Message, Task, Result, MessageType, TaskStatus, ResultStatus
from .base_agent import BaseAgent


class ToolAgent(BaseAgent):
    """
    工具驱动智能体：专门用于执行工具操作
    
    特点：
    - 专注于工具调用
    - 自动选择合适的工具
    - 支持工具链式调用
    - 适合需要精确工具操作的任务
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
        max_iterations: int = 5,
        verbose: bool = True,
        **kwargs
    ):
        """
        初始化工具驱动智能体
        
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
        
        # 创建工具调用提示模板
        self.tool_prompt = self._create_tool_prompt()
        
        # 创建智能体执行器
        self.agent_executor = self._create_agent_executor()
        
        self.logger.info(f"工具驱动智能体 {self.name} 初始化完成，可用工具: {len(self.tools)}个")
    
    def _create_tool_prompt(self) -> ChatPromptTemplate:
        """
        创建工具调用提示模板
        
        Returns:
            工具调用提示模板
        """
        system_message = """你是一个专业的工具操作助手。你的任务是理解用户的需求，并选择最合适的工具来完成任务。

你有以下工具可以使用：
{tools}

请根据用户的需求，选择合适的工具并执行相应的操作。如果需要多个工具，请按顺序执行。

记住：
1. 仔细阅读工具的描述和参数要求
2. 确保输入参数格式正确
3. 如果工具执行失败，尝试其他方法
4. 始终以用户友好的方式解释结果

用户需求: {input}

请开始执行:"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
    
    def _create_agent_executor(self) -> AgentExecutor:
        """
        创建智能体执行器
        
        Returns:
            智能体执行器
        """
        if not self.tools:
            raise ValueError("工具驱动智能体需要至少一个工具")
        
        try:
            # 检查LLM是否支持工具调用
            if hasattr(self.llm, 'bind_tools') and callable(getattr(self.llm, 'bind_tools', None)):
                try:
                    # 创建工具调用智能体
                    agent = create_tool_calling_agent(
                        llm=self.llm,
                        tools=self.tools,
                        prompt=self.tool_prompt
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
                except Exception as e:
                    self.logger.warning(f"工具调用智能体创建失败，尝试ReAct模式: {e}")
                    # 继续尝试ReAct模式
            
            # 使用ReAct模式
            from langchain.agents import create_react_agent
            from langchain_core.prompts import PromptTemplate
            
            # 创建ReAct提示模板
            react_template = """你是一个专业的工具操作助手。你有以下工具可以使用：

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

            react_prompt = PromptTemplate(
                template=react_template,
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
            )
            
            # 创建ReAct智能体
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=react_prompt
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
                
        except Exception as e:
            self.logger.warning(f"创建智能体执行器失败，使用简化模式: {e}")
            # 如果都失败了，返回None，在act方法中处理
            return None
    
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
        执行任务（工具驱动模式）
        
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
            
            # 执行工具调用
            if self.agent_executor is not None:
                result_data = self.agent_executor.invoke({"input": input_text})
            else:
                # 如果智能体执行器不可用，使用简化的工具执行
                result_data = self._execute_tools_simple(input_text)
            
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
            content = f"工具操作完成！\n结果: {result.data.get('output', '无输出')}"
        else:
            content = f"工具操作失败: {result.error_message}"
        
        message = self._create_message(
            content=content,
            receiver="system",
            message_type=MessageType.AGENT_RESPONSE,
            conversation_id=result.task_id
        )
        
        self.logger.info(f"返回结果: {result.status.value}")
        
        return message
    
    def _execute_tools_simple(self, input_text: str) -> Dict[str, Any]:
        """
        简化的工具执行方法（当智能体执行器不可用时使用）
        
        Args:
            input_text: 输入文本
            
        Returns:
            执行结果
        """
        try:
            # 简单的工具选择逻辑
            # 根据输入文本选择最合适的工具
            if "计算" in input_text or "数学" in input_text or "+" in input_text or "-" in input_text or "*" in input_text or "/" in input_text:
                # 选择计算器工具
                calculator_tool = self.get_tool("calculator_tool")
                if calculator_tool:
                    # 提取数学表达式
                    import re
                    expressions = re.findall(r'[\d\+\-\*\/\(\)\.\s]+', input_text)
                    if expressions:
                        expression = expressions[0].strip()
                        result = calculator_tool.run(expression)
                        return {
                            "output": f"计算结果: {result}",
                            "tool_used": "calculator_tool",
                            "input": expression
                        }
            
            elif "搜索" in input_text or "查找" in input_text:
                # 选择搜索工具
                search_tool = self.get_tool("web_search_tool")
                if search_tool:
                    result = search_tool.run(input_text)
                    return {
                        "output": f"搜索结果: {result}",
                        "tool_used": "web_search_tool",
                        "input": input_text
                    }
            
            elif "代码" in input_text or "python" in input_text.lower() or "print" in input_text:
                # 选择代码执行工具
                code_tool = self.get_tool("code_execution_tool")
                if code_tool:
                    # 提取代码
                    import re
                    code_match = re.search(r'print\([^)]+\)', input_text)
                    if code_match:
                        code = code_match.group()
                        result = code_tool.run(code)
                        return {
                            "output": f"代码执行结果: {result}",
                            "tool_used": "code_execution_tool",
                            "input": code
                        }
            
            # 如果没有匹配的工具，返回默认响应
            return {
                "output": f"已收到任务: {input_text}，但无法找到合适的工具来执行。可用工具: {[tool.name for tool in self.tools]}",
                "tool_used": "none",
                "input": input_text
            }
            
        except Exception as e:
            return {
                "output": f"工具执行失败: {str(e)}",
                "tool_used": "error",
                "input": input_text
            }
    
    def execute_tool_directly(self, tool_name: str, tool_input: str) -> Dict[str, Any]:
        """
        直接执行指定工具
        
        Args:
            tool_name: 工具名称
            tool_input: 工具输入
            
        Returns:
            工具执行结果
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"工具 {tool_name} 不存在"
            }
        
        try:
            result = tool.run(tool_input)
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
    
    def get_tool_usage_stats(self) -> Dict[str, int]:
        """
        获取工具使用统计
        
        Returns:
            工具使用次数统计
        """
        # 这里可以实现更复杂的统计逻辑
        # 目前返回简单的工具列表
        return {tool.name: 0 for tool in self.tools}
    
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
    
    def __str__(self) -> str:
        return f"工具驱动智能体[{self.agent_id}]: {self.name}"
