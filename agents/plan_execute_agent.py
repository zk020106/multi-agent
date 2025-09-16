"""
基于LangChain的Plan-and-Execute智能体
继承自BaseAgent，实现任务分解和逐步执行
"""

import time
from typing import List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.memory import BaseMemory
from langchain_core.tools import BaseTool
from langchain_experimental.plan_and_execute import PlanAndExecute, load_chat_planner, load_agent_executor

from schema import Message, Task, Result, MessageType, TaskStatus, ResultStatus
from .base_agent import BaseAgent


class LangChainPlanExecuteAgent(BaseAgent):
    """
    Plan-and-Execute智能体
    使用LangChain实验性模块进行计划和执行
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        llm: BaseLanguageModel,
        tools: List[BaseTool] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: str = "",
        **kwargs
    ):
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

        # 初始化LangChain Planner和Executor
        self.planner = load_chat_planner(llm)
        self.executor = load_agent_executor(llm, self.tools, verbose=True)

        # 构建Plan-and-Execute对象
        self.agent = PlanAndExecute(
            planner=self.planner,
            executor=self.executor,
            verbose=True
        )

        self.logger.info(f"LangChain Plan-and-Execute智能体 {self.name} 初始化完成")

    def receive_message(self, message: Message) -> None:
        self.logger.info(f"接收到消息: {message.content[:100]}...")

        if self.memory:
            if message.message_type == MessageType.USER_INPUT:
                self.memory.chat_memory.add_user_message(message.content)
            elif message.message_type == MessageType.AGENT_RESPONSE:
                self.memory.chat_memory.add_ai_message(message.content)

    def act(self, task: Task) -> Result:
        start_time = time.time()
        self.is_busy = True
        self.current_task = task
        self.logger.task_start(task.id, task.title)
        self._update_task_status(task, TaskStatus.IN_PROGRESS)

        try:
            # 执行任务描述
            task_description = f"{task.title}\n{task.description}"
            if task.metadata:
                task_description += f"\n额外信息: {task.metadata}"

            # 调用Plan-and-Execute智能体
            output = self.agent.run(task_description)

            execution_time = time.time() - start_time

            result = self._create_result(
                task=task,
                status=ResultStatus.SUCCESS,
                data={
                    "output": output
                },
                execution_time=execution_time
            )

            self.tasks_completed += 1
            self.total_execution_time += execution_time
            self._update_task_status(task, TaskStatus.COMPLETED)
            self.logger.task_complete(task.id, "成功")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            self.tasks_failed += 1
            self.total_execution_time += execution_time
            self._update_task_status(task, TaskStatus.FAILED)
            self.logger.error(f"任务执行失败: {error_msg}")

            return self._create_result(
                task=task,
                status=ResultStatus.ERROR,
                error_message=error_msg,
                execution_time=execution_time
            )

        finally:
            self.is_busy = False
            self.current_task = None

    def return_result(self, result: Result) -> Message:
        if result.status == ResultStatus.SUCCESS:
            content = f"""任务执行完成！

输出结果:
{result.data.get("output")}
"""
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
