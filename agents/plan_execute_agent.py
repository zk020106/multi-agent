"""
基于LangChain的计划执行智能体
Plan-and-Execute模式：先制定计划，再执行计划
"""

import json
import time
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from schema import Message, Task, Result, MessageType, TaskStatus, ResultStatus
from .base_agent import BaseAgent


class PlanExecuteAgent(BaseAgent):
    """
    计划执行智能体：先制定详细计划，再按计划执行
    
    特点：
    - 将复杂任务分解为多个子任务
    - 制定详细的执行计划
    - 按计划逐步执行
    - 适合需要结构化处理的任务
    """

    def __init__(
            self,
            agent_id: str,
            name: str,
            description: str,
            llm: BaseLanguageModel,
            tools: List[BaseTool] = None,
            memory=None,
            system_prompt: str = "",
            max_plan_steps: int = 10,
            **kwargs
    ):
        """
        初始化计划执行智能体
        
        Args:
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            llm: 大语言模型
            tools: 工具列表
            memory: 记忆组件
            system_prompt: 系统提示词
            max_plan_steps: 最大计划步骤数
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

        self.max_plan_steps = max_plan_steps

        # 创建计划提示模板
        self.planning_prompt = self._create_planning_prompt()

        # 创建执行提示模板
        self.execution_prompt = self._create_execution_prompt()

        self.logger.info(f"计划执行智能体 {self.name} 初始化完成，最大计划步骤: {max_plan_steps}")

    def _create_planning_prompt(self) -> PromptTemplate:
        """
        创建计划制定提示模板
        
        Returns:
            计划提示模板
        """
        template = """你是一个任务规划专家。请将给定的任务分解为详细的执行计划。

任务: {task_description}

可用工具: {available_tools}

请制定一个详细的执行计划，包含以下信息：
1. 总体目标
2. 具体步骤（最多{max_steps}步）
3. 每步需要的工具
4. 步骤间的依赖关系

请以JSON格式返回计划：
{{
    "goal": "总体目标",
    "steps": [
        {{
            "step_id": 1,
            "description": "步骤描述",
            "tool": "需要的工具名称",
            "input": "工具输入参数",
            "depends_on": []
        }}
    ]
}}

计划:"""

        return PromptTemplate(
            template=template,
            input_variables=["task_description", "available_tools", "max_steps"]
        )

    def _create_execution_prompt(self) -> PromptTemplate:
        """
        创建执行提示模板
        
        Returns:
            执行提示模板
        """
        template = """你正在执行任务计划中的第{step_number}步。

步骤描述: {step_description}
使用工具: {tool_name}
工具输入: {tool_input}

请执行这个步骤并返回结果。

执行结果:"""

        return PromptTemplate(
            template=template,
            input_variables=["step_number", "step_description", "tool_name", "tool_input"]
        )

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
        执行任务（计划-执行模式）
        
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

            # 第一阶段：制定计划
            plan = self._create_plan(task)
            if not plan:
                raise Exception("无法制定执行计划")

            self.logger.info(f"制定了包含{len(plan['steps'])}个步骤的执行计划")

            # 第二阶段：执行计划
            execution_results = self._execute_plan(plan, task)

            execution_time = time.time() - start_time

            # 创建成功结果
            result = self._create_result(
                task=task,
                status=ResultStatus.SUCCESS,
                data={
                    "plan": plan,
                    "execution_results": execution_results,
                    "total_steps": len(plan['steps']),
                    "completed_steps": len([r for r in execution_results if r['success']])
                },
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

    def _create_plan(self, task: Task) -> Optional[Dict[str, Any]]:
        """
        制定执行计划
        
        Args:
            task: 任务对象
            
        Returns:
            执行计划字典
        """
        try:
            # 准备计划输入
            task_description = f"{task.title}\n{task.description}"
            if task.metadata:
                task_description += f"\n额外信息: {task.metadata}"

            available_tools = ", ".join([tool.name for tool in self.tools])

            # 调用LLM制定计划
            prompt_input = self.planning_prompt.format(
                task_description=task_description,
                available_tools=available_tools,
                max_steps=self.max_plan_steps
            )

            response = self.llm.invoke(prompt_input)

            # 解析JSON响应
            plan_text = response.content if hasattr(response, 'content') else str(response)

            # 尝试提取JSON部分
            start_idx = plan_text.find('{')
            end_idx = plan_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_text = plan_text[start_idx:end_idx]
                plan = json.loads(json_text)

                # 验证计划格式
                if 'goal' in plan and 'steps' in plan:
                    return plan

            raise Exception("无法解析LLM返回的计划")

        except Exception as e:
            self.logger.error(f"制定计划失败: {str(e)}")
            return None

    def _execute_plan(self, plan: Dict[str, Any], task: Task) -> List[Dict[str, Any]]:
        """
        执行计划
        
        Args:
            plan: 执行计划
            task: 任务对象
            
        Returns:
            执行结果列表
        """
        results = []
        completed_steps = set()

        for step in plan['steps']:
            step_id = step['step_id']
            step_description = step['description']
            tool_name = step['tool']
            tool_input = step['input']
            depends_on = step.get('depends_on', [])

            # 检查依赖是否完成
            if not all(dep in completed_steps for dep in depends_on):
                self.logger.warning(f"步骤{step_id}的依赖未完成，跳过")
                results.append({
                    'step_id': step_id,
                    'success': False,
                    'error': '依赖未完成',
                    'result': None
                })
                continue

            try:
                # 执行步骤
                self.logger.info(f"执行步骤{step_id}: {step_description}")

                # 获取工具
                tool = self.get_tool(tool_name)
                if not tool:
                    raise Exception(f"工具 {tool_name} 不存在")

                # 执行工具
                tool_result = tool.run(tool_input)

                # 记录结果
                results.append({
                    'step_id': step_id,
                    'success': True,
                    'error': None,
                    'result': tool_result
                })

                completed_steps.add(step_id)

                self.logger.info(f"步骤{step_id}执行成功")

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"步骤{step_id}执行失败: {error_msg}")

                results.append({
                    'step_id': step_id,
                    'success': False,
                    'error': error_msg,
                    'result': None
                })

        return results

    def return_result(self, result: Result) -> Message:
        """
        返回执行结果
        
        Args:
            result: 执行结果
            
        Returns:
            结果消息
        """
        if result.status == ResultStatus.SUCCESS:
            data = result.data
            total_steps = data.get('total_steps', 0)
            completed_steps = data.get('completed_steps', 0)

            content = f"""任务执行完成！

执行计划: {data.get('plan', {}).get('goal', '未知目标')}
总步骤数: {total_steps}
完成步骤数: {completed_steps}
成功率: {completed_steps / total_steps * 100:.1f}% if total_steps > 0 else 0

详细结果请查看result.data中的execution_results字段。"""
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

    def get_plan(self) -> Optional[Dict[str, Any]]:
        """
        获取当前任务的执行计划
        
        Returns:
            执行计划或None
        """
        if self.current_task and hasattr(self, '_current_plan'):
            return self._current_plan
        return None

    def __str__(self) -> str:
        return f"计划执行智能体[{self.agent_id}]: {self.name}"
