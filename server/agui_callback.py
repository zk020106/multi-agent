"""
ag-ui 事件回调：
将 LangChain 执行过程中的关键事件（思考/逐token/工具调用）
统一转换为可推送给前端（SSE/WS）的事件对象。
使用官方 ag-ui-protocol 包统一事件格式。
"""

import asyncio
from typing import Any, Dict

from langchain_core.callbacks import AsyncCallbackHandler
from ag_ui.core import (
    TextMessageContentEvent,
    TextMessageStartEvent,
    TextMessageEndEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ThinkingStartEvent,
    ThinkingEndEvent,
    ThinkingTextMessageContentEvent,
    RunStartedEvent,
    RunFinishedEvent,
    RunErrorEvent,
    EventType
)


class AguiEventStreamer(AsyncCallbackHandler):
    """ag-ui 事件流回调处理器。

    - 接收 LangChain 执行回调
    - 通过异步队列输出统一事件，供 SSE / WebSocket 推送层消费
    - 使用官方 ag-ui-protocol 事件格式
    """

    def __init__(self, queue: asyncio.Queue, session_id: str = None, message_id: str = None):
        # 由适配层传入的异步队列，用于承载事件
        self.queue = queue
        self.thread_id = session_id  # 使用 thread_id 而不是 session_id
        self.run_id = message_id     # 使用 run_id 而不是 message_id
        print(f"🔧 AguiEventStreamer 初始化: thread_id={self.thread_id}, run_id={self.run_id}")

    async def _emit_event(self, event) -> None:
        # 使用官方 ag-ui-protocol 事件对象
        await self.queue.put(event)

    # 提供给协调器/执行器的通用触发接口
    async def emit(self, event: str, payload: Dict[str, Any]) -> None:
        # 保持向后兼容，将旧格式转换为新格式
        if event == "start":
            ag_event = RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=self.thread_id,
                run_id=self.run_id
            )
        elif event == "end":
            ag_event = RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=self.thread_id,
                run_id=self.run_id
            )
        elif event == "thought":
            ag_event = ThinkingTextMessageContentEvent(
                type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                delta=payload.get("text", "")
            )
        elif event == "tool_start":
            ag_event = ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=f"tool_{self.run_id}",
                tool_call_name=payload.get("tool", ""),
                parent_message_id=self.run_id
            )
        elif event == "tool_end":
            ag_event = ToolCallResultEvent(
                type=EventType.TOOL_CALL_RESULT,
                message_id=self.run_id,
                tool_call_id=f"tool_{self.run_id}",
                content=payload.get("output", ""),
                role="tool"
            )
        elif event == "tool_error":
            ag_event = RunErrorEvent(
                type=EventType.RUN_ERROR,
                thread_id=self.thread_id,
                run_id=self.run_id,
                error=payload.get("error", "")
            )
        elif event == "message":
            ag_event = TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=self.run_id,
                delta=payload.get("content", "")
            )
        elif event == "error":
            ag_event = RunErrorEvent(
                type=EventType.RUN_ERROR,
                thread_id=self.thread_id,
                run_id=self.run_id,
                error=payload.get("error", "")
            )
        else:
            # 对于未知事件，创建通用事件
            ag_event = TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=self.run_id,
                delta=f"{event}: {payload}"
            )
        
        await self._emit_event(ag_event)

    # ===== 链路级回调 =====
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        ag_event = RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id=self.thread_id,
            run_id=self.run_id
        )
        # 同步放入队列
        self.queue.put_nowait(ag_event)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        ag_event = RunFinishedEvent(
            type=EventType.RUN_FINISHED,
            thread_id=self.thread_id,
            run_id=self.run_id
        )
        self.queue.put_nowait(ag_event)

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        ag_event = RunErrorEvent(
            type=EventType.RUN_ERROR,
            thread_id=self.thread_id,
            run_id=self.run_id,
            error=str(error)
        )
        self.queue.put_nowait(ag_event)

    # ===== LLM 输出（思考/逐token） =====
    async def on_text(self, text: str, **kwargs: Any) -> None:
        print(f"📝 on_text 被调用: {text[:50]}...")
        # 思考片段（可用于在前端展示思维过程）
        ag_event = ThinkingTextMessageContentEvent(
            type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
            delta=text
        )
        await self.queue.put(ag_event)
        print(f"✅ on_text 事件已放入队列")

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(f"🎯 on_llm_new_token 被调用: {token}")
        # 逐token 推送（可选开启）
        ag_event = TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=self.run_id,
            delta=token
        )
        await self.queue.put(ag_event)
        print(f"✅ on_llm_new_token 事件已放入队列")

    # ===== 工具调用 =====
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "tool"
        print(f"🔧 on_tool_start 被调用: {name}, input: {input_str[:50]}...")
        print(f"🔧 工具调用参数: serialized={serialized}, kwargs={kwargs}")
        ag_event = ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=f"tool_{self.run_id}",
            tool_call_name=name,
            parent_message_id=self.run_id
        )
        await self.queue.put(ag_event)
        print(f"✅ on_tool_start 事件已放入队列")

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        print(f"🔧 on_tool_end 被调用: {output[:50]}...")
        
        # 发送工具调用结束事件
        end_event = ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=f"tool_{self.run_id}"
        )
        await self.queue.put(end_event)
        print(f"✅ on_tool_end 结束事件已放入队列")
        
        # 发送工具调用结果事件
        result_event = ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=self.run_id,
            tool_call_id=f"tool_{self.run_id}",
            content=output,
            role="tool"
        )
        await self.queue.put(result_event)
        print(f"✅ on_tool_end 结果事件已放入队列")

    async def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        print(f"❌ on_tool_error 被调用: {error}")
        ag_event = RunErrorEvent(
            type=EventType.RUN_ERROR,
            thread_id=self.thread_id,
            run_id=self.run_id,
            error=str(error)
        )
        await self.queue.put(ag_event)
        print(f"✅ on_tool_error 事件已放入队列")

    # ===== 同步工具调用回调（备用） =====
    def on_tool_start_sync(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """同步版本的工具开始回调"""
        name = serialized.get("name") or serialized.get("id") or "tool"
        print(f"🔧 [SYNC] on_tool_start 被调用: {name}, input: {input_str[:50]}...")
        ag_event = ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=f"tool_{self.run_id}",
            tool_call_name=name,
            parent_message_id=self.run_id
        )
        self.queue.put_nowait(ag_event)
        print(f"✅ [SYNC] on_tool_start 事件已放入队列")

    def on_tool_end_sync(self, output: str, **kwargs: Any) -> None:
        """同步版本的工具结束回调"""
        print(f"🔧 [SYNC] on_tool_end 被调用: {output[:50]}...")
        
        # 发送工具调用结束事件
        end_event = ToolCallEndEvent(
            type=EventType.TOOL_CALL_END,
            tool_call_id=f"tool_{self.run_id}"
        )
        self.queue.put_nowait(end_event)
        print(f"✅ [SYNC] on_tool_end 结束事件已放入队列")
        
        # 发送工具调用结果事件
        result_event = ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=self.run_id,
            tool_call_id=f"tool_{self.run_id}",
            content=output,
            role="tool"
        )
        self.queue.put_nowait(result_event)
        print(f"✅ [SYNC] on_tool_end 结果事件已放入队列")

    def on_tool_error_sync(self, error: BaseException, **kwargs: Any) -> None:
        """同步版本的工具错误回调"""
        print(f"❌ [SYNC] on_tool_error 被调用: {error}")
        ag_event = RunErrorEvent(
            type=EventType.RUN_ERROR,
            thread_id=self.thread_id,
            run_id=self.run_id,
            error=str(error)
        )
        self.queue.put_nowait(ag_event)
        print(f"✅ [SYNC] on_tool_error 事件已放入队列")

    # ===== Agent 层事件（ReAct 动作与结束） =====
    async def on_agent_action(self, action, **kwargs: Any) -> None:
        print(f"🤖 on_agent_action 被调用: {action}")
        # action包含 tool / tool_input / log（思考片段）
        try:
            if getattr(action, "log", None):
                print(f"📝 Agent log: {action.log[:50]}...")
                ag_event = ThinkingTextMessageContentEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                    delta=action.log
                )
                await self.queue.put(ag_event)
                print(f"✅ Agent log 事件已放入队列")
            
            tool_name = getattr(action, "tool", None)
            tool_input = getattr(action, "tool_input", None)
            if tool_name:
                print(f"🔧 Agent tool: {tool_name}, input: {tool_input[:50] if tool_input else 'None'}...")
                ag_event = ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=f"tool_{self.run_id}",
                    tool_call_name=tool_name,
                    parent_message_id=self.run_id
                )
                await self.queue.put(ag_event)
                print(f"✅ Agent tool 事件已放入队列")
        except Exception as e:
            print(f"❌ on_agent_action 错误: {e}")

    async def on_agent_finish(self, finish, **kwargs: Any) -> None:
        print(f"🏁 on_agent_finish 被调用: {finish}")
        try:
            output = None
            if hasattr(finish, "return_values") and isinstance(finish.return_values, dict):
                output = finish.return_values.get("output")
            
            if output:
                print(f"📝 Agent output: {output[:50]}...")
                ag_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=self.run_id,
                    delta=output
                )
                await self.queue.put(ag_event)
                print(f"✅ Agent output 事件已放入队列")
        except Exception as e:
            print(f"❌ on_agent_finish 错误: {e}")


