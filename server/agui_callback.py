"""
ag-ui 事件回调：
将 LangChain 执行过程中的关键事件（思考/逐token/工具调用）
统一转换为可推送给前端（SSE/WS）的事件对象。
使用官方 ag-ui-protocol 包统一事件格式。
"""

import asyncio
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler
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


class AguiEventStreamer(BaseCallbackHandler):
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
    def on_text(self, text: str, **kwargs: Any) -> None:
        # 思考片段（可用于在前端展示思维过程）
        ag_event = ThinkingTextMessageContentEvent(
            type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
            delta=text
        )
        self.queue.put_nowait(ag_event)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # 逐token 推送（可选开启）
        ag_event = TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id=self.run_id,
            delta=token
        )
        self.queue.put_nowait(ag_event)

    # ===== 工具调用 =====
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "tool"
        ag_event = ToolCallStartEvent(
            type=EventType.TOOL_CALL_START,
            tool_call_id=f"tool_{self.run_id}",
            tool_call_name=name,
            parent_message_id=self.run_id
        )
        self.queue.put_nowait(ag_event)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        ag_event = ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=self.run_id,
            tool_call_id=f"tool_{self.run_id}",
            content=output,
            role="tool"
        )
        self.queue.put_nowait(ag_event)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        ag_event = RunErrorEvent(
            type=EventType.RUN_ERROR,
            thread_id=self.thread_id,
            run_id=self.run_id,
            error=str(error)
        )
        self.queue.put_nowait(ag_event)

    # ===== Agent 层事件（ReAct 动作与结束） =====
    def on_agent_action(self, action, **kwargs: Any) -> None:
        # action包含 tool / tool_input / log（思考片段）
        try:
            if getattr(action, "log", None):
                ag_event = ThinkingTextMessageContentEvent(
                    type=EventType.THINKING_TEXT_MESSAGE_CONTENT,
                    delta=action.log
                )
                self.queue.put_nowait(ag_event)
            
            tool_name = getattr(action, "tool", None)
            tool_input = getattr(action, "tool_input", None)
            if tool_name:
                ag_event = ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=f"tool_{self.run_id}",
                    tool_call_name=tool_name,
                    parent_message_id=self.run_id
                )
                self.queue.put_nowait(ag_event)
        except Exception:
            pass

    def on_agent_finish(self, finish, **kwargs: Any) -> None:
        try:
            output = None
            if hasattr(finish, "return_values") and isinstance(finish.return_values, dict):
                output = finish.return_values.get("output")
            
            if output:
                ag_event = TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=self.run_id,
                    delta=output
                )
                self.queue.put_nowait(ag_event)
        except Exception:
            pass


