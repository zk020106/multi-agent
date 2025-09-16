"""
ag-ui 事件回调：
将 LangChain 执行过程中的关键事件（思考/逐token/工具调用）
统一转换为可推送给前端（SSE/WS）的事件对象。
"""

import asyncio
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler


class AguiEventStreamer(BaseCallbackHandler):
    """ag-ui 事件流回调处理器。

    - 接收 LangChain 执行回调
    - 通过异步队列输出统一事件，供 SSE / WebSocket 推送层消费
    """

    def __init__(self, queue: asyncio.Queue):
        # 由适配层传入的异步队列，用于承载事件
        self.queue = queue

    async def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        # 统一事件格式：{ "event": <事件名>, ...payload }
        await self.queue.put({"event": event, **payload})

    # 提供给协调器/执行器的通用触发接口
    async def emit(self, event: str, payload: Dict[str, Any]) -> None:
        await self._emit(event, payload)

    # ===== 链路级回调 =====
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        await self._emit("chain_start", {"inputs": inputs})

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        await self._emit("chain_end", {"outputs": outputs})

    async def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        await self._emit("chain_error", {"error": str(error)})

    # ===== LLM 输出（思考/逐token） =====
    async def on_text(self, text: str, **kwargs: Any) -> None:
        # 思考片段（可用于在前端展示思维过程）
        await self._emit("thought", {"text": text})

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # 逐token 推送（可选开启）
        await self._emit("token", {"token": token})

    # ===== 工具调用 =====
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "tool"
        await self._emit("tool_start", {"tool": name, "input": input_str})

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        await self._emit("tool_end", {"output": output})

    async def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        await self._emit("tool_error", {"error": str(error)})


