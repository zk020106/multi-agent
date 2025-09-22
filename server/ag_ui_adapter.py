from typing import Any, Dict, List, Optional, AsyncGenerator
import json
import asyncio
import logging
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

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
from ag_ui.encoder import EventEncoder

from enhanced_main import EnhancedMultiAgentSystem
from schema import Task, TaskPriority
from server.agui_callback import AguiEventStreamer


logger = logging.getLogger("ag_ui")

app = FastAPI(title="AG-UI 适配层", docs_url="/doc.html", redoc_url="/redoc", openapi_url="/openapi.json")


class ChatMessage(BaseModel):
    """聊天消息模型（与 ag-ui 对齐）。
    role: 消息角色（user/assistant/system）
    content: 消息内容
    """
    role: str
    content: str


class ChatRequest(BaseModel):
    """聊天请求模型。
    session_id: 会话ID
    messages: 历史消息列表
    stream: 是否期望流式（HTTP端点可忽略；SSE/WS天然流式）
    priority: 任务优先级（low/normal/high）
    """
    session_id: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    priority: Optional[str] = "normal"


class ChatResponse(BaseModel):
    """聊天响应模型（一次性HTTP返回）。"""
    session_id: str
    content: str
    metadata: Dict[str, Any] = {}


# 懒初始化的系统实例（进程级单例）
_system: Optional[EnhancedMultiAgentSystem] = None
_coordinator_id = "coord_http"


def _ensure_system() -> EnhancedMultiAgentSystem:
    global _system
    if _system is None:
        # 优先使用项目根目录 config.yaml
        cfg_path = os.path.join(os.getcwd(), "config.yaml")
        _system = EnhancedMultiAgentSystem(cfg_path if os.path.exists(cfg_path) else None)
        # 引导：创建多个智能体与顺序协调器
        try:
            from agents import AgentType
            # 创建ReAct智能体
            _system.create_agent(
                agent_type=AgentType.REACT,
                agent_id="react_default",
                name="默认推理智能体",
                description="通用推理与工具使用"
            )
            # 创建Plan智能体
            _system.create_agent(
                agent_type=AgentType.PLAN_EXECUTE,
                agent_id="plan_default",
                name="默认计划智能体",
                description="擅长制定计划和执行复杂任务"
            )
            # 创建Tool智能体
            _system.create_agent(
                agent_type=AgentType.TOOL,
                agent_id="tool_default",
                name="默认工具智能体",
                description="专门执行工具操作"
            )
            
            coord = _system.create_coordinator(
                coordinator_type="sequential",
                coordinator_id=_coordinator_id,
                name="HTTP协调器"
            )
            # 添加所有智能体到协调器
            _system.add_agent_to_coordinator(_coordinator_id, "react_default", role="reasoning")
            _system.add_agent_to_coordinator(_coordinator_id, "plan_default", role="planning")
            _system.add_agent_to_coordinator(_coordinator_id, "tool_default", role="execution")
        except Exception as e:
            logger.error(f"初始化系统失败: {e}")
    return _system


def _build_probe_payload() -> Dict[str, Any]:
    system = _ensure_system()
    cfg = system.config
    return  cfg.system_name + '系统启动成功!'


@app.get("/")
async def root_probe():
    """根路径探针，输出项目信息。"""
    return JSONResponse(_build_probe_payload())


@app.get("/agui/status")
async def agui_status():
    """获取系统状态，包括智能体信息。"""
    system = _ensure_system()
    status = system.get_system_status()
    return JSONResponse(status)


def _extract_user_prompt(messages: List[ChatMessage]) -> str:
    """提取用户最新一条消息作为任务描述；若无则拼接全部内容。"""
    for msg in reversed(messages):
        if msg.role.lower() == "user":
            return msg.content
    # 兜底：拼接所有消息内容
    return "\n".join([m.content for m in messages])


def _priority_from_str(value: Optional[str]) -> TaskPriority:
    mapping = {
        "low": TaskPriority.LOW,
        "normal": TaskPriority.NORMAL,
        "high": TaskPriority.HIGH,
    }
    return mapping.get((value or "normal").lower(), TaskPriority.NORMAL)


@app.post("/agui/chat")
async def agui_chat(req: ChatRequest):
    system = _ensure_system()
    prompt = _extract_user_prompt(req.messages)
    task = Task(id=f"task_{req.session_id}", title="AGUI Chat", description=prompt, priority=_priority_from_str(req.priority))
    result = await system.execute_task(_coordinator_id, task)
    content = result.data.get("output") if result and result.data else (result.error_message if result else "")
    return JSONResponse(ChatResponse(session_id=req.session_id, content=content or "").model_dump())


@app.post("/agui/chat/stream")
async def agui_chat_stream(req: ChatRequest):
    system = _ensure_system()
    prompt = _extract_user_prompt(req.messages)

    async def event_stream() -> AsyncGenerator[bytes, None]:
        queue: asyncio.Queue = asyncio.Queue()
        encoder = EventEncoder()
        message_id = f"msg_{req.session_id}_{int(time.time())}"
        
        # 创建事件流处理器，传入 session_id 和 message_id
        callbacks = [AguiEventStreamer(queue, session_id=req.session_id, message_id=message_id)]
        
        # 起始事件
        start_event = RunStartedEvent(
            thread_id=req.session_id,
            run_id=message_id
        )
        yield encoder.encode(start_event).encode("utf-8")
        
        # run task in background and stream callbacks
        task = Task(id=f"task_{req.session_id}", title="AGUI Chat", description=prompt, priority=_priority_from_str(req.priority))
        run_coro = system.execute_task(_coordinator_id, task, callbacks=callbacks)
        runner = asyncio.create_task(run_coro)

        try:
            while True:
                done, _ = await asyncio.wait({runner}, timeout=0.01)
                # drain queue
                while not queue.empty():
                    ag_event = await queue.get()
                    # 使用 EventEncoder 编码事件
                    yield encoder.encode(ag_event).encode("utf-8")
                if runner in done:
                    result = runner.result()
                    content = result.data.get("output") if result and result.data else (result.error_message if result else "")
                    
                    # 发送最终消息内容
                    if content:
                        final_event = TextMessageContentEvent(
                            message_id=message_id,
                            delta=content
                        )
                        yield encoder.encode(final_event).encode("utf-8")
                    break
        finally:
            # 结束事件
            end_event = RunFinishedEvent(
                thread_id=req.session_id,
                run_id=message_id
            )
            yield encoder.encode(end_event).encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.websocket("/agui/chat/ws")
async def agui_chat_ws(ws: WebSocket):
    await ws.accept()
    system = _ensure_system()
    try:
        while True:
            data = await ws.receive_json()
            session_id = data.get("session_id", "ws")
            messages = [ChatMessage(**m) for m in data.get("messages", [])]
            prompt = _extract_user_prompt(messages)
            task = Task(id=f"task_{session_id}", title="AGUI Chat", description=prompt, priority=_priority_from_str(data.get("priority")))

            message_id = f"msg_{session_id}_{int(time.time())}"
            
            # 发送开始事件
            start_event = RunStartedEvent(
                thread_id=session_id,
                run_id=message_id
            )
            await ws.send_json(start_event.model_dump())
            
            queue: asyncio.Queue = asyncio.Queue()
            callbacks = [AguiEventStreamer(queue, session_id=session_id, message_id=message_id)]
            run_coro = system.execute_task(_coordinator_id, task, callbacks=callbacks)
            runner = asyncio.create_task(run_coro)
            try:
                while True:
                    done, _ = await asyncio.wait({runner}, timeout=0.01)
                    while not queue.empty():
                        ag_event = await queue.get()
                        # 直接发送事件字典给前端
                        await ws.send_json(ag_event.model_dump())
                    if runner in done:
                        result = runner.result()
                        content = result.data.get("output") if result and result.data else (result.error_message if result else "")
                        
                        # 发送最终消息内容
                        if content:
                            final_event = TextMessageContentEvent(
                                message_id=message_id,
                                delta=content
                            )
                            await ws.send_json(final_event.model_dump())
                        break
            finally:
                # 发送结束事件
                end_event = RunFinishedEvent(
                    thread_id=session_id,
                    run_id=message_id
                )
                await ws.send_json(end_event.model_dump())
    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")
    except Exception as e:
        error_event = RunErrorEvent(
            thread_id=session_id if 'session_id' in locals() else "unknown",
            run_id=message_id if 'message_id' in locals() else None,
            error=str(e)
        )
        await ws.send_json(error_event.model_dump())

