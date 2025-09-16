# 多智能体系统（增强版，含 ag-ui 适配）

## 🧭 概述
- 基于 LangChain 的多智能体系统，支持 ReAct、Plan-and-Execute、工具驱动等智能体形式。
- 统一 YAML 配置（config.yaml），代码默认值由 utils.config 提供，YAML 优先级更高。
- 内置联网搜索（Serper）、代码执行（支持沙箱）、监控与性能优化。
- 提供 ag-ui 适配层：HTTP/SSE/WebSocket 三种接口形态，统一事件流（思考、计划、工具调用、最终答案等）。

---

## 📁 项目结构
```
multi-agent/
├─ agents/                 # 各类智能体（ReAct / Plan-Execute / Tool）
├─ coordinator/            # 协调器（顺序执行等）
├─ memory/                 # 记忆组件与管理
├─ router/                 # 路由器（规则/LLM/自定义）
├─ schema/                 # Task/Message/Result 等数据结构
├─ server/
│  ├─ ag_ui_adapter.py     # ag-ui 适配（HTTP/SSE/WebSocket）
│  └─ agui_callback.py     # 回调事件 → SSE/WS 推送
├─ tools/
│  ├─ langchain_tools.py   # 工具集（serper_search、code_execution 等）
│  └─ api_tool.py          # 通用 API 调用工具（当前默认未启用）
├─ utils/                  # 日志、监控、配置等
├─ enhanced_main.py        # 增强版系统入口（演示/性能测试）
├─ config.yaml             # 统一 YAML 配置（优先级高于默认值）
├─ README.md               # 本文档（已合并增强文档）
└─ requirements.txt / uv.lock
```

---

## ⚙️ 配置（config.yaml）
- YAML 覆盖 `utils/config.py` 的默认值；缺失项回落默认。
- 结构为嵌套键（如 `llm.base_url`），不要使用扁平键（如 `llm_base_url`）。
- `llm.base_url` 建议到 `/v1` 结尾，不包含 `/chat/completions` 或 `/completions`。

示例：
```yaml
system_name: "多智能体系统"
debug: false
log_level: "INFO"

llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  api_key: "${OPENAI_API_KEY}"
  # base_url: https://your-proxy.example.com/v1
  temperature: 0.7
  max_tokens: 1000

coordinator:
  type: "sequential"
  max_parallel_tasks: 5
  timeout: 300

# 第三方
serper_api_key: "${SERPER_API_KEY}"
```

---

## 🔌 工具
- serper_search（联网搜索）
  - 读取 `serper_api_key`（优先配置，其次环境变量），默认 5 秒冷却与简单去重。
- code_execution_tool（代码执行）
  - 默认 `sandbox: "restricted"` 使用 RestrictedPython 沙箱；可设 `sandbox: "none"` 用子进程执行（仅适合受信任环境）。
  - 返回 `{ success, return_code, stdout, stderr }` 或 `{ success:false, error }`。

---

## 🎛️ ag-ui 适配
提供 FastAPI 适配层，统一对接 HTTP / SSE / WebSocket，并推送执行轨迹事件。

### 路由
- POST `/agui/chat` 一次性响应
- POST `/agui/chat/stream` SSE 单向流
- WS `/agui/chat/ws` 双向流

请求（HTTP/SSE）
```json
{
  "session_id": "s1",
  "messages": [{"role": "user", "content": "帮我搜索 2024 AI 趋势"}],
  "priority": "normal"
}
```
响应（HTTP）
```json
{ "session_id": "s1", "content": "最终答案..." }
```

SSE 事件示例
```
event: start
data: {"status":"started"}

event: plan_start
data: {"task_id":"task_s1","title":"AGUI Chat"}

event: plan_end
data: {"task_id":"task_s1"}

event: execute_start
data: {"task_id":"task_s1","agent_id":"react_default"}

event: thought
data: {"text":"我需要先搜索…"}

event: tool_start
data: {"tool":"serper_search","input":"人工智能 最新信息 2024"}

event: tool_end
data: {"output":"{...serper响应...}"}

event: message
data: {"session_id":"s1","content":"最终答案..."}

event: end
data: {"status":"completed"}
```

WS 事件（与 SSE data 一致）
```json
{"event":"start","session_id":"s1"}
{"event":"plan_start","task_id":"task_s1","title":"AGUI Chat"}
{"event":"thought","text":"我需要先搜索…"}
{"event":"tool_start","tool":"serper_search","input":"人工智能 最新信息 2024"}
{"event":"tool_end","output":"{...}"}
{"event":"message","session_id":"s1","content":"最终答案..."}
{"event":"end","session_id":"s1"}
```

运行适配层
```bash
uvicorn server.ag_ui_adapter:app --host 0.0.0.0 --port 8000 --reload
```

事件切面
- 计划：plan_start / plan_step（可扩展） / plan_end
- 执行：execute_start / thought / token（可选） / tool_start / tool_end / tool_error / message / end

---

## 🚀 快速开始
```bash
# 安装依赖（任选）
pip install -r requirements.txt
# 或使用 uv（推荐）
uv pip install -r requirements.txt

# 运行演示
python enhanced_main.py

# 启动 ag-ui 适配层
uvicorn server.ag_ui_adapter:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔍 特点
- 清晰分层：Agent、工具、协调器、路由、记忆各司其职
- 易扩展：新增 Agent/Tool/Coordinator 无需改动核心逻辑
- 可观测：日志、监控、事件流，便于前端实时呈现
