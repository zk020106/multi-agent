_# 多智能体系统架构（LangChain）

## 目录结构（Tree 架构）
```graphql
multi_agent_system/
│
├── agents/                     # 智能体核心模块
│   ├── base_agent.py           # 基类 Agent，定义统一接口：receive_message, act, return_result
│   ├── react_agent.py          # ReAct 风格 Agent
│   ├── plan_execute_agent.py   # Plan-and-Execute Agent
│   ├── tool_agent.py           # 可调用工具的 Agent
│   └── agent_factory.py        # Agent 工厂，创建不同类型 Agent
│
├── roles/                      # 角色管理模块
│   ├── role.py                 # Role 类，定义角色属性：name, description, tools, permissions
│   ├── role_registry.py        # 角色注册与管理
│   └── role_loader.py          # 角色配置加载（JSON/YAML）
│
├── tools/                      # 工具集成模块
│   ├── base_tool.py            # 工具基类，统一接口：execute
│   ├── api_tool.py             # 调用外部 API 的工具
│   ├── db_tool.py              # 数据库查询工具
│   ├── doc_search_tool.py      # 文档检索工具
│   └── code_execution_tool.py  # 代码执行工具
│
├── coordinator/                # 任务协调器模块
│   ├── base_coordinator.py     # 基类 Coordinator，定义 task 分发、结果聚合接口
│   ├── sequential_coordinator.py # 顺序执行策略
│   ├── parallel_coordinator.py  # 并行执行策略
│   └── voting_coordinator.py    # 投票/冲突决策策略
│
├── router/                     # 消息路由模块
│   ├── router.py               # 消息路由主类
│   ├── rule_router.py          # 基于规则的路由
│   ├── llm_router.py           # 基于 LLM 判断路由
│   └── custom_router.py        # 可扩展自定义路由
│
├── memory/                     # 上下文与记忆管理
│   ├── base_memory.py          # Memory 基类
│   ├── in_memory.py            # 内存存储
│   ├── redis_memory.py         # Redis 存储
│   └── vector_memory.py        # 向量存储（Milvus/FAISS）
│
├── schema/                     # 数据结构和类型定义
│   ├── message.py              # 消息对象定义
│   ├── task.py                 # 任务对象定义
│   └── result.py               # 结果对象定义
│
├── utils/                      # 工具函数模块
│   ├── logger.py               # 日志封装
│   ├── config.py               # 配置读取
│   └── async_utils.py          # 异步调度辅助
│
├── tests/                      # 测试模块
│   ├── test_agents.py
│   ├── test_coordinator.py
│   └── test_tools.py
│
├── main.py                     # 系统启动入口/示例
└── requirements.txt
```

---

## 模块功能点

### 1. `agents/`
- 封装智能体能力，统一接口。
- 功能点：
  - 接收消息、解析意图
  - 调用工具或 LLM
  - 返回结果或反馈
  - 支持日志记录与上下文记忆
  - 支持 Agent 类型扩展（ReAct、Plan-and-Execute、工具驱动）

### 2. `roles/`
- 管理不同角色的职责和权限。
- 功能点：
  - 定义角色属性（名称、职责、工具权限）
  - 注册、加载、更新角色
  - 支持角色与 Agent 的绑定

### 3. `tools/`
- 提供可调用功能或外部服务接口。
- 功能点：
  - 封装 API / DB / 文档搜索 / 代码执行
  - 统一接口 `execute(input: dict) -> dict`
  - 可扩展自定义工具
  - 支持工具权限控制

### 4. `coordinator/`
- 管理任务分配和结果聚合。
- 功能点：
  - 分解复杂任务到多 Agent
  - 聚合多个 Agent 的结果
  - 支持不同执行策略（顺序、并行、投票）
  - 错误处理与重试机制

### 5. `router/`
- 消息智能分发。
- 功能点：
  - 根据规则或 LLM 分配任务
  - 支持自定义路由策略
  - 可插拔扩展，支持多种路由算法

### 6. `memory/`
- 保存上下文、历史消息和任务状态。
- 功能点：
  - 多种存储策略：内存、Redis、向量数据库
  - 支持多轮对话记忆
  - 支持检索和历史上下文注入

### 7. `schema/`
- 数据结构统一定义。
- 功能点：
  - 消息对象（发送者、内容、时间、类型）
  - 任务对象（任务ID、任务内容、优先级）
  - 结果对象（执行状态、结果内容、日志）

### 8. `utils/`
- 系统辅助功能
- 功能点：
  - 日志统一封装
  - 异步工具函数
  - 配置文件管理

---

## 系统特点
- **清晰分层**：Agent、工具、协调器、路由、记忆各司其职
- **易扩展**：增加新 Agent / Tool / Coordinator 无需改动核心逻辑
- **可维护**：日志、测试和工具函数独立，方便系统调试和升级_
