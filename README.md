# 基于LangChain的多智能体系统

一个功能完整的多智能体系统，基于LangChain框架构建，支持多种智能体类型、工具集成、记忆管理和任务协调。

## 🌟 特性

- **多种智能体类型**：ReAct、Plan-and-Execute、工具驱动智能体
- **丰富的工具集成**：API调用、数据库操作、文档搜索、代码执行等
- **灵活的记忆管理**：支持内存、Redis、向量数据库等多种存储方式
- **智能任务协调**：顺序执行、并行执行、投票决策等协调策略
- **完整的中文支持**：所有注释和文档均为中文
- **基于LangChain**：充分利用LangChain生态系统的优势

## 📁 项目结构

```
multi_agent_system/
├── agents/                     # 智能体核心模块
│   ├── base_agent.py           # 智能体基类
│   ├── react_agent.py          # ReAct智能体
│   ├── plan_execute_agent.py   # 计划执行智能体
│   ├── tool_agent.py           # 工具驱动智能体
│   └── agent_factory.py        # 智能体工厂
├── tools/                      # 工具集成模块
│   └── langchain_tools.py      # LangChain工具实现
├── memory/                     # 记忆管理模块
│   ├── memory_manager.py       # 记忆管理器
│   ├── conversation_memory.py  # 对话记忆
│   ├── vector_memory.py        # 向量记忆
│   └── redis_memory.py         # Redis记忆
├── coordinator/                # 任务协调器模块
│   ├── base_coordinator.py     # 协调器基类
│   ├── sequential_coordinator.py # 顺序执行协调器
│   ├── parallel_coordinator.py  # 并行执行协调器
│   └── voting_coordinator.py    # 投票协调器
├── schema/                     # 数据结构定义
│   ├── message.py              # 消息对象
│   ├── task.py                 # 任务对象
│   └── result.py               # 结果对象
├── utils/                      # 工具函数模块
│   ├── logger.py               # 日志工具
│   ├── config.py               # 配置管理
│   └── async_utils.py          # 异步工具
├── main.py                     # 主程序入口
└── README.md                   # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. 运行演示

```bash
python main.py
```

## 💡 使用示例

### 创建智能体

```python
from agents import AgentFactory, AgentType
from tools import get_all_tools

# 创建智能体工厂
factory = AgentFactory()

# 创建ReAct智能体
react_agent = factory.create_react_agent(
    agent_id="react_001",
    name="推理智能体",
    description="擅长推理和问题解决",
    llm=llm,
    tools=get_all_tools()
)

# 创建计划执行智能体
plan_agent = factory.create_plan_execute_agent(
    agent_id="plan_001",
    name="计划智能体", 
    description="擅长制定和执行计划",
    llm=llm,
    tools=get_all_tools()
)
```

### 创建协调器

```python
from coordinator import SequentialCoordinator

# 创建顺序执行协调器
coordinator = SequentialCoordinator(
    coordinator_id="coord_001",
    name="主协调器"
)

# 添加智能体到协调器
coordinator.add_agent(react_agent, "reasoning")
coordinator.add_agent(plan_agent, "planning")
```

### 执行任务

```python
from schema import Task, TaskPriority

# 创建任务
task = Task(
    id="task_001",
    title="计算数学表达式",
    description="计算 (2 + 3) * 4 - 1",
    priority=TaskPriority.NORMAL
)

# 执行任务
result = await coordinator.coordinate(task)
print(f"执行结果: {result.data}")
```

## 🔧 配置说明

系统支持多种配置方式：

### 1. 环境变量配置

```bash
export MAS_LLM_PROVIDER="openai"
export MAS_LLM_MODEL="gpt-3.5-turbo"
export MAS_LLM_API_KEY="your-api-key"
export MAS_LOG_LEVEL="INFO"
```

### 2. 配置文件

创建 `config.yaml` 文件：

```yaml
system_name: "多智能体系统"
debug: false
log_level: "INFO"

llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  api_key: "your-api-key"
  temperature: 0.7
  max_tokens: 1000

memory:
  type: "buffer"
  max_memory_size: 1000

coordinator:
  type: "sequential"
  max_parallel_tasks: 5
  timeout: 300
```

## 🛠️ 工具说明

系统内置多种工具：

- **API工具**：HTTP API调用
- **数据库工具**：SQLite数据库操作
- **文档搜索工具**：文本文件搜索
- **代码执行工具**：Python代码执行
- **网络搜索工具**：DuckDuckGo搜索
- **计算器工具**：数学表达式计算

## 🧠 记忆管理

支持多种记忆存储方式：

- **内存存储**：快速访问，适合临时对话
- **Redis存储**：持久化存储，支持分布式
- **向量存储**：语义搜索，支持相似性检索

## 📊 监控和日志

系统提供完整的监控和日志功能：

- 彩色日志输出
- 智能体状态监控
- 任务执行统计
- 性能指标追踪

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

感谢 LangChain 团队提供的优秀框架，以及所有开源贡献者的支持。

---

**注意**：使用前请确保已正确配置 OpenAI API 密钥，或使用其他兼容的 LLM 服务。
