"""
多智能体系统主程序入口
演示如何使用基于LangChain的多智能体系统
"""

import asyncio
import os
from typing import List

from langchain_community.llms.openai import BaseOpenAI
from langchain_openai import OpenAIEmbeddings

from agents import AgentFactory, AgentType
from coordinator import SequentialCoordinator
from memory import MemoryManager
from schema import Task, TaskPriority
from tools import get_all_tools
from utils import setup_logging, get_config


class MultiAgentSystem:
    """
    多智能体系统主类
    
    整合所有组件，提供统一的多智能体系统接口
    """
    
    def __init__(self, config_path: str = None, llm=None, embeddings=None):
        """
        初始化多智能体系统
        
        Args:
            config_path: 配置文件路径
            llm: 自定义LLM实例，如果为None则使用配置中的LLM
            embeddings: 自定义嵌入模型实例，如果为None则使用配置中的嵌入模型
        """
        # 设置日志
        setup_logging(level="INFO", use_colors=True)
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            from utils import load_config
            self.config = load_config(config_path)
        else:
            self.config = get_config()
        
        # 初始化组件
        self.agent_factory = AgentFactory()
        self.memory_manager = MemoryManager()
        self.coordinators = {}
        
        # 初始化LLM
        if llm is not None:
            self.llm = llm
            print(f"✅ 使用自定义LLM: {type(llm).__name__}")
        else:
            self.llm = self._init_llm()
        
        # 初始化嵌入模型
        if embeddings is not None:
            self.embeddings = embeddings
            print(f"✅ 使用自定义嵌入模型: {type(embeddings).__name__}")
        else:
            self.embeddings = self._init_embeddings()
        
        # 初始化工具
        self.tools = get_all_tools()
        
        print("🚀 多智能体系统初始化完成！")
    
    def _init_llm(self):
        """初始化大语言模型"""
        try:
            # 使用OpenAI模型
            llm = BaseOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                api_key=self.config.llm.api_key
            )
            print(f"✅ LLM初始化成功: {self.config.llm.model}")
            return llm
        except Exception as e:
            print(f"❌ LLM初始化失败: {e}")
            # 使用模拟LLM
            from langchain_community.llms import FakeListLLM
            return FakeListLLM(responses=["这是一个模拟响应"])
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            embeddings = OpenAIEmbeddings(
                api_key=self.config.llm.api_key
            )
            print("✅ 嵌入模型初始化成功")
            return embeddings
        except Exception as e:
            print(f"❌ 嵌入模型初始化失败: {e}")
            return None
    
    def create_agent(
        self,
        agent_type: AgentType,
        agent_id: str,
        name: str,
        description: str,
        tools: List = None,
        memory_type: str = "buffer"
    ):
        """
        创建智能体
        
        Args:
            agent_type: 智能体类型
            agent_id: 智能体ID
            name: 智能体名称
            description: 智能体描述
            tools: 工具列表
            memory_type: 记忆类型
        """
        # 创建记忆
        memory = self.memory_manager.create_memory(
            agent_id=agent_id,
            memory_type=memory_type,
            llm=self.llm
        )
        
        # 创建智能体
        agent = self.agent_factory.create_agent(
            agent_type=agent_type,
            agent_id=agent_id,
            name=name,
            description=description,
            llm=self.llm,
            tools=tools or self.tools,
            memory=memory
        )
        
        print(f"✅ 智能体创建成功: {name} ({agent_type.value})")
        return agent
    
    def create_coordinator(self, coordinator_type: str, coordinator_id: str, name: str):
        """
        创建协调器
        
        Args:
            coordinator_type: 协调器类型
            coordinator_id: 协调器ID
            name: 协调器名称
        """
        if coordinator_type == "sequential":
            coordinator = SequentialCoordinator(coordinator_id, name)
        else:
            raise ValueError(f"不支持的协调器类型: {coordinator_type}")
        
        self.coordinators[coordinator_id] = coordinator
        print(f"✅ 协调器创建成功: {name} ({coordinator_type})")
        return coordinator
    
    def add_agent_to_coordinator(self, coordinator_id: str, agent_id: str, role: str = "general"):
        """
        将智能体添加到协调器
        
        Args:
            coordinator_id: 协调器ID
            agent_id: 智能体ID
            role: 智能体角色
        """
        coordinator = self.coordinators.get(coordinator_id)
        agent = self.agent_factory.get_agent(agent_id)
        
        if coordinator and agent:
            coordinator.add_agent(agent, role)
            print(f"✅ 智能体 {agent.name} 已添加到协调器 {coordinator.name}")
        else:
            print("❌ 协调器或智能体不存在")
    
    async def execute_task(self, coordinator_id: str, task: Task):
        """
        执行任务
        
        Args:
            coordinator_id: 协调器ID
            task: 任务对象
        """
        coordinator = self.coordinators.get(coordinator_id)
        if not coordinator:
            print(f"❌ 协调器 {coordinator_id} 不存在")
            return None
        
        print(f"🎯 开始执行任务: {task.title}")
        result = await coordinator.coordinate(task)
        
        if result.status.value == "success":
            print(f"✅ 任务执行成功: {task.title}")
        else:
            print(f"❌ 任务执行失败: {task.title} - {result.error_message}")
        
        return result
    
    def get_system_status(self):
        """获取系统状态"""
        status = {
            "agents": self.agent_factory.get_all_agents_info(),
            "coordinators": {cid: coord.get_status() for cid, coord in self.coordinators.items()},
            "memory_stats": self.memory_manager.get_memory_stats()
        }
        return status


async def demo():
    """演示多智能体系统"""
    print("🎬 开始多智能体系统演示")
    
    # 初始化系统
    system = MultiAgentSystem()
    
    # 创建智能体
    react_agent = system.create_agent(
        agent_type=AgentType.REACT,
        agent_id="react_001",
        name="推理智能体",
        description="擅长推理和问题解决的智能体",
        tools=system.tools[:3]  # 使用前3个工具
    )
    
    plan_agent = system.create_agent(
        agent_type=AgentType.PLAN_EXECUTE,
        agent_id="plan_001", 
        name="计划智能体",
        description="擅长制定计划和执行复杂任务的智能体",
        tools=system.tools[3:]  # 使用后几个工具
    )
    
    tool_agent = system.create_agent(
        agent_type=AgentType.TOOL,
        agent_id="tool_001",
        name="工具智能体", 
        description="专门执行工具操作的智能体",
        tools=system.tools
    )
    
    # 创建协调器
    coordinator = system.create_coordinator(
        coordinator_type="sequential",
        coordinator_id="coord_001",
        name="主协调器"
    )
    
    # 添加智能体到协调器
    system.add_agent_to_coordinator("coord_001", "react_001", "reasoning")
    system.add_agent_to_coordinator("coord_001", "plan_001", "planning")
    system.add_agent_to_coordinator("coord_001", "tool_001", "execution")
    
    # 创建任务
    tasks = [
        Task(
            id="task_001",
            title="计算数学表达式",
            description="计算表达式 (2 + 3) * 4 - 1",
            priority=TaskPriority.NORMAL
        ),
        Task(
            id="task_002", 
            title="搜索信息",
            description="搜索关于人工智能的最新信息",
            priority=TaskPriority.HIGH
        ),
        Task(
            id="task_003",
            title="执行Python代码",
            description="执行代码: print('Hello, Multi-Agent System!')",
            priority=TaskPriority.LOW
        )
    ]
    
    # 执行任务
    print("\n📋 开始执行任务列表")
    for task in tasks:
        result = await system.execute_task("coord_001", task)
        if result:
            print(f"📊 任务结果: {result.data}")
        print("-" * 50)
    
    # 显示系统状态
    print("\n📈 系统状态:")
    status = system.get_system_status()
    print(f"智能体数量: {len(status['agents'])}")
    print(f"协调器数量: {len(status['coordinators'])}")
    print(f"记忆统计: {status['memory_stats']}")
    
    print("\n🎉 演示完成！")


if __name__ == "__main__":
    # 设置环境变量（如果没有配置文件）
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  未设置OPENAI_API_KEY环境变量，将使用模拟LLM")
    
    # 运行演示
    asyncio.run(demo())
