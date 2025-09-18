"""
增强版多智能体系统主程序
整合所有改进功能：智能选择、错误处理、配置管理、性能优化、监控
"""

import asyncio
import os
import time
from typing import List, Any
import logging

from langchain_community.llms.openai import BaseOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from agents import AgentFactory, AgentType
from coordinator import SequentialCoordinator
from coordinator.smart_agent_selector import SelectionStrategy
from memory import MemoryManager
from schema import Task, TaskPriority
from tools import get_all_tools
from utils import setup_logging
from utils.config import load_config, set_config
from utils.monitoring import MonitoringManager
from utils.performance_optimizer import PerformanceOptimizer, TaskPriority as PerfTaskPriority


class EnhancedMultiAgentSystem:
    """
    增强版多智能体系统主类
    
    整合所有改进功能：
    - 智能智能体选择
    - 增强错误处理
    - 简化配置管理
    - 性能优化
    - 监控和可观测性
    """
    
    def __init__(self, config_path: str = None, llm=None, embeddings=None):
        """
        初始化增强版多智能体系统
        
        Args:
            config_path: 配置文件路径
            llm: 自定义LLM实例
            embeddings: 自定义嵌入模型实例
        """
        # 设置日志
        setup_logging(level="INFO", use_colors=True)
        self.logger = logging.getLogger("enhanced_main")
        
        # 加载配置（使用 utils.config 的配置结构）
        cfg_path = config_path if config_path else str(os.path.join(os.getcwd(), "config.yaml"))
        self.config = load_config(cfg_path)
        # 设置为全局配置，供工具等组件统一读取
        set_config(self.config)
        
        # 初始化组件
        self.agent_factory = AgentFactory()
        self.memory_manager = MemoryManager()
        self.coordinators = {}
        
        # 初始化LLM
        if llm is not None:
            self.llm = llm
            self.logger.info(f"✅ 使用自定义LLM: {type(llm).__name__}")
        else:
            self.llm = self._init_llm()
        
        # 初始化嵌入模型
        if embeddings is not None:
            self.embeddings = embeddings
            self.logger.info(f"✅ 使用自定义嵌入模型: {type(embeddings).__name__}")
        else:
            self.embeddings = self._init_embeddings()
        
        # 初始化工具
        self.tools = get_all_tools()
        
        # 初始化性能优化器
        self.performance_optimizer = PerformanceOptimizer(
            max_agents=self.config.max_agents,
            min_agents=2,
            max_queue_size=1000,
            max_concurrent_tasks=self.config.coordinator.max_parallel_tasks
        )
        
        # 初始化监控管理器
        self.monitoring_manager = MonitoringManager(enable_system_metrics=True)
        self.monitoring_manager.add_agent_health_checker({})  # 稍后更新
        
        self.logger.info("🚀 增强版多智能体系统初始化完成！")
        self.logger.info(f"📊 配置摘要: provider={self.config.llm.provider}, model={self.config.llm.model}, memory={self.config.memory.type}, coordinator={self.config.coordinator.type}")
    
    def _init_llm(self):
        """初始化大语言模型"""
        try:
            # 使用 ChatOpenAI 对话模型
            llm_kwargs = {
                "model": self.config.llm.model,
                "temperature": self.config.llm.temperature,
                "api_key": self.config.llm.api_key,
                "max_tokens": self.config.llm.max_tokens,
            }
            if self.config.llm.base_url:
                llm_kwargs["base_url"] = self.config.llm.base_url

            llm = ChatOpenAI(**llm_kwargs, streaming=True)
            self.logger.info(f"✅ LLM初始化成功: {self.config.llm.model}")
            return llm
        except Exception as e:
            self.logger.error(f"❌ LLM初始化失败: {e}")
            # 使用模拟LLM
            from langchain_community.llms import FakeListLLM
            return FakeListLLM(responses=["这是一个模拟响应"])
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            emb_kwargs = {"api_key": self.config.llm.api_key}
            if self.config.llm.base_url:
                emb_kwargs["base_url"] = self.config.llm.base_url
            embeddings = OpenAIEmbeddings(**emb_kwargs)
            self.logger.info("✅ 嵌入模型初始化成功")
            return embeddings
        except Exception as e:
            self.logger.error(f"❌ 嵌入模型初始化失败: {e}")
            return None

    def create_agent(
        self,
        agent_type: AgentType,
        agent_id: str,
        name: str,
        description: str,
        tools: List = None,
        memory_type: str = "buffer",
        selection_strategy: SelectionStrategy = SelectionStrategy.HYBRID
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
            selection_strategy: 选择策略
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
        
        # 添加到性能优化器
        self.performance_optimizer.add_agent(agent_id, agent)
        
        self.logger.info(f"✅ 智能体创建成功: {name} ({agent_type.value})")
        return agent
    
    def create_coordinator(
        self, 
        coordinator_type: str, 
        coordinator_id: str, 
        name: str,
        selection_strategy: SelectionStrategy = SelectionStrategy.HYBRID
    ):
        """
        创建协调器
        
        Args:
            coordinator_type: 协调器类型
            coordinator_id: 协调器ID
            name: 协调器名称
            selection_strategy: 智能体选择策略
        """
        if coordinator_type == "sequential":
            coordinator = SequentialCoordinator(
                coordinator_id=coordinator_id, 
                name=name,
                selection_strategy=selection_strategy
            )
        else:
            raise ValueError(f"不支持的协调器类型: {coordinator_type}")
        
        self.coordinators[coordinator_id] = coordinator
        self.logger.info(f"✅ 协调器创建成功: {name} ({coordinator_type})")
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
            self.logger.info(f"✅ 智能体 {agent.name} 已添加到协调器 {coordinator.name}")
        else:
            self.logger.error("❌ 协调器或智能体不存在")
    
    async def execute_task(self, coordinator_id: str, task: Task, callbacks: List[Any] = None):
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
        
        self.logger.info(f"🎯 开始执行任务: {task.title}")
        
        # 记录任务开始时间
        start_time = time.time()
        
        try:
            result = await coordinator.coordinate(task, callbacks=callbacks)
            execution_time = time.time() - start_time
            
            # 记录监控指标
            success = result.status.value == "success"
            self.monitoring_manager.record_task_metric(task.id, execution_time, success)
            
            if success:
                self.logger.info(f"✅ 任务执行成功: {task.title}")
            else:
                self.logger.error(f"❌ 任务执行失败: {task.title} - {result.error_message}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.monitoring_manager.record_task_metric(task.id, execution_time, False)
            self.logger.error(f"❌ 任务执行异常: {task.title} - {str(e)}")
            return None
    
    async def execute_task_with_optimization(self, task: Task, priority: PerfTaskPriority = PerfTaskPriority.NORMAL):
        """
        使用性能优化器执行任务
        
        Args:
            task: 任务对象
            priority: 任务优先级
        """
        self.logger.info(f"🎯 使用性能优化器执行任务: {task.title}")
        
        # 提交任务到性能优化器
        success = await self.performance_optimizer.submit_task(
            task_id=task.id,
            priority=priority,
            task_data=task,
            estimated_duration=30.0
        )
        
        if not success:
            self.logger.error(f"❌ 任务提交失败: {task.title}")
            return None
        
        # 处理任务
        result = await self.performance_optimizer.process_next_task()
        
        if result:
            self.logger.info(f"✅ 任务处理成功: {task.title}")
        else:
            self.logger.error(f"❌ 任务处理失败: {task.title}")
        
        return result
    
    def get_system_status(self):
        """获取系统状态"""
        # 更新监控管理器中的智能体信息
        agents = {agent_id: agent for agent_id, agent in self.agent_factory._created_agents.items()}
        self.monitoring_manager.add_agent_health_checker(agents)
        
        status = {
            "agents": self.agent_factory.get_all_agents_info(),
            "coordinators": {cid: coord.get_status() for cid, coord in self.coordinators.items()},
            "memory_stats": self.memory_manager.get_memory_stats(),
            "performance_metrics": self.performance_optimizer.get_performance_metrics(),
            "monitoring_metrics": self.monitoring_manager.get_metrics_summary(),
            "config_summary": {
                "system_name": self.config.system_name,
                "debug": self.config.debug,
                "log_level": self.config.log_level,
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "memory_type": self.config.memory.type,
                "coordinator_type": self.config.coordinator.type,
                "max_agents": self.config.max_agents,
                "max_parallel_tasks": self.config.coordinator.max_parallel_tasks
            }
        }
        return status
    
    def start_monitoring(self, dashboard_port: int = 8080):
        """启动监控"""
        self.monitoring_manager.start_monitoring(dashboard_port)
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_manager.stop_monitoring()
    
    def optimize_performance(self):
        """执行性能优化"""
        return self.performance_optimizer.optimize_performance()


async def enhanced_demo():
    """增强版多智能体系统演示"""
    print("🎬 开始增强版多智能体系统演示")
    
    # 初始化系统
    system = EnhancedMultiAgentSystem()
    
    # 启动监控
    system.start_monitoring(8080)
    
    # 创建智能体
    react_agent = system.create_agent(
        agent_type=AgentType.REACT,
        agent_id="react_001",
        name="推理智能体",
        description="擅长推理和问题解决的智能体",
        tools=system.tools[:3]
    )
    
    plan_agent = system.create_agent(
        agent_type=AgentType.PLAN_EXECUTE,
        agent_id="plan_001", 
        name="计划智能体",
        description="擅长制定计划和执行复杂任务的智能体",
        tools=system.tools[3:]
    )
    
    tool_agent = system.create_agent(
        agent_type=AgentType.TOOL,
        agent_id="tool_001",
        name="工具智能体", 
        description="专门执行工具操作的智能体",
        tools=system.tools
    )
    
    # 创建协调器（使用混合选择策略）
    coordinator = system.create_coordinator(
        coordinator_type="sequential",
        coordinator_id="coord_001",
        name="主协调器",
        selection_strategy=SelectionStrategy.HYBRID
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
            description="执行代码: print('Hello, Enhanced Multi-Agent System!')",
            priority=TaskPriority.LOW
        )
    ]
    
    # 执行任务（使用协调器）
    print("\n📋 开始执行任务列表（协调器模式）")
    for task in tasks:
        result = await system.execute_task("coord_001", task)
        if result:
            print(f"📊 任务结果: {result.data}")
        print("-" * 50)
    
    # 执行任务（使用性能优化器）
    print("\n📋 开始执行任务列表（性能优化模式）")
    for task in tasks:
        result = await system.execute_task_with_optimization(task, PerfTaskPriority.NORMAL)
        if result:
            print(f"📊 优化结果: {result}")
        print("-" * 50)
    
    # 显示系统状态
    print("\n📈 系统状态:")
    status = system.get_system_status()
    print(f"智能体数量: {len(status['agents'])}")
    print(f"协调器数量: {len(status['coordinators'])}")
    print(f"性能指标: {status['performance_metrics']}")
    print(f"监控指标: {status['monitoring_metrics']}")
    
    # 执行性能优化
    print("\n🔧 执行性能优化:")
    optimization_result = system.optimize_performance()
    print(f"优化结果: {optimization_result}")


async def performance_test():
    """性能测试"""
    print("🧪 开始性能测试")
    
    system = EnhancedMultiAgentSystem()
    
    # 创建多个智能体
    for i in range(5):
        agent = system.create_agent(
            agent_type=AgentType.TOOL,
            agent_id=f"test_agent_{i}",
            name=f"测试智能体{i}",
            description=f"用于性能测试的智能体{i}",
            tools=system.tools[:2]
        )
    
    # 创建协调器
    coordinator = system.create_coordinator(
        coordinator_type="sequential",
        coordinator_id="perf_test_coord",
        name="性能测试协调器",
        selection_strategy=SelectionStrategy.PERFORMANCE_BASED
    )
    
    # 添加智能体到协调器
    for i in range(5):
        system.add_agent_to_coordinator("perf_test_coord", f"test_agent_{i}", "test")
    
    # 创建大量任务
    tasks = []
    for i in range(20):
        task = Task(
            id=f"perf_task_{i}",
            title=f"性能测试任务{i}",
            description=f"这是第{i}个性能测试任务",
            priority=TaskPriority.NORMAL
        )
        tasks.append(task)
    
    # 并发执行任务
    print("🚀 开始并发执行任务...")
    start_time = time.time()
    
    # 使用asyncio.gather并发执行
    results = await asyncio.gather(
        *[system.execute_task("perf_test_coord", task) for task in tasks],
        return_exceptions=True
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 统计结果
    successful_tasks = sum(1 for result in results if result and not isinstance(result, Exception))
    failed_tasks = len(results) - successful_tasks
    
    print(f"📊 性能测试结果:")
    print(f"总任务数: {len(tasks)}")
    print(f"成功任务数: {successful_tasks}")
    print(f"失败任务数: {failed_tasks}")
    print(f"总执行时间: {total_time:.2f}秒")
    print(f"平均任务时间: {total_time/len(tasks):.2f}秒")
    print(f"吞吐量: {len(tasks)/total_time:.2f}任务/秒")
    
    # 显示系统状态
    status = system.get_system_status()
    print(f"性能指标: {status['performance_metrics']}")


if __name__ == "__main__":
    # 设置环境变量（如果没有配置文件）
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("⚠️  未设置OPENAI_API_KEY环境变量，将使用模拟LLM")
    
    # 运行增强版演示
    asyncio.run(enhanced_demo())
    
    # 运行性能测试
    print("\n" + "="*60)
    # asyncio.run(performance_test())
