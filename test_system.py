"""
多智能体系统测试脚本
验证系统基本功能
"""

import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import AgentFactory, AgentType
from tools import get_all_tools, CalculatorTool, APITool
from memory import MemoryManager
from coordinator import SequentialCoordinator
from schema import Task, TaskPriority
from utils import setup_logging


async def test_basic_functionality():
    """测试基本功能"""
    print("🧪 开始测试多智能体系统基本功能")
    
    # 设置日志
    setup_logging(level="INFO", use_colors=True)
    
    # 创建模拟LLM
    from langchain.llms.fake import FakeListLLM
    llm = FakeListLLM(responses=[
        "我需要使用计算器工具来计算这个表达式",
        "计算结果：23",
        "任务完成"
    ])
    
    # 创建智能体工厂
    factory = AgentFactory()
    
    # 创建工具
    tools = [CalculatorTool(), APITool()]
    
    # 创建ReAct智能体
    react_agent = factory.create_react_agent(
        agent_id="test_react_001",
        name="测试推理智能体",
        description="用于测试的推理智能体",
        llm=llm,
        tools=tools,
        max_iterations=3
    )
    
    print(f"✅ 智能体创建成功: {react_agent.name}")
    
    # 创建任务
    task = Task(
        id="test_task_001",
        title="计算数学表达式",
        description="计算表达式 (2 + 3) * 4 - 1",
        priority=TaskPriority.NORMAL
    )
    
    print(f"✅ 任务创建成功: {task.title}")
    
    # 执行任务
    print("🎯 开始执行任务...")
    result = react_agent.act(task)
    
    print(f"📊 任务执行结果:")
    print(f"   状态: {result.status.value}")
    print(f"   执行时间: {result.execution_time:.2f}秒")
    if result.data:
        print(f"   数据: {result.data}")
    if result.error_message:
        print(f"   错误: {result.error_message}")
    
    # 测试记忆管理
    print("\n🧠 测试记忆管理...")
    memory_manager = MemoryManager()
    
    # 创建记忆
    memory = memory_manager.create_memory("test_agent_001", "buffer")
    print("✅ 记忆组件创建成功")
    
    # 测试协调器
    print("\n🎭 测试协调器...")
    coordinator = SequentialCoordinator(
        coordinator_id="test_coord_001",
        name="测试协调器"
    )
    
    # 添加智能体到协调器
    coordinator.add_agent(react_agent, "test_role")
    print("✅ 智能体已添加到协调器")
    
    # 通过协调器执行任务
    print("🎯 通过协调器执行任务...")
    coord_result = await coordinator.coordinate(task)
    
    print(f"📊 协调器执行结果:")
    print(f"   状态: {coord_result.status.value}")
    print(f"   执行时间: {coord_result.execution_time:.2f}秒")
    
    # 显示系统状态
    print("\n📈 系统状态:")
    print(f"   智能体数量: {factory.get_agent_count()}")
    print(f"   协调器状态: {coordinator.get_status()}")
    print(f"   记忆统计: {memory_manager.get_memory_stats()}")
    
    print("\n🎉 所有测试完成！")


async def test_tools():
    """测试工具功能"""
    print("\n🔧 测试工具功能...")
    
    # 测试计算器工具
    calculator = CalculatorTool()
    result = calculator.run("(2 + 3) * 4 - 1")
    print(f"计算器工具测试: {result}")
    
    # 测试API工具（模拟）
    api_tool = APITool()
    # 注意：这里使用一个安全的测试URL
    result = api_tool.run("https://httpbin.org/get", "GET")
    print(f"API工具测试: {result[:100]}...")
    
    print("✅ 工具测试完成")


if __name__ == "__main__":
    print("🚀 多智能体系统测试")
    print("=" * 50)
    
    # 运行测试
    asyncio.run(test_basic_functionality())
    asyncio.run(test_tools())
    
    print("\n✨ 测试完成！系统运行正常。")
