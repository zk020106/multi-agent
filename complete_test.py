"""
完整的多智能体系统测试
包括所有模块的功能测试
"""

import asyncio
import os
import sys

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有模块导入"""
    print("🧪 测试模块导入...")
    
    try:
        from schema import Message, Task, Result, MessageType, TaskStatus, ResultStatus, TaskPriority
        print("✅ Schema模块导入成功")
    except Exception as e:
        print(f"❌ Schema模块导入失败: {e}")
        return False
    
    try:
        from utils import setup_logging, get_config
        print("✅ Utils模块导入成功")
    except Exception as e:
        print(f"❌ Utils模块导入失败: {e}")
        return False
    
    try:
        from tools import CalculatorTool, APITool
        print("✅ Tools模块导入成功")
    except Exception as e:
        print(f"❌ Tools模块导入失败: {e}")
        return False
    
    try:
        from agents import AgentFactory, AgentType
        print("✅ Agents模块导入成功")
    except Exception as e:
        print(f"❌ Agents模块导入失败: {e}")
        return False
    
    try:
        from memory import MemoryManager
        print("✅ Memory模块导入成功")
    except Exception as e:
        print(f"❌ Memory模块导入失败: {e}")
        return False
    
    try:
        from coordinator import SequentialCoordinator
        print("✅ Coordinator模块导入成功")
    except Exception as e:
        print(f"❌ Coordinator模块导入失败: {e}")
        return False
    
    try:
        from router import RuleRouter, LLMRouter, CustomRouter, RouterFactory, RouterType
        print("✅ Router模块导入成功")
    except Exception as e:
        print(f"❌ Router模块导入失败: {e}")
        return False
    
    return True

def test_basic_objects():
    """测试基本对象创建"""
    print("\n🧪 测试基本对象创建...")
    
    try:
        from schema import Task, TaskPriority, Message, MessageType
        
        # 创建任务
        task = Task(
            id="test_001",
            title="测试任务",
            description="这是一个测试任务",
            priority=TaskPriority.NORMAL
        )
        print(f"✅ 任务创建成功: {task.title}")
        
        # 创建消息
        message = Message(
            id="msg_001",
            sender="user",
            receiver="agent",
            content="测试消息",
            message_type=MessageType.USER_INPUT
        )
        print(f"✅ 消息创建成功: {message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本对象创建失败: {e}")
        return False

def test_tools():
    """测试工具"""
    print("\n🧪 测试工具...")
    
    try:
        from tools import CalculatorTool
        
        calculator = CalculatorTool()
        result = calculator.run("2 + 3")
        print(f"✅ 计算器工具测试成功: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具测试失败: {e}")
        return False

def test_router():
    """测试路由器"""
    print("\n🧪 测试路由器...")
    
    try:
        from router import RuleRouter, RouterFactory, RouterType
        from schema import Message, MessageType
        
        # 创建路由器工厂
        factory = RouterFactory()
        
        # 创建规则路由器
        router = factory.create_rule_router(
            router_id="test_router_001",
            name="测试路由器"
        )
        print("✅ 规则路由器创建成功")
        
        # 创建测试消息
        message = Message(
            id="test_msg_001",
            sender="user",
            receiver="",
            content="计算 2 + 3",
            message_type=MessageType.USER_INPUT
        )
        
        # 测试路由（没有注册智能体，应该返回空列表）
        target_agents = router.route_message(message)
        print(f"✅ 消息路由测试成功: {target_agents}")
        
        # 测试路由器统计
        stats = router.get_routing_stats()
        print(f"✅ 路由器统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 路由器测试失败: {e}")
        return False

def test_memory():
    """测试记忆管理"""
    print("\n🧪 测试记忆管理...")
    
    try:
        from memory import MemoryManager
        from schema import Message, MessageType
        
        # 创建记忆管理器
        memory_manager = MemoryManager()
        
        # 创建记忆
        memory = memory_manager.create_memory("test_agent_001", "buffer")
        print("✅ 记忆组件创建成功")
        
        # 创建测试消息
        message = Message(
            id="test_msg_001",
            sender="user",
            receiver="agent",
            content="测试记忆",
            message_type=MessageType.USER_INPUT
        )
        
        # 添加消息到记忆
        memory_manager.add_message("test_agent_001", message)
        print("✅ 消息添加到记忆成功")
        
        # 获取记忆统计
        stats = memory_manager.get_memory_stats()
        print(f"✅ 记忆统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 记忆管理测试失败: {e}")
        return False

def test_coordinator():
    """测试协调器"""
    print("\n🧪 测试协调器...")
    
    try:
        from coordinator import SequentialCoordinator
        from schema import Task, TaskPriority
        
        # 创建协调器
        coordinator = SequentialCoordinator(
            coordinator_id="test_coord_001",
            name="测试协调器"
        )
        print("✅ 协调器创建成功")
        
        # 创建测试任务
        task = Task(
            id="test_task_001",
            title="测试任务",
            description="这是一个测试任务",
            priority=TaskPriority.NORMAL
        )
        
        # 测试协调器状态
        status = coordinator.get_status()
        print(f"✅ 协调器状态: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ 协调器测试失败: {e}")
        return False

async def test_agent_system():
    """测试智能体系统"""
    print("\n🧪 测试智能体系统...")
    
    try:
        from agents import AgentFactory, AgentType
        from tools import CalculatorTool
        from langchain.llms.fake import FakeListLLM
        from schema import Task, TaskPriority
        
        # 创建模拟LLM
        llm = FakeListLLM(responses=[
            "我需要使用计算器工具来计算这个表达式",
            "计算结果：5",
            "任务完成"
        ])
        
        # 创建智能体工厂
        factory = AgentFactory()
        
        # 创建工具
        tools = [CalculatorTool()]
        
        # 创建ReAct智能体
        agent = factory.create_react_agent(
            agent_id="test_agent_001",
            name="测试智能体",
            description="用于测试的智能体",
            llm=llm,
            tools=tools,
            max_iterations=3
        )
        print("✅ 智能体创建成功")
        
        # 创建测试任务
        task = Task(
            id="test_task_001",
            title="计算数学表达式",
            description="计算 2 + 3",
            priority=TaskPriority.NORMAL
        )
        
        # 执行任务
        result = agent.act(task)
        print(f"✅ 任务执行完成: {result.status.value}")
        
        # 获取智能体状态
        status = agent.get_status()
        print(f"✅ 智能体状态: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ 智能体系统测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 多智能体系统完整测试")
    print("=" * 60)
    
    # 设置日志
    from utils import setup_logging
    setup_logging(level="INFO", use_colors=True)
    
    # 测试导入
    if not test_imports():
        print("\n❌ 导入测试失败，请检查依赖")
        return
    
    # 测试基本对象
    if not test_basic_objects():
        print("\n❌ 基本对象测试失败")
        return
    
    # 测试工具
    if not test_tools():
        print("\n❌ 工具测试失败")
        return
    
    # 测试路由器
    if not test_router():
        print("\n❌ 路由器测试失败")
        return
    
    # 测试记忆管理
    if not test_memory():
        print("\n❌ 记忆管理测试失败")
        return
    
    # 测试协调器
    if not test_coordinator():
        print("\n❌ 协调器测试失败")
        return
    
    # 测试智能体系统
    if not asyncio.run(test_agent_system()):
        print("\n❌ 智能体系统测试失败")
        return
    
    print("\n🎉 所有测试通过！多智能体系统功能完整。")
    print("\n📋 系统组件总结:")
    print("   ✅ Schema - 数据结构定义")
    print("   ✅ Utils - 工具函数和配置")
    print("   ✅ Tools - 工具集成")
    print("   ✅ Agents - 智能体实现")
    print("   ✅ Memory - 记忆管理")
    print("   ✅ Coordinator - 任务协调")
    print("   ✅ Router - 消息路由")
    print("\n🚀 系统已准备就绪，可以开始使用！")

if __name__ == "__main__":
    main()
