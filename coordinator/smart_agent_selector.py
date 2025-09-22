"""
智能智能体选择器
基于能力匹配、负载均衡和性能历史的智能选择算法
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from agents.base_agent import BaseAgent
from schema import Task, TaskPriority


class SelectionStrategy(Enum):
    """选择策略枚举"""
    CAPABILITY_MATCH = "capability_match"  # 能力匹配
    LOAD_BALANCE = "load_balance"  # 负载均衡
    PERFORMANCE_BASED = "performance_based"  # 基于性能
    HYBRID = "hybrid"  # 混合策略


@dataclass
class AgentCapability:
    """智能体能力描述"""
    agent_id: str
    capabilities: List[str]  # 能力列表
    specializations: List[str]  # 专业领域
    performance_score: float  # 性能评分 (0-1)
    success_rate: float  # 成功率 (0-1)
    avg_execution_time: float  # 平均执行时间
    current_load: float  # 当前负载 (0-1)
    last_activity: float  # 最后活动时间戳


@dataclass
class TaskRequirement:
    """任务需求描述"""
    task_id: str
    required_capabilities: List[str]  # 必需能力
    preferred_capabilities: List[str]  # 偏好能力
    priority: TaskPriority
    estimated_duration: float  # 预估执行时间
    complexity: float  # 复杂度 (0-1)


class BaseAgentSelector(ABC):
    """智能体选择器基类"""
    
    @abstractmethod
    def select_agent(
        self, 
        task: Task, 
        available_agents: List[BaseAgent],
        agent_capabilities: Dict[str, AgentCapability]
    ) -> Optional[BaseAgent]:
        """
        选择最适合的智能体
        
        Args:
            task: 任务对象
            available_agents: 可用智能体列表
            agent_capabilities: 智能体能力信息
            
        Returns:
            选中的智能体或None
        """
        pass


class CapabilityBasedSelector(BaseAgentSelector):
    """基于能力匹配的选择器"""
    
    def select_agent(
        self, 
        task: Task, 
        available_agents: List[BaseAgent],
        agent_capabilities: Dict[str, AgentCapability]
    ) -> Optional[BaseAgent]:
        """基于能力匹配选择智能体"""
        if not available_agents:
            return None
        
        # 分析任务需求
        task_req = self._analyze_task_requirements(task)
        
        # 计算每个智能体的匹配度
        best_agent = None
        best_score = -1
        
        for agent in available_agents:
            if agent.agent_id not in agent_capabilities:
                continue
                
            capability = agent_capabilities[agent.agent_id]
            score = self._calculate_capability_match_score(task_req, capability)
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _analyze_task_requirements(self, task: Task) -> TaskRequirement:
        """分析任务需求"""
        # 从任务描述中提取能力需求
        required_capabilities = []
        preferred_capabilities = []
        
        # 基于任务标题和描述进行简单的关键词匹配
        task_text = f"{task.title} {task.description}".lower()
        
        # 定义能力关键词映射
        capability_keywords = {
            "计算": ["math", "calculate", "compute", "arithmetic", "计算", "数学"],
            "搜索": ["search", "find", "lookup", "query", "搜索", "查找"],
            "编程": ["code", "program", "script", "python", "javascript", "编程", "代码"],
            "分析": ["analyze", "analysis", "process", "evaluate", "分析", "评估"],
            "翻译": ["translate", "translation", "language", "翻译", "语言"],
            "写作": ["write", "compose", "create", "generate", "写", "创作", "生成"],
            "总结": ["summarize", "summary", "abstract", "总结", "摘要"],
            "问答": ["question", "answer", "qa", "help", "问题", "回答", "帮助"],
            "计划": ["plan", "planning", "schedule", "itinerary", "计划", "规划", "行程", "安排", "攻略"],
            "组织": ["organize", "organization", "structure", "组织", "结构化", "整理"],
            "项目管理": ["project", "management", "coordinate", "项目", "管理", "协调"]
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in task_text for keyword in keywords):
                if capability in ["计算", "搜索", "编程", "计划", "组织", "项目管理"]:  # 核心能力
                    required_capabilities.append(capability)
                else:
                    preferred_capabilities.append(capability)
        
        # 如果没有检测到特定能力，使用通用能力
        if not required_capabilities:
            required_capabilities = ["通用"]
        
        return TaskRequirement(
            task_id=task.id,
            required_capabilities=required_capabilities,
            preferred_capabilities=preferred_capabilities,
            priority=task.priority,
            estimated_duration=self._estimate_task_duration(task),
            complexity=self._estimate_task_complexity(task)
        )
    
    def _calculate_capability_match_score(
        self, 
        task_req: TaskRequirement, 
        agent_cap: AgentCapability
    ) -> float:
        """计算能力匹配分数"""
        score = 0.0
        
        # 必需能力匹配 (权重: 0.6)
        required_match = 0.0
        for req_cap in task_req.required_capabilities:
            if req_cap in agent_cap.capabilities:
                required_match += 1.0
            elif req_cap in agent_cap.specializations:
                required_match += 0.8  # 专业领域匹配度稍低
        
        if task_req.required_capabilities:
            required_match /= len(task_req.required_capabilities)
        score += required_match * 0.6
        
        # 偏好能力匹配 (权重: 0.2)
        preferred_match = 0.0
        for pref_cap in task_req.preferred_capabilities:
            if pref_cap in agent_cap.capabilities:
                preferred_match += 1.0
            elif pref_cap in agent_cap.specializations:
                preferred_match += 0.8
        
        if task_req.preferred_capabilities:
            preferred_match /= len(task_req.preferred_capabilities)
        score += preferred_match * 0.2
        
        # 性能因素 (权重: 0.2)
        performance_factor = (
            agent_cap.performance_score * 0.5 +
            agent_cap.success_rate * 0.3 +
            (1.0 - min(agent_cap.current_load, 1.0)) * 0.2
        )
        score += performance_factor * 0.2
        
        return score
    
    def _estimate_task_duration(self, task: Task) -> float:
        """估算任务执行时间"""
        # 基于任务复杂度进行简单估算
        complexity = self._estimate_task_complexity(task)
        base_time = 10.0  # 基础时间（秒）
        return base_time * (1 + complexity * 2)
    
    def _estimate_task_complexity(self, task: Task) -> float:
        """估算任务复杂度"""
        # 基于任务描述长度和关键词进行简单估算
        text_length = len(task.description)
        complexity = min(text_length / 200.0, 1.0)  # 归一化到0-1
        
        # 复杂关键词增加复杂度
        complex_keywords = ["复杂", "multiple", "advanced", "detailed", "comprehensive"]
        if any(keyword in task.description.lower() for keyword in complex_keywords):
            complexity = min(complexity + 0.3, 1.0)
        
        return complexity


class LoadBalancedSelector(BaseAgentSelector):
    """负载均衡选择器"""
    
    def select_agent(
        self, 
        task: Task, 
        available_agents: List[BaseAgent],
        agent_capabilities: Dict[str, AgentCapability]
    ) -> Optional[BaseAgent]:
        """基于负载均衡选择智能体"""
        if not available_agents:
            return None
        
        # 选择负载最低的智能体
        best_agent = None
        lowest_load = float('inf')
        
        for agent in available_agents:
            if agent.agent_id not in agent_capabilities:
                continue
            
            capability = agent_capabilities[agent.agent_id]
            current_load = capability.current_load
            
            # 考虑历史执行时间
            if capability.avg_execution_time > 0:
                time_factor = min(capability.avg_execution_time / 60.0, 1.0)
                current_load += time_factor * 0.2
            
            if current_load < lowest_load:
                lowest_load = current_load
                best_agent = agent
        
        return best_agent


class PerformanceBasedSelector(BaseAgentSelector):
    """基于性能的选择器"""
    
    def select_agent(
        self, 
        task: Task, 
        available_agents: List[BaseAgent],
        agent_capabilities: Dict[str, AgentCapability]
    ) -> Optional[BaseAgent]:
        """基于性能选择智能体"""
        if not available_agents:
            return None
        
        # 计算综合性能分数
        best_agent = None
        best_performance = -1
        
        for agent in available_agents:
            if agent.agent_id not in agent_capabilities:
                continue
            
            capability = agent_capabilities[agent.agent_id]
            
            # 综合性能分数
            performance_score = (
                capability.performance_score * 0.4 +
                capability.success_rate * 0.4 +
                (1.0 - min(capability.current_load, 1.0)) * 0.2
            )
            
            # 根据任务优先级调整
            if task.priority == TaskPriority.URGENT:
                # 紧急任务优先选择成功率高的
                performance_score = capability.success_rate
            elif task.priority == TaskPriority.HIGH:
                # 高优先级任务平衡成功率和性能
                performance_score = (capability.success_rate + capability.performance_score) / 2
            
            if performance_score > best_performance:
                best_performance = performance_score
                best_agent = agent
        
        return best_agent


class HybridSelector(BaseAgentSelector):
    """混合策略选择器"""
    
    def __init__(self, strategy_weights: Dict[SelectionStrategy, float] = None):
        """
        初始化混合选择器
        
        Args:
            strategy_weights: 策略权重字典
        """
        self.strategy_weights = strategy_weights or {
            SelectionStrategy.CAPABILITY_MATCH: 0.5,
            SelectionStrategy.LOAD_BALANCE: 0.3,
            SelectionStrategy.PERFORMANCE_BASED: 0.2
        }
        
        self.capability_selector = CapabilityBasedSelector()
        self.load_selector = LoadBalancedSelector()
        self.performance_selector = PerformanceBasedSelector()
    
    def select_agent(
        self, 
        task: Task, 
        available_agents: List[BaseAgent],
        agent_capabilities: Dict[str, AgentCapability]
    ) -> Optional[BaseAgent]:
        """使用混合策略选择智能体"""
        if not available_agents:
            return None
        
        # 获取各策略的推荐
        capability_agent = self.capability_selector.select_agent(task, available_agents, agent_capabilities)
        load_agent = self.load_selector.select_agent(task, available_agents, agent_capabilities)
        performance_agent = self.performance_selector.select_agent(task, available_agents, agent_capabilities)
        
        # 计算加权分数
        agent_scores = {}
        
        if capability_agent:
            agent_scores[capability_agent.agent_id] = self.strategy_weights[SelectionStrategy.CAPABILITY_MATCH]
        
        if load_agent:
            score = agent_scores.get(load_agent.agent_id, 0)
            agent_scores[load_agent.agent_id] = score + self.strategy_weights[SelectionStrategy.LOAD_BALANCE]
        
        if performance_agent:
            score = agent_scores.get(performance_agent.agent_id, 0)
            agent_scores[performance_agent.agent_id] = score + self.strategy_weights[SelectionStrategy.PERFORMANCE_BASED]
        
        # 选择分数最高的智能体
        if agent_scores:
            best_agent_id = max(agent_scores, key=agent_scores.get)
            for agent in available_agents:
                if agent.agent_id == best_agent_id:
                    return agent
        
        # 如果没有明确的推荐，返回第一个可用智能体
        return available_agents[0] if available_agents else None


class SmartAgentSelector:
    """智能智能体选择器主类"""
    
    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.HYBRID):
        """
        初始化智能选择器
        
        Args:
            strategy: 选择策略
        """
        self.strategy = strategy
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.selection_history: List[Dict[str, Any]] = []
        
        # 创建选择器实例
        if strategy == SelectionStrategy.CAPABILITY_MATCH:
            self.selector = CapabilityBasedSelector()
        elif strategy == SelectionStrategy.LOAD_BALANCE:
            self.selector = LoadBalancedSelector()
        elif strategy == SelectionStrategy.PERFORMANCE_BASED:
            self.selector = PerformanceBasedSelector()
        else:
            self.selector = HybridSelector()
    
    def register_agent_capability(self, capability: AgentCapability) -> None:
        """注册智能体能力信息"""
        self.agent_capabilities[capability.agent_id] = capability
    
    def update_agent_performance(
        self, 
        agent_id: str, 
        success: bool, 
        execution_time: float
    ) -> None:
        """更新智能体性能信息"""
        if agent_id not in self.agent_capabilities:
            return
        
        capability = self.agent_capabilities[agent_id]
        
        # 更新成功率（使用指数移动平均）
        alpha = 0.1
        if success:
            capability.success_rate = capability.success_rate * (1 - alpha) + alpha
        else:
            capability.success_rate = capability.success_rate * (1 - alpha)
        
        # 更新平均执行时间（使用指数移动平均）
        capability.avg_execution_time = (
            capability.avg_execution_time * (1 - alpha) + 
            execution_time * alpha
        )
        
        # 更新最后活动时间
        capability.last_activity = time.time()
    
    def update_agent_load(self, agent_id: str, load: float) -> None:
        """更新智能体负载"""
        if agent_id in self.agent_capabilities:
            self.agent_capabilities[agent_id].current_load = min(load, 1.0)
    
    def select_agent(
        self, 
        task: Task, 
        available_agents: List[BaseAgent]
    ) -> Optional[BaseAgent]:
        """选择最适合的智能体"""
        start_time = time.time()
        
        # 过滤掉没有能力信息的智能体
        valid_agents = [
            agent for agent in available_agents 
            if agent.agent_id in self.agent_capabilities
        ]
        
        if not valid_agents:
            return None
        
        # 执行选择
        selected_agent = self.selector.select_agent(task, valid_agents, self.agent_capabilities)
        
        # 记录选择历史
        selection_time = time.time() - start_time
        self.selection_history.append({
            "timestamp": time.time(),
            "task_id": task.id,
            "selected_agent": selected_agent.agent_id if selected_agent else None,
            "available_agents": [agent.agent_id for agent in valid_agents],
            "selection_time": selection_time,
            "strategy": self.strategy.value
        })
        
        # 保持历史记录在合理范围内
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-500:]
        
        return selected_agent
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """获取选择统计信息"""
        if not self.selection_history:
            return {"total_selections": 0}
        
        recent_selections = self.selection_history[-100:]  # 最近100次选择
        
        # 统计各智能体被选中的次数
        agent_selection_count = {}
        total_selections = len(recent_selections)
        
        for selection in recent_selections:
            agent_id = selection["selected_agent"]
            if agent_id:
                agent_selection_count[agent_id] = agent_selection_count.get(agent_id, 0) + 1
        
        # 计算平均选择时间
        avg_selection_time = sum(s["selection_time"] for s in recent_selections) / total_selections
        
        return {
            "total_selections": total_selections,
            "agent_selection_distribution": agent_selection_count,
            "average_selection_time": avg_selection_time,
            "strategy": self.strategy.value,
            "registered_agents": len(self.agent_capabilities)
        }
    
    def set_strategy(self, strategy: SelectionStrategy) -> None:
        """更改选择策略"""
        self.strategy = strategy
        
        # 重新创建选择器
        if strategy == SelectionStrategy.CAPABILITY_MATCH:
            self.selector = CapabilityBasedSelector()
        elif strategy == SelectionStrategy.LOAD_BALANCE:
            self.selector = LoadBalancedSelector()
        elif strategy == SelectionStrategy.PERFORMANCE_BASED:
            self.selector = PerformanceBasedSelector()
        else:
            self.selector = HybridSelector()
    
    def __str__(self) -> str:
        return f"智能选择器(策略: {self.strategy.value}, 注册智能体: {len(self.agent_capabilities)})"
