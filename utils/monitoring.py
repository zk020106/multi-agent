"""
监控和可观测性模块
提供性能指标收集、健康检查和实时监控功能
"""

import asyncio
import json
import time
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: psutil未安装，系统指标收集功能将被禁用")
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
import weakref


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"  # 仪表盘
    HISTOGRAM = "histogram"  # 直方图
    TIMER = "timer"  # 计时器


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """指标数据"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """健康检查"""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    response_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """指标收集器"""
    
    def __init__(self, max_history_size: int = 1000):
        """
        初始化指标收集器
        
        Args:
            max_history_size: 最大历史记录大小
        """
        self.max_history_size = max_history_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None) -> None:
        """增加计数器"""
        with self._lock:
            self.counters[name] += value
            self._record_metric(name, self.counters[name], MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """设置仪表盘值"""
        with self._lock:
            self.gauges[name] = value
            self._record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """记录直方图值"""
        with self._lock:
            self.histograms[name].append(value)
            # 保持直方图大小在合理范围内
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-500:]
            self._record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None) -> None:
        """记录计时器值"""
        with self._lock:
            self.timers[name].append(duration)
            # 保持计时器大小在合理范围内
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-500:]
            self._record_metric(name, duration, MetricType.TIMER, tags)
    
    def _record_metric(self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str] = None) -> None:
        """记录指标"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            tags=tags or {}
        )
        self.metrics[name].append(metric)
    
    def get_metric_value(self, name: str, metric_type: MetricType) -> Optional[float]:
        """获取指标值"""
        with self._lock:
            if metric_type == MetricType.COUNTER:
                return self.counters.get(name)
            elif metric_type == MetricType.GAUGE:
                return self.gauges.get(name)
            elif metric_type == MetricType.HISTOGRAM:
                values = self.histograms.get(name, [])
                return sum(values) / len(values) if values else None
            elif metric_type == MetricType.TIMER:
                values = self.timers.get(name, [])
                return sum(values) / len(values) if values else None
            return None
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[Metric]:
        """获取指标历史"""
        with self._lock:
            history = list(self.metrics.get(name, []))
            return history[-limit:] if limit else history
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                    "p95": self._percentile(values, 95) if values else 0,
                    "p99": self._percentile(values, 99) if values else 0
                } for name, values in self.histograms.items()},
                "timers": {name: {
                    "count": len(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                    "p95": self._percentile(values, 95) if values else 0,
                    "p99": self._percentile(values, 99) if values else 0
                } for name, values in self.timers.items()}
            }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def clear_metrics(self) -> None:
        """清除所有指标"""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()


class SystemMetricsCollector:
    """系统指标收集器"""
    
    def __init__(self, metric_collector: MetricCollector):
        """
        初始化系统指标收集器
        
        Args:
            metric_collector: 指标收集器实例
        """
        self.metric_collector = metric_collector
        self._running = False
        self._thread = None
    
    def start(self, interval: float = 10.0) -> None:
        """
        开始收集系统指标
        
        Args:
            interval: 收集间隔（秒）
        """
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, args=(interval,), daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止收集系统指标"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def _collect_loop(self, interval: float) -> None:
        """收集循环"""
        while self._running:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                print(f"系统指标收集错误: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> None:
        """收集系统指标"""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metric_collector.set_gauge("system.cpu.usage", cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self.metric_collector.set_gauge("system.memory.usage", memory.percent)
            self.metric_collector.set_gauge("system.memory.available", memory.available)
            self.metric_collector.set_gauge("system.memory.total", memory.total)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            self.metric_collector.set_gauge("system.disk.usage", disk.percent)
            self.metric_collector.set_gauge("system.disk.free", disk.free)
            self.metric_collector.set_gauge("system.disk.total", disk.total)
            
            # 网络IO
            net_io = psutil.net_io_counters()
            self.metric_collector.set_gauge("system.network.bytes_sent", net_io.bytes_sent)
            self.metric_collector.set_gauge("system.network.bytes_recv", net_io.bytes_recv)
            
            # 进程信息
            process = psutil.Process()
            self.metric_collector.set_gauge("system.process.cpu_percent", process.cpu_percent())
            self.metric_collector.set_gauge("system.process.memory_percent", process.memory_percent())
            self.metric_collector.set_gauge("system.process.memory_rss", process.memory_info().rss)
            self.metric_collector.set_gauge("system.process.memory_vms", process.memory_info().vms)
            
        except Exception as e:
            print(f"收集系统指标时出错: {e}")


class HealthChecker(ABC):
    """健康检查器基类"""
    
    @abstractmethod
    async def check_health(self) -> HealthCheck:
        """执行健康检查"""
        pass


class SystemHealthChecker(HealthChecker):
    """系统健康检查器"""
    
    def __init__(self, metric_collector: MetricCollector):
        """
        初始化系统健康检查器
        
        Args:
            metric_collector: 指标收集器实例
        """
        self.metric_collector = metric_collector
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0
        }
    
    async def check_health(self) -> HealthCheck:
        """检查系统健康状态"""
        start_time = time.time()
        
        try:
            # 获取系统指标
            cpu_usage = self.metric_collector.get_metric_value("system.cpu.usage", MetricType.GAUGE) or 0
            memory_usage = self.metric_collector.get_metric_value("system.memory.usage", MetricType.GAUGE) or 0
            disk_usage = self.metric_collector.get_metric_value("system.disk.usage", MetricType.GAUGE) or 0
            
            # 判断健康状态
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_usage > self.thresholds["cpu_usage"]:
                status = HealthStatus.WARNING
                issues.append(f"CPU使用率过高: {cpu_usage:.1f}%")
            
            if memory_usage > self.thresholds["memory_usage"]:
                status = HealthStatus.WARNING
                issues.append(f"内存使用率过高: {memory_usage:.1f}%")
            
            if disk_usage > self.thresholds["disk_usage"]:
                status = HealthStatus.CRITICAL
                issues.append(f"磁盘使用率过高: {disk_usage:.1f}%")
            
            # 如果有多个警告，升级为严重
            if len(issues) > 1 and status == HealthStatus.WARNING:
                status = HealthStatus.CRITICAL
            
            message = "系统运行正常" if status == HealthStatus.HEALTHY else "; ".join(issues)
            
            response_time = time.time() - start_time
            
            return HealthCheck(
                name="system_health",
                status=status,
                message=message,
                timestamp=time.time(),
                response_time=response_time,
                details={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage,
                    "thresholds": self.thresholds
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_health",
                status=HealthStatus.CRITICAL,
                message=f"健康检查失败: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time
            )


class AgentHealthChecker(HealthChecker):
    """智能体健康检查器"""
    
    def __init__(self, agents: Dict[str, Any]):
        """
        初始化智能体健康检查器
        
        Args:
            agents: 智能体字典
        """
        self.agents = agents
    
    async def check_health(self) -> HealthCheck:
        """检查智能体健康状态"""
        start_time = time.time()
        
        try:
            healthy_agents = 0
            total_agents = len(self.agents)
            agent_details = {}
            
            for agent_id, agent in self.agents.items():
                try:
                    # 检查智能体是否响应
                    if hasattr(agent, 'get_status'):
                        status = agent.get_status()
                        is_healthy = status.get('is_busy', False) is not False
                    else:
                        is_healthy = True  # 假设没有状态方法的智能体是健康的
                    
                    if is_healthy:
                        healthy_agents += 1
                    
                    agent_details[agent_id] = {
                        "healthy": is_healthy,
                        "name": getattr(agent, 'name', 'Unknown'),
                        "type": type(agent).__name__
                    }
                    
                except Exception as e:
                    agent_details[agent_id] = {
                        "healthy": False,
                        "error": str(e),
                        "name": getattr(agent, 'name', 'Unknown'),
                        "type": type(agent).__name__
                    }
            
            # 判断整体健康状态
            health_ratio = healthy_agents / total_agents if total_agents > 0 else 1.0
            
            if health_ratio >= 0.9:
                status = HealthStatus.HEALTHY
                message = f"所有智能体运行正常 ({healthy_agents}/{total_agents})"
            elif health_ratio >= 0.7:
                status = HealthStatus.WARNING
                message = f"部分智能体异常 ({healthy_agents}/{total_agents})"
            else:
                status = HealthStatus.CRITICAL
                message = f"多数智能体异常 ({healthy_agents}/{total_agents})"
            
            response_time = time.time() - start_time
            
            return HealthCheck(
                name="agent_health",
                status=status,
                message=message,
                timestamp=time.time(),
                response_time=response_time,
                details={
                    "total_agents": total_agents,
                    "healthy_agents": healthy_agents,
                    "health_ratio": health_ratio,
                    "agent_details": agent_details
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="agent_health",
                status=HealthStatus.CRITICAL,
                message=f"智能体健康检查失败: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time
            )


class MonitoringDashboard:
    """监控仪表板"""
    
    def __init__(self, metric_collector: MetricCollector):
        """
        初始化监控仪表板
        
        Args:
            metric_collector: 指标收集器实例
        """
        self.metric_collector = metric_collector
        self.health_checkers: List[HealthChecker] = []
        self._running = False
        self._dashboard_thread = None
    
    def add_health_checker(self, checker: HealthChecker) -> None:
        """添加健康检查器"""
        self.health_checkers.append(checker)
    
    def start_dashboard(self, port: int = 8080, interval: float = 30.0) -> None:
        """
        启动监控仪表板
        
        Args:
            port: 仪表板端口
            interval: 健康检查间隔
        """
        if self._running:
            return
        
        self._running = True
        self._dashboard_thread = threading.Thread(
            target=self._dashboard_loop, 
            args=(port, interval), 
            daemon=True
        )
        self._dashboard_thread.start()
        print(f"监控仪表板已启动，访问 http://localhost:{port}")
    
    def stop_dashboard(self) -> None:
        """停止监控仪表板"""
        self._running = False
        if self._dashboard_thread:
            self._dashboard_thread.join(timeout=5.0)
    
    def _dashboard_loop(self, port: int, interval: float) -> None:
        """仪表板循环"""
        import http.server
        import socketserver
        
        class MonitoringHandler(http.server.BaseHTTPRequestHandler):
            def __init__(self, dashboard, *args, **kwargs):
                self.dashboard = dashboard
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/':
                    self._serve_dashboard()
                elif self.path == '/metrics':
                    self._serve_metrics()
                elif self.path == '/health':
                    self._serve_health()
                else:
                    self.send_error(404)
            
            def _serve_dashboard(self):
                html = self.dashboard._generate_dashboard_html()
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            
            def _serve_metrics(self):
                metrics = self.dashboard.metric_collector.get_all_metrics()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(metrics, indent=2).encode())
            
            def _serve_health(self):
                health_data = asyncio.run(self.dashboard._get_health_data())
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(health_data, indent=2).encode())
        
        def handler_factory(*args, **kwargs):
            return MonitoringHandler(self, *args, **kwargs)
        
        try:
            with socketserver.TCPServer(("", port), handler_factory) as httpd:
                httpd.serve_forever()
        except Exception as e:
            print(f"监控仪表板启动失败: {e}")
    
    def _generate_dashboard_html(self) -> str:
        """生成仪表板HTML"""
        metrics = self.metric_collector.get_all_metrics()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>多智能体系统监控仪表板</title>
            <meta charset="utf-8">
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .healthy {{ background-color: #d4edda; }}
                .warning {{ background-color: #fff3cd; }}
                .critical {{ background-color: #f8d7da; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            </style>
        </head>
        <body>
            <h1>多智能体系统监控仪表板</h1>
            <p>最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric-grid">
                <div class="metric">
                    <h3>系统指标</h3>
                    <p>CPU使用率: {metrics.get('gauges', {}).get('system.cpu.usage', 0):.1f}%</p>
                    <p>内存使用率: {metrics.get('gauges', {}).get('system.memory.usage', 0):.1f}%</p>
                    <p>磁盘使用率: {metrics.get('gauges', {}).get('system.disk.usage', 0):.1f}%</p>
                </div>
                
                <div class="metric">
                    <h3>应用指标</h3>
                    <p>任务完成数: {metrics.get('counters', {}).get('tasks.completed', 0)}</p>
                    <p>任务失败数: {metrics.get('counters', {}).get('tasks.failed', 0)}</p>
                    <p>平均响应时间: {metrics.get('timers', {}).get('task.execution_time', {}).get('avg', 0):.2f}s</p>
                </div>
            </div>
            
            <div class="metric">
                <h3>健康状态</h3>
                <p><a href="/health">查看详细健康状态</a></p>
            </div>
            
            <div class="metric">
                <h3>原始指标数据</h3>
                <p><a href="/metrics">查看JSON格式指标</a></p>
            </div>
        </body>
        </html>
        """
        return html
    
    async def _get_health_data(self) -> Dict[str, Any]:
        """获取健康数据"""
        health_checks = []
        
        for checker in self.health_checkers:
            try:
                health_check = await checker.check_health()
                health_checks.append({
                    "name": health_check.name,
                    "status": health_check.status.value,
                    "message": health_check.message,
                    "timestamp": health_check.timestamp,
                    "response_time": health_check.response_time,
                    "details": health_check.details
                })
            except Exception as e:
                health_checks.append({
                    "name": checker.__class__.__name__,
                    "status": "critical",
                    "message": f"健康检查失败: {str(e)}",
                    "timestamp": time.time(),
                    "response_time": 0.0
                })
        
        return {
            "overall_status": self._calculate_overall_status(health_checks),
            "health_checks": health_checks,
            "timestamp": time.time()
        }
    
    def _calculate_overall_status(self, health_checks: List[Dict[str, Any]]) -> str:
        """计算整体健康状态"""
        if not health_checks:
            return "unknown"
        
        statuses = [check["status"] for check in health_checks]
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"


class MonitoringManager:
    """监控管理器主类"""
    
    def __init__(self, enable_system_metrics: bool = True):
        """
        初始化监控管理器
        
        Args:
            enable_system_metrics: 是否启用系统指标收集
        """
        self.metric_collector = MetricCollector()
        self.dashboard = MonitoringDashboard(self.metric_collector)
        self.system_metrics_collector = None
        
        if enable_system_metrics:
            self.system_metrics_collector = SystemMetricsCollector(self.metric_collector)
            self.system_metrics_collector.start()
        
        # 添加默认健康检查器
        self.dashboard.add_health_checker(SystemHealthChecker(self.metric_collector))
    
    def add_agent_health_checker(self, agents: Dict[str, Any]) -> None:
        """添加智能体健康检查器"""
        self.dashboard.add_health_checker(AgentHealthChecker(agents))
    
    def start_monitoring(self, dashboard_port: int = 8080) -> None:
        """启动监控"""
        self.dashboard.start_dashboard(port=dashboard_port)
        print(f"监控系统已启动，仪表板: http://localhost:{dashboard_port}")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        if self.system_metrics_collector:
            self.system_metrics_collector.stop()
        self.dashboard.stop_dashboard()
    
    def record_task_metric(self, task_id: str, duration: float, success: bool) -> None:
        """记录任务指标"""
        self.metric_collector.record_timer("task.execution_time", duration)
        if success:
            self.metric_collector.increment_counter("tasks.completed")
        else:
            self.metric_collector.increment_counter("tasks.failed")
    
    def record_agent_metric(self, agent_id: str, metric_name: str, value: float) -> None:
        """记录智能体指标"""
        self.metric_collector.set_gauge(f"agent.{agent_id}.{metric_name}", value)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        metrics = self.metric_collector.get_all_metrics()
        
        return {
            "system_health": {
                "cpu_usage": metrics.get('gauges', {}).get('system.cpu.usage', 0),
                "memory_usage": metrics.get('gauges', {}).get('system.memory.usage', 0),
                "disk_usage": metrics.get('gauges', {}).get('system.disk.usage', 0)
            },
            "application_metrics": {
                "tasks_completed": metrics.get('counters', {}).get('tasks.completed', 0),
                "tasks_failed": metrics.get('counters', {}).get('tasks.failed', 0),
                "avg_task_duration": metrics.get('timers', {}).get('task.execution_time', {}).get('avg', 0)
            },
            "timestamp": time.time()
        }


# 全局监控管理器实例
_global_monitoring_manager: Optional[MonitoringManager] = None


def get_monitoring_manager() -> MonitoringManager:
    """获取全局监控管理器"""
    global _global_monitoring_manager
    if _global_monitoring_manager is None:
        _global_monitoring_manager = MonitoringManager()
    return _global_monitoring_manager


def set_monitoring_manager(manager: MonitoringManager) -> None:
    """设置全局监控管理器"""
    global _global_monitoring_manager
    _global_monitoring_manager = manager


# 便捷装饰器
def monitor_task_execution(func: Callable) -> Callable:
    """监控任务执行的装饰器"""
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            get_monitoring_manager().record_task_metric("unknown", duration, True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            get_monitoring_manager().record_task_metric("unknown", duration, False)
            raise e
    
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            get_monitoring_manager().record_task_metric("unknown", duration, True)
            return result
        except Exception as e:
            duration = time.time() - start_time
            get_monitoring_manager().record_task_metric("unknown", duration, False)
            raise e
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
