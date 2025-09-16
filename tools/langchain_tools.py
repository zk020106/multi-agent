"""
基于LangChain的工具实现
提供各种智能体可用的工具
"""

import json
import os
import sqlite3
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Type, Tuple, Literal
import sys
from typing import Any, Dict, List, Optional, Type, Tuple
import os
import time
from utils.config import get_config

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class APIToolInput(BaseModel):
    """API工具输入模型"""
    url: str = Field(description="API端点URL")
    method: str = Field(default="GET", description="HTTP方法 (GET, POST, PUT, DELETE)")
    headers: Optional[Dict[str, str]] = Field(default=None, description="HTTP请求头")
    params: Optional[Dict[str, Any]] = Field(default=None, description="URL参数")
    data: Optional[Dict[str, Any]] = Field(default=None, description="请求体数据")
    timeout: Optional[int] = Field(default=30, description="请求超时时间（秒）")


class APITool(BaseTool):
    """HTTP API调用工具"""
    
    name: str = "api_tool"
    description: str = "调用HTTP API接口，支持GET、POST、PUT、DELETE等方法"
    args_schema: Type[BaseModel] = APIToolInput
    
    def _run(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = 30,
        **kwargs
    ) -> str:
        """
        执行API调用
        
        Args:
            url: API端点URL
            method: HTTP方法
            headers: 请求头
            params: URL参数
            data: 请求体数据
            timeout: 超时时间
            
        Returns:
            API响应结果
        """
        try:
            # 准备请求参数
            request_kwargs = {
                "timeout": timeout,
                "headers": headers or {}
            }
            
            if params:
                request_kwargs["params"] = params
            
            if data and method.upper() in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = data
            
            # 发送请求
            response = requests.request(method.upper(), url, **request_kwargs)
            
            # 处理响应
            try:
                result = response.json()
            except json.JSONDecodeError:
                result = response.text
            
            return json.dumps({
                "status_code": response.status_code,
                "data": result,
                "headers": dict(response.headers)
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return f"API调用失败: {str(e)}"


class DatabaseToolInput(BaseModel):
    """数据库工具输入模型"""
    query: str = Field(description="SQL查询语句")
    database_path: Optional[str] = Field(default=":memory:", description="数据库文件路径")


class DatabaseTool(BaseTool):
    """SQLite数据库操作工具"""
    
    name: str = "database_tool"
    description: str = "执行SQLite数据库查询操作，支持SELECT、INSERT、UPDATE、DELETE等SQL语句"
    args_schema: Type[BaseModel] = DatabaseToolInput
    
    def _run(
        self,
        query: str,
        database_path: Optional[str] = ":memory:",
        **kwargs
    ) -> str:
        """
        执行数据库查询
        
        Args:
            query: SQL查询语句
            database_path: 数据库文件路径
            
        Returns:
            查询结果
        """
        try:
            # 连接数据库
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            
            # 执行查询
            cursor.execute(query)
            
            # 获取结果
            if query.strip().upper().startswith(('SELECT', 'PRAGMA')):
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description] if cursor.description else []
                
                # 格式化结果
                if results:
                    formatted_results = []
                    for row in results:
                        formatted_results.append(dict(zip(columns, row)))
                    return json.dumps(formatted_results, ensure_ascii=False, indent=2)
                else:
                    return "查询结果为空"
            else:
                # 对于非查询语句，提交事务
                conn.commit()
                return f"操作成功，影响行数: {cursor.rowcount}"
            
        except Exception as e:
            return f"数据库操作失败: {str(e)}"
        finally:
            if 'conn' in locals():
                conn.close()


class DocumentSearchToolInput(BaseModel):
    """文档搜索工具输入模型"""
    query: str = Field(description="搜索查询")
    document_path: Optional[str] = Field(default=None, description="文档文件路径（可选，不提供则跳过文档搜索）")
    max_results: Optional[int] = Field(default=5, description="最大结果数量")


class DocumentSearchTool(BaseTool):
    """文档搜索工具"""
    
    name: str = "document_search_tool"
    description: str = "在文档中搜索相关内容，支持文本文件搜索"
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    def _run(
        self,
        query: str,
        document_path: Optional[str] = None,
        max_results: Optional[int] = 5,
        **kwargs
    ) -> str:
        """
        在文档中搜索内容
        
        Args:
            query: 搜索查询
            document_path: 文档路径
            max_results: 最大结果数量
            
        Returns:
            搜索结果
        """
        try:
            # 若未提供文档路径，优雅跳过，避免触发参数校验错误
            if not document_path:
                return "未提供文档路径，跳过文档搜索。"
            if not os.path.exists(document_path):
                return f"文档文件不存在: {document_path}"
            
            # 读取文档内容
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的文本搜索
            lines = content.split('\n')
            results = []
            
            for i, line in enumerate(lines):
                if query.lower() in line.lower():
                    results.append({
                        "line_number": i + 1,
                        "content": line.strip(),
                        "context": lines[max(0, i-1):i+2] if len(lines) > 1 else [line]
                    })
                    
                    if len(results) >= max_results:
                        break
            
            if results:
                return json.dumps(results, ensure_ascii=False, indent=2)
            else:
                return f"未找到包含 '{query}' 的内容"
                
        except Exception as e:
            return f"文档搜索失败: {str(e)}"


class CodeExecutionToolInput(BaseModel):
    """代码执行工具输入模型"""
    code: str = Field(description="要执行的Python代码")
    language: Literal["python"] = Field(default="python", description="编程语言")
    sandbox: Literal["restricted", "none"] = Field(default="restricted", description="执行沙箱模式：restricted 使用 RestrictedPython，none 使用子进程")


class CodeExecutionTool(BaseTool):
    """代码执行工具"""
    
    name: str = "code_execution_tool"
    description: str = "执行Python代码并返回结果，支持基本的Python代码执行"
    args_schema: Type[BaseModel] = CodeExecutionToolInput
    
    def _run(
        self,
        code: str,
        language: str = "python",
        sandbox: str = "restricted",
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行代码（仅支持 Python）。默认使用 RestrictedPython 沙箱。
        sandbox:
          - restricted: 使用 RestrictedPython 安全执行（推荐）
          - none: 使用子进程执行（不隔离，仅限受信任环境）
        """
        if language.lower() != "python":
            return {"success": False, "error": f"暂不支持 {language} 语言，仅支持Python"}

        if sandbox == "restricted":
            try:
                from RestrictedPython import compile_restricted
                from RestrictedPython import safe_globals, limited_builtins
            except Exception as e:
                return {"success": False, "error": f"RestrictedPython 未安装或不可用: {e}. 可设置 sandbox='none' 退回子进程执行。"}

            try:
                byte_code = compile_restricted(code, filename='<restricted>', mode='exec')
                exec_globals: Dict[str, Any] = dict(safe_globals)
                # 提供极少量安全内建
                exec_globals['__builtins__'] = limited_builtins
                exec_locals: Dict[str, Any] = {}

                # 捕获输出
                import io
                import contextlib
                stdout_capture = io.StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    exec(byte_code, exec_globals, exec_locals)
                return {
                    "success": True,
                    "return_code": 0,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": "",
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        # 非沙箱模式：子进程执行（不安全，仅限受信任环境）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "代码执行超时（30秒）"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass




class SerperSearchToolInput(BaseModel):
    """Serper 联网搜索工具输入模型"""
    query: str = Field(description="搜索查询关键词")
    gl: Optional[str] = Field(default="us", description="地域，如 cn/us")
    hl: Optional[str] = Field(default="en", description="语言，如 zh-cn/en")
    num: Optional[int] = Field(default=10, description="返回结果数量")
    api_key: Optional[str] = Field(default=None, description="Serper API Key（可选，默认读环境变量 SERPER_API_KEY）")


class SerperSearchTool(BaseTool):
    """基于 Serper 的联网搜索工具"""

    name: str = "serper_search"
    description: str = "使用 Serper（google.serper.dev）进行网络搜索"
    args_schema: Type[BaseModel] = SerperSearchToolInput

    def _run(
        self,
        query: str,
        gl: Optional[str] = "us",
        hl: Optional[str] = "en",
        num: Optional[int] = 10,
        api_key: Optional[str] = None,
        **kwargs
    ) -> str:
        try:
            # 规范化查询，去除多余引号/空白
            norm_query = (query or "").strip().strip('"').strip("'")
            if not norm_query:
                return "Serper 搜索失败: query 不能为空"

            if api_key:
                key = api_key
            else:
                # 优先从全局配置读取，其次环境变量
                try:
                    cfg = get_config()
                    key = (cfg.serper_api_key if hasattr(cfg, 'serper_api_key') else None) or os.getenv("SERPER_API_KEY")
                    cooldown = getattr(cfg, 'serper_cooldown_seconds', 5) if hasattr(cfg, 'serper_cooldown_seconds') else 5
                except Exception:
                    key = os.getenv("SERPER_API_KEY")
                    cooldown = 5
            if not key:
                return "Serper API Key 未设置。请传入 api_key 或设置环境变量 SERPER_API_KEY。"

            # 简单去重/冷却窗口：短时间内重复请求直接返回缓存
            cache_key = f"{norm_query}|{gl}|{hl}|{num}"
            cached = _SERPER_CACHE.get(cache_key)
            if cached and (time.time() - cached[0]) < cooldown:
                return json.dumps({
                    "status_code": 200,
                    "data": cached[1],
                    "cached": True
                }, ensure_ascii=False, indent=2)

            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": key,
                "Content-Type": "application/json",
            }
            payload = {
                "q": norm_query,
                "gl": gl,
                "hl": hl,
                "num": num,
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            try:
                data = resp.json()
            except json.JSONDecodeError:
                data = resp.text

            # 写入缓存
            try:
                _SERPER_CACHE[cache_key] = (time.time(), data)
            except Exception:
                pass

            return json.dumps({
                "status_code": resp.status_code,
                "data": data
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return f"Serper 搜索失败: {str(e)}"


class CalculatorToolInput(BaseModel):
    """计算器工具输入模型"""
    expression: str = Field(description="数学表达式")


class CalculatorTool(BaseTool):
    """计算器工具"""
    
    name: str = "calculator_tool"
    description: str = "执行数学计算，支持基本的数学运算"
    args_schema: Type[BaseModel] = CalculatorToolInput
    
    def _run(
        self,
        expression: str,
        **kwargs
    ) -> str:
        """
        执行数学计算
        
        Args:
            expression: 数学表达式
            
        Returns:
            计算结果
        """
        try:
            # 安全的数学表达式计算
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in expression):
                return "表达式包含不安全的字符"
            
            # 执行计算
            result = eval(expression)
            return f"{expression} = {result}"
            
        except ZeroDivisionError:
            return "错误: 除零错误"
        except SyntaxError:
            return "错误: 表达式语法错误"
        except Exception as e:
            return f"计算失败: {str(e)}"


# 工具注册表
AVAILABLE_TOOLS = {
    # "api": APITool,                        # 暂不启用
    "database": DatabaseTool,
    # "document_search": DocumentSearchTool, # 暂不启用
    "code_execution": CodeExecutionTool,
    "serper_search": SerperSearchTool,
    "calculator": CalculatorTool
}

# 简单的进程级缓存：key -> (timestamp, payload)
_SERPER_CACHE: Dict[str, Tuple[float, Any]] = {}


def get_tool(tool_name: str) -> Optional[BaseTool]:
    """
    获取工具实例
    
    Args:
        tool_name: 工具名称
        
    Returns:
        工具实例或None
    """
    if tool_name in AVAILABLE_TOOLS:
        return AVAILABLE_TOOLS[tool_name]()
    return None


def get_all_tools() -> List[BaseTool]:
    """
    获取所有可用工具
    
    Returns:
        工具实例列表
    """
    return [tool_class() for tool_class in AVAILABLE_TOOLS.values()]


def list_tool_names() -> List[str]:
    """
    列出所有工具名称
    
    Returns:
        工具名称列表
    """
    return list(AVAILABLE_TOOLS.keys())
