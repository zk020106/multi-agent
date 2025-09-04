"""
基于LangChain的工具实现
提供各种智能体可用的工具
"""

import json
import os
import sqlite3
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Type

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
    document_path: str = Field(description="文档文件路径")
    max_results: Optional[int] = Field(default=5, description="最大结果数量")


class DocumentSearchTool(BaseTool):
    """文档搜索工具"""
    
    name: str = "document_search_tool"
    description: str = "在文档中搜索相关内容，支持文本文件搜索"
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    def _run(
        self,
        query: str,
        document_path: str,
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
    language: str = Field(default="python", description="编程语言")


class CodeExecutionTool(BaseTool):
    """代码执行工具"""
    
    name: str = "code_execution_tool"
    description: str = "执行Python代码并返回结果，支持基本的Python代码执行"
    args_schema: Type[BaseModel] = CodeExecutionToolInput
    
    def _run(
        self,
        code: str,
        language: str = "python",
        **kwargs
    ) -> str:
        """
        执行代码
        
        Args:
            code: 要执行的代码
            language: 编程语言
            
        Returns:
            执行结果
        """
        if language.lower() != "python":
            return f"暂不支持 {language} 语言，仅支持Python"
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # 执行代码
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                output = {
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
                if result.returncode == 0:
                    return f"代码执行成功:\n{result.stdout}"
                else:
                    return f"代码执行失败:\n{result.stderr}"
                    
            finally:
                # 清理临时文件
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return "代码执行超时（30秒）"
        except Exception as e:
            return f"代码执行失败: {str(e)}"


class WebSearchToolInput(BaseModel):
    """网络搜索工具输入模型"""
    query: str = Field(description="搜索查询")
    max_results: Optional[int] = Field(default=5, description="最大结果数量")


class WebSearchTool(BaseTool):
    """网络搜索工具（简化版本）"""
    
    name: str = "web_search_tool"
    description: str = "模拟网络搜索功能"
    args_schema: Type[BaseModel] = WebSearchToolInput
    
    def _run(
        self,
        query: str,
        max_results: Optional[int] = 5,
        **kwargs
    ) -> str:
        """
        执行网络搜索（模拟版本）
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            
        Returns:
            搜索结果
        """
        try:
            # 模拟搜索结果
            result = f"模拟搜索结果 for '{query}':\n"
            result += f"1. 相关结果 1\n"
            result += f"2. 相关结果 2\n"
            result += f"3. 相关结果 3\n"
            return result
        except Exception as e:
            return f"网络搜索失败: {str(e)}"


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
    "api": APITool,
    "database": DatabaseTool,
    "document_search": DocumentSearchTool,
    "code_execution": CodeExecutionTool,
    "web_search": WebSearchTool,
    "calculator": CalculatorTool
}


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
