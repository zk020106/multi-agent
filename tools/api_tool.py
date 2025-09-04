"""
用于发送HTTP请求的API工具。
"""

import json

import requests

from .base_tool import BaseTool, ToolResult


class APITool(BaseTool):
    """
    用于发送HTTP API请求的工具。
    """
    
    def __init__(self, name: str = "api_tool", description: str = "发送HTTP API请求"):
        super().__init__(name, description)
        
        # 定义输入模式
        self.input_schema = {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "API端点URL"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    "description": "HTTP方法"
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP请求头"
                },
                "params": {
                    "type": "object",
                    "description": "URL参数"
                },
                "data": {
                    "type": "object",
                    "description": "请求体数据"
                },
                "json": {
                    "type": "object",
                    "description": "JSON请求体"
                },
                "timeout": {
                    "type": "number",
                    "description": "请求超时时间（秒）"
                }
            },
            "required": ["url", "method"]
        }
        
        # 定义输出模式
        self.output_schema = {
            "type": "object",
            "properties": {
                "status_code": {
                    "type": "integer",
                    "description": "HTTP状态码"
                },
                "headers": {
                    "type": "object",
                    "description": "响应头"
                },
                "data": {
                    "type": "object",
                    "description": "响应数据"
                },
                "text": {
                    "type": "string",
                    "description": "响应文本"
                }
            }
        }
        
        self.required_params = ["url", "method"]
        self.optional_params = ["headers", "params", "data", "json", "timeout"]
    
    def execute(self, **kwargs) -> ToolResult:
        """
        执行API请求。
        
        参数:
            url: API端点URL
            method: HTTP方法
            headers: HTTP请求头
            params: URL参数
            data: 请求体数据
            json: JSON请求体
            timeout: 请求超时时间
            
        返回:
            包含API响应的ToolResult
        """
        try:
            url = kwargs["url"]
            method = kwargs["method"].upper()
            
            # 准备请求参数
            request_kwargs = {}
            
            if "headers" in kwargs:
                request_kwargs["headers"] = kwargs["headers"]
            
            if "params" in kwargs:
                request_kwargs["params"] = kwargs["params"]
            
            if "data" in kwargs:
                request_kwargs["data"] = kwargs["data"]
            
            if "json" in kwargs:
                request_kwargs["json"] = kwargs["json"]
            
            if "timeout" in kwargs:
                request_kwargs["timeout"] = kwargs["timeout"]
            else:
                request_kwargs["timeout"] = 30  # 默认超时时间
            
            # 发送请求
            response = requests.request(method, url, **request_kwargs)
            
            # 解析响应
            result_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "text": response.text
            }
            
            # 尝试解析JSON响应
            try:
                result_data["data"] = response.json()
            except (json.JSONDecodeError, ValueError):
                result_data["data"] = None
            
            # 检查请求是否成功
            success = 200 <= response.status_code < 300
            
            return ToolResult(
                success=success,
                data=result_data,
                error_message=None if success else f"HTTP {response.status_code}: {response.text}",
                metadata={
                    "url": url,
                    "method": method,
                    "status_code": response.status_code
                }
            )
            
        except requests.exceptions.RequestException as e:
            return ToolResult(
                success=False,
                error_message=f"请求失败: {str(e)}",
                metadata={"url": kwargs.get("url"), "method": kwargs.get("method")}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"意外错误: {str(e)}",
                metadata={"url": kwargs.get("url"), "method": kwargs.get("method")}
            )


class RESTAPITool(APITool):
    """
    具有常见REST操作的专用REST API工具。
    """
    
    def __init__(self, base_url: str = "", name: str = "rest_api_tool", description: str = "REST API操作"):
        super().__init__(name, description)
        self.base_url = base_url.rstrip("/")
    
    def _build_url(self, endpoint: str) -> str:
        """从端点构建完整URL。"""
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}" if self.base_url else endpoint
    
    def get(self, endpoint: str, **kwargs) -> ToolResult:
        """GET请求。"""
        return self.execute(
            url=self._build_url(endpoint),
            method="GET",
            **kwargs
        )
    
    def post(self, endpoint: str, **kwargs) -> ToolResult:
        """POST请求。"""
        return self.execute(
            url=self._build_url(endpoint),
            method="POST",
            **kwargs
        )
    
    def put(self, endpoint: str, **kwargs) -> ToolResult:
        """PUT请求。"""
        return self.execute(
            url=self._build_url(endpoint),
            method="PUT",
            **kwargs
        )
    
    def delete(self, endpoint: str, **kwargs) -> ToolResult:
        """DELETE请求。"""
        return self.execute(
            url=self._build_url(endpoint),
            method="DELETE",
            **kwargs
        )
    
    def patch(self, endpoint: str, **kwargs) -> ToolResult:
        """PATCH请求。"""
        return self.execute(
            url=self._build_url(endpoint),
            method="PATCH",
            **kwargs
        )


class WebhookTool(BaseTool):
    """
    用于发送webhook通知的工具。
    """
    
    def __init__(self, name: str = "webhook_tool", description: str = "发送webhook通知"):
        super().__init__(name, description)
        
        self.input_schema = {
            "type": "object",
            "properties": {
                "webhook_url": {
                    "type": "string",
                    "description": "Webhook URL"
                },
                "payload": {
                    "type": "object",
                    "description": "Webhook负载"
                },
                "headers": {
                    "type": "object",
                    "description": "额外请求头"
                }
            },
            "required": ["webhook_url", "payload"]
        }
        
        self.required_params = ["webhook_url", "payload"]
        self.optional_params = ["headers"]
    
    def execute(self, **kwargs) -> ToolResult:
        """
        发送webhook通知。
        
        参数:
            webhook_url: Webhook URL
            payload: Webhook负载
            headers: 额外请求头
            
        返回:
            包含webhook响应的ToolResult
        """
        try:
            webhook_url = kwargs["webhook_url"]
            payload = kwargs["payload"]
            headers = kwargs.get("headers", {})
            
            # 设置默认请求头
            default_headers = {
                "Content-Type": "application/json",
                "User-Agent": "Multi-Agent-System/1.0"
            }
            default_headers.update(headers)
            
            # 发送webhook
            response = requests.post(
                webhook_url,
                json=payload,
                headers=default_headers,
                timeout=30
            )
            
            success = 200 <= response.status_code < 300
            
            return ToolResult(
                success=success,
                data={
                    "status_code": response.status_code,
                    "response": response.text
                },
                error_message=None if success else f"Webhook失败: {response.status_code}",
                metadata={
                    "webhook_url": webhook_url,
                    "status_code": response.status_code
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Webhook错误: {str(e)}",
                metadata={"webhook_url": kwargs.get("webhook_url")}
            )
