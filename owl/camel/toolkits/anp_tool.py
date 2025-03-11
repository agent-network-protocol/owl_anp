import asyncio
import json
import yaml
import aiohttp
from pathlib import Path
from typing import Dict, Any, Optional, List

from camel.models import BaseModelBackend
from camel.toolkits.function_tool import FunctionTool

from loguru import logger
from .did_auth_client import DIDAuthClient
from camel.toolkits.base import BaseToolkit

class ANPTool(BaseToolkit):
    parameters = {
        "type": "function",
        "function": {  
            "name": "anp_tool",
            "description": "使用 Agent Network Protocol (ANP) 与其他智能体进行交互。第一次使用时，请输入 URL: https://agent-search.ai/ad.json，这是一个智能体搜索服务，可以利用里面的接口查询能够提供酒店、门票景点的智能体。收到智能体的描述文档后，你可以根据智能体描述文档中的数据链接URL，爬取数据。中间过程中可以调用API完成服务。知道你认为任务完成。注意，使用ANPTool获得的任何URL，必须使用ANPTool调用，不要自己直接调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "(required) 智能体描述文件或 API 端点的 URL",
                    },
                    "method": {
                        "type": "string",
                        "description": "(optional) HTTP 方法，如 GET、POST、PUT 等，默认为 GET",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                        "default": "GET",
                    },
                    "headers": {
                        "type": "object",
                        "description": "(optional) HTTP 请求头",
                        "default": {},
                    },
                    "params": {
                        "type": "object",
                        "description": "(optional) URL 查询参数",
                        "default": {},
                    },
                    "body": {
                        "type": "object",
                        "description": "(optional) 请求体，用于 POST/PUT 请求",
                    },
                },
                "required": ["url"],
            },
        }
    }
    
    # 声明 auth_client 字段
    auth_client: Optional[DIDAuthClient] = None

    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
    ) -> None:
        self.model = model
        # 获取当前脚本目录
        current_dir = Path(__file__).parent
        # 获取项目根目录
        base_dir = current_dir.parent.parent
        
        # 初始化 DID 身份验证客户端
        did_path = str(base_dir / "did_test_public_doc/did.json")
        key_path = str(base_dir / "did_test_public_doc/key-1_private.pem") 
        
        logger.info(f"ANPTool 初始化 - DID路径: {did_path}, 私钥路径: {key_path}")
        
        self.auth_client = DIDAuthClient(
            did_document_path=did_path,
            private_key_path=key_path
        )

    async def execute(
        self, 
        url: str, 
        method: str = "GET", 
        headers: Dict[str, str] = None, 
        params: Dict[str, Any] = None, 
        body: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        执行 HTTP 请求与其他智能体交互
        
        Args:
            url (str): 智能体描述文件或 API 端点的 URL
            method (str, optional): HTTP 方法，默认为 "GET"
            headers (Dict[str, str], optional): HTTP 请求头
            params (Dict[str, Any], optional): URL 查询参数
            body (Dict[str, Any], optional): 请求体，用于 POST/PUT 请求
            
        Returns:
            Dict[str, Any]: 响应内容
        """

        if headers is None:
            headers = {}
        if params is None:
            params = {}
        
        logger.info(f"ANP请求: {method} {url}")
        
        # 添加基本请求头
        if "Content-Type" not in headers and method in ["POST", "PUT", "PATCH"]:
            headers["Content-Type"] = "application/json"
            
        # 添加 DID 身份验证
        if self.auth_client:
            try:
                auth_headers = self.auth_client.get_auth_header(url)
                headers.update(auth_headers)
            except Exception as e:
                logger.error(f"获取身份验证头失败: {str(e)}")
        

        async with aiohttp.ClientSession() as session:
            # 准备请求参数
            request_kwargs = {
                "url": url,
                "headers": headers,
                "params": params,
            }
            
            # 如果有请求体且方法支持，添加请求体
            if body is not None and method in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = body
                
            # 执行请求
            http_method = getattr(session, method.lower())
            
            try:
                async with http_method(**request_kwargs) as response:
                    logger.info(f"ANP响应: 状态码 {response.status}")
                    
                    # 检查响应状态
                    if response.status == 401 and "Authorization" in headers and self.auth_client:
                        logger.warning("认证失败 (401)，尝试重新获取身份验证")
                        # 如果认证失败且使用了令牌，清除令牌并重试
                        self.auth_client.clear_token(url)
                        # 重新获取身份验证头
                        headers.update(self.auth_client.get_auth_header(url, force_new=True))
                        # 重新执行请求
                        request_kwargs["headers"] = headers
                        async with http_method(**request_kwargs) as retry_response:
                            logger.info(f"ANP重试响应: 状态码 {retry_response.status}")
                            return await self._process_response(retry_response, url)

                    return await self._process_response(response, url)
            except aiohttp.ClientError as e:
                logger.error(f"HTTP 请求失败: {str(e)}")
                return {
                    "error": f"HTTP 请求失败: {str(e)}",
                    "status_code": 500
                }

    
    async def _process_response(self, response, url):
        """处理 HTTP 响应"""
        # 如果认证成功，更新令牌
        if response.status == 200 and self.auth_client:
            try:
                self.auth_client.update_token(url, dict(response.headers))
            except Exception as e:
                logger.error(f"更新令牌失败: {str(e)}")
        
        # 获取响应内容类型
        content_type = response.headers.get('Content-Type', '').lower()
        
        # 获取响应文本
        text = await response.text()
        
        # 根据内容类型处理响应
        if 'application/json' in content_type:
            # 处理 JSON 响应
            try:
                result = json.loads(text)
                logger.info("成功解析 JSON 响应")
            except json.JSONDecodeError:
                logger.warning("Content-Type 声明为 JSON 但解析失败，返回原始文本")
                result = {"text": text, "format": "text", "content_type": content_type}
        elif 'application/yaml' in content_type or 'application/x-yaml' in content_type:
            # 处理 YAML 响应
            try:
                result = yaml.safe_load(text)
                logger.info("成功解析 YAML 响应")
                result = {"data": result, "format": "yaml", "content_type": content_type}
            except yaml.YAMLError:
                logger.warning("Content-Type 声明为 YAML 但解析失败，返回原始文本")
                result = {"text": text, "format": "text", "content_type": content_type}
        else:
            # 尝试自动检测格式
            # 首先尝试解析为 JSON
            try:
                result = json.loads(text)
                logger.info("自动检测到 JSON 格式并成功解析")
                result = {"data": result, "format": "json", "content_type": content_type}
            except json.JSONDecodeError:
                # 然后尝试解析为 YAML
                if text.strip() and (':' in text or '-' in text):  # 简单检查是否可能是 YAML
                    try:
                        result = yaml.safe_load(text)
                        if isinstance(result, (dict, list)):  # 确认是结构化数据
                            logger.info("自动检测到 YAML 格式并成功解析")
                            result = {"data": result, "format": "yaml", "content_type": content_type}
                        else:
                            # 可能是纯文本但恰好符合 YAML 语法
                            result = {"text": text, "format": "text", "content_type": content_type}
                    except yaml.YAMLError:
                        # 不是 YAML，返回原始文本
                        result = {"text": text, "format": "text", "content_type": content_type}
                else:
                    # 不像是 YAML，返回原始文本
                    result = {"text": text, "format": "text", "content_type": content_type}
        
        # 添加状态码到结果
        if isinstance(result, dict):
            result["status_code"] = response.status
        else:
            result = {"data": result, "status_code": response.status, "format": "unknown", "content_type": content_type}
        
        # 添加 URL 到结果，方便追踪
        result["url"] = str(url)
            
        return result
    
    def anp_tool(self, url: str, method: str = "GET", headers: Dict[str, str] = None, params: Dict[str, Any] = None, body: Dict[str, Any] = None) -> Dict[str, Any]:
        """使用 Agent Network Protocol (ANP) 与其他智能体进行交互。
        第一次使用时，请输入 URL: https://agent-search.ai/ad.json，这是一个智能体搜索服务，可以利用里面的接口查询能够提供酒店、门票景点的智能体。
        收到智能体的描述文档后，你可以根据智能体描述文档中的数据链接URL，爬取数据。中间过程中可以调用API完成服务。知道你认为任务完成。
        注意，使用ANPTool获得的任何URL，必须使用ANPTool调用，不要自己直接调用。
        """
        # 使用asyncio运行异步方法
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute(url, method, headers, params, body))
        
    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions in the toolkit.
        
        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the functions in the toolkit.
        """
        return [
            FunctionTool(self.anp_tool, openai_tool_schema=self.parameters),
        ]