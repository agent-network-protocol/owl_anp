# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

from typing import List, Dict, Any, Optional
import requests
import json
from datetime import datetime
from loguru import logger

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelType, ModelPlatformType
from typing import Optional, Dict, Any, List, Union
import os
import json
import logging
import aiohttp
import asyncio
import time
import traceback
import uuid
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
from utils.log_base import set_log_color_level

# Get the absolute path to the root directory
ROOT_DIR = Path(__file__).resolve().parent


# Load environment variables
load_dotenv(ROOT_DIR / ".env")

SEARCH_AGENT_PROMPT_TEMPLATE = """
你是一个通用智能网络数据探索工具。你的目标是通过递归访问各种格式的数据（包括JSON-LD、YAML等），找到用户需要的信息、API，以完成指定的任务。

## 当前任务
{task_description}

## 重要说明
1. 你将收到一个起始URL({initial_url})，这是一个搜索智能体的描述文件
2. 你需要理解这个搜索智能体的结构、功能和API用法
3. 你需要像网络爬虫一样，不断从中发现并访问新的URL和API端点
4. 你可以使用fetch_url工具来获取任何URL的内容
5. 该工具可以处理多种格式的响应，包括：
   - JSON格式：将直接解析为JSON对象
   - YAML格式：将返回文本内容，你需要分析其结构
   - 其他文本格式：将返回原始文本内容
6. 阅读每个文档，寻找与任务相关的信息或API端点
7. 你需要自己决定爬取路径，不要等待用户指示

## 爬取策略
1. 首先获取初始URL内容，理解搜索智能体的结构和API
2. 识别文档中的所有URL和链接，特别是serviceEndpoint、url、@id等字段
3. 分析API文档，理解API的使用方法、参数和返回值
4. 根据API文档，构造合适的请求找到所需信息
5. 记录你访问过的所有URL，避免重复爬取
6. 总结发现的所有相关信息，提供详细的建议

## 工作流程
1. 获取起始URL内容，理解搜索智能体的功能
2. 分析内容，找出所有可能的链接和API文档
3. 解析API文档，理解API的使用方法
4. 根据任务需求，构造请求获取所需信息
5. 继续探索相关链接，直到找到足够的信息
6. 总结信息，提供最适合用户的建议

## JSON-LD数据解析提示
1. 注意@context字段，它定义了数据的语义上下文
2. @type字段表示实体类型，帮助你理解数据的含义
3. @id字段通常是一个可以进一步访问的URL
4. 寻找serviceEndpoint、url等字段，它们通常指向API或更多数据

提供详细信息和清晰解释，让用户理解你找到的信息和你的推荐理由。
"""

# 定义初始URL
initial_url = "https://agent-search.ai/ad.json"

# 定义可用的工具
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch content from a URL, supporting various formats like JSON, YAML, and plain text",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch content from"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method to use (GET or POST)",
                        "enum": ["GET", "POST"]
                    },
                    "data": {
                        "type": "object",
                        "description": "Data to send in the request body (for POST requests)"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Additional headers to send with the request"
                    }
                },
                "required": ["url"]
            }
        }
    }
]


class HotelToolkit(BaseToolkit):
    r"""A toolkit for querying hotel information.
    
    This toolkit provides methods to search for hotels, get detailed information
    about specific hotels, check room availability, and make reservations.
    
    Args:
        model (Optional[BaseModelBackend]): The model to use for enhancing hotel search and recommendations.

    """
    
    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,

    ) -> None:
        self.model = model
        self.progress_list = []
        self.visited_urls = set()
        
        # 初始化 Azure OpenAI 客户端
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        )

    def add_progress_step(self, step_id: str, title: str, status: str = 'pending', details: Dict = None) -> Dict:
        """添加一个新的处理步骤"""
        step = {
            'id': step_id,
            'title': title,
            'status': status,
            'timestamp': time.time()
        }
        if details:
            step['details'] = details
        self.progress_list.append(step)
        logging.info(f"Added progress step: {step}")
        return step

    def update_progress(self, step_id: str, status: str) -> None:
        """更新进度状态"""
        for step in self.progress_list:
            if step['id'] == step_id:
                step['status'] = status
                step['timestamp'] = time.time()
                logging.info(f"Updated progress step {step_id} to {status}: {step}")
                return

    # 已删除 process_http_response 函数，因为它已经不再被使用，相关功能已整合到 fetch_url_content 函数中

    def fetch_url_content(self, url: str, method: str = "GET", data: Dict = None, headers: Dict = None) -> Dict[str, Any]:
        """获取URL内容"""
        logging.info(f"Fetching document from URL: {url} with method: {method}")
        try:
            request_kwargs = {}
            if headers:
                request_kwargs["headers"] = headers
            if data and method == "POST":
                request_kwargs["json"] = data
            
            if method == "GET":
                response = requests.get(url, **request_kwargs)
            elif method == "POST":
                response = requests.post(url, **request_kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            # 处理响应
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    return response.json()
                else:
                    return {"content": response.text, "content_type": content_type}
            else:
                return {"error": f"HTTP error {response.status_code}", "content": response.text}
        except Exception as e:
            logging.error(f"Error fetching URL {url} with method {method}: {str(e)}")
            return {"error": str(e)}

    def handle_tool_call(self, tool_call: Any, messages: List[Dict]) -> None:
        """处理工具调用"""
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "fetch_url":
            url = function_args.get("url")
            method = function_args.get("method", "GET")
            data = function_args.get("data")
            headers = function_args.get("headers")
            
            if url in self.visited_urls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "error": f"你已经访问过这个URL: {url}",
                        "suggestion": "请尝试访问不同的URL，或者基于已有信息提供总结和建议。"
                    }),
                })
                return
            
            random_id = str(uuid.uuid4())[:8]
            self.add_progress_step(f'fetch_url_{random_id}', f'获取URL内容: {url}', 'in-progress', {'url': url})
            
            try:
                result = self.fetch_url_content(url, method=method, data=data, headers=headers)
                logging.info(f"HTTP response [url: {url}]:")
                logging.info(result)

                self.visited_urls.add(url)
                self.update_progress(f'fetch_url_{random_id}', 'completed')
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            except Exception as e:
                logging.error(f"Error fetching URL {url}: {str(e)}")
                self.update_progress(f'fetch_url_{random_id}', 'error')
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "error": f"Failed to fetch URL: {url}",
                        "message": str(e)
                    }),
                })

    def process_user_input(self, user_input: str, task_type: str = "general") -> Dict[str, Any]:
        """处理用户输入"""
        logging.info(f"Starting to process user input for task type: {task_type}")
        
        try:
            # 重置状态
            self.progress_list.clear()
            self.visited_urls.clear()
            
            # 添加初始步骤
            self.add_progress_step('create_agent', f'启动{task_type}搜索助手', 'in-progress')
            self.add_progress_step('fetch_initial_url', '获取搜索智能体描述', 'in-progress', {'url': initial_url})
            
            # 获取初始URL内容
            initial_content = self.fetch_url_content(initial_url)
            self.visited_urls.add(initial_url)
            self.update_progress('fetch_initial_url', 'completed')
            
            # 准备消息列表
            messages = self.prepare_initial_messages(user_input, initial_content)
            
            # 处理对话
            result = self.process_conversation(messages, task_type)
            return result
            
        except Exception as e:
            logging.error(f"Error processing user input: {str(e)}")
            logging.error(traceback.format_exc())
            for step in progress_list:
                if step['status'] == 'in-progress':
                    step['status'] = 'error'
            raise e

    def prepare_initial_messages(self, user_input: str, initial_content: Dict) -> List[Dict]:
        """准备初始消息列表"""
        formatted_prompt = SEARCH_AGENT_PROMPT_TEMPLATE.format(
            task_description=user_input,
            initial_url=initial_url
        )
        
        return [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_input},
            {"role": "system", "content": f"我已经获取了初始URL的内容。以下是搜索智能体的描述数据:\n\n```json\n{json.dumps(initial_content, ensure_ascii=False, indent=2)}\n```\n\n请分析这个数据，理解搜索智能体的功能和API用法。找出你需要访问的链接，通过fetch_url工具来获取更多信息，完成用户的任务。"}
        ]

    def process_conversation(self, messages: List[Dict], task_type: str) -> Dict[str, Any]:
        """处理对话流程"""
        max_iterations = 15
        current_iteration = 0
        
        while current_iteration < max_iterations:
            current_iteration += 1
            logging.info(f"Starting iteration {current_iteration}/{max_iterations}")
            
            # 获取模型响应
            completion = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_MODEL"),
                messages=messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
            )
            
            response_message = completion.choices[0].message
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": response_message.tool_calls
            })
            
            # 检查是否结束对话
            if self.should_end_conversation(response_message, current_iteration, max_iterations):
                self.update_progress('create_agent', 'completed')
                return self.create_final_response(response_message, task_type)
                
            # 处理工具调用
            for tool_call in response_message.tool_calls:
                self.handle_tool_call(tool_call, messages)

    def should_end_conversation(self, response_message: Any, current_iteration: int, max_iterations: int) -> bool:
        """判断是否应该结束对话"""
        if not response_message.tool_calls:
            return True
            
        if current_iteration >= max_iterations - 1:
            return True
            
        return False

    def create_final_response(self, response_message: Any, task_type: str) -> Dict[str, Any]:
        """创建最终响应"""
        return {
            "content": response_message.content,
            "type": "text",
            "visited_urls": list(self.visited_urls),
            "task_type": task_type
        }

    def recommend_room_with_model_v2(self, question: str, guest_preferences: str = "") -> Dict[str, Any]:
        """使用模型推荐酒店房间

        Args:
            question (str): 用户的酒店查询问题
            guest_preferences (str, optional): 客人偏好. Defaults to "".

        Returns:
            Dict[str, Any]: 推荐结果
        """
        # 合并问题和偏好
        if guest_preferences:
            full_query = f"{question}\n客人偏好: {guest_preferences}"
        else:
            full_query = question
            
        logger.info(f"开始处理酒店查询: {full_query}")
        
        # 创建任务
        task = {
            "input": full_query,
            # "type": "hotel_booking"
        }
        
        # logger.info(f"\n=== 测试任务: {task['type']} ===")
        logger.info(f"用户输入: {task['input']}")
        
        # 使用现有的 process_user_input 函数处理用户请求
        try:
            result = self.process_user_input(task['input'])
            
            # 构建结果
            response = {
                "recommendation": result["content"],
                "timestamp": datetime.now().isoformat(),
                "query": full_query,
                "visited_urls": result.get("visited_urls", [])
            }
            
            logger.info("酒店推荐生成成功")
            return response
            
        except Exception as e:
            logger.error(f"生成酒店推荐时出错: {e}")
            return {
                "error": f"生成酒店推荐时出错: {str(e)}",
                "query": full_query
            }
            
    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the functions in the toolkit.
        
        Returns:
            List[FunctionTool]: A list of FunctionTool objects representing the functions in the toolkit.
        """
        return [
            FunctionTool(self.recommend_room_with_model_v2),
        ]