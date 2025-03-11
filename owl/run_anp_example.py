from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

from typing import List, Dict
from loguru import logger

from utils import OwlRolePlaying, run_society
import os
from dotenv import load_dotenv
from pathlib import Path

# 导入我们的酒店工具包
from camel.toolkits import *

# Get the absolute path to the root directory
ROOT_DIR = Path(__file__).resolve().parent


# Load environment variables
load_dotenv(ROOT_DIR / ".env")

def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct the society based on the question."""

    user_role_name = "user"
    assistant_role_name = "assistant"
    
    user_model = ModelFactory.create(
        model_platform=ModelPlatformType.AZURE,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(), # [Optional] the config for model
    )

    assistant_model = ModelFactory.create(
        model_platform=ModelPlatformType.AZURE,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig(temperature=0, top_p=1).as_dict(), # [Optional] the config for model
    )
    
    # 创建工具列表，只包含酒店工具包
    tools_list = [
        *ANPTool().get_tools(),
    ]

    user_agent_kwargs = dict(model=user_model)
    assistant_agent_kwargs = dict(model=assistant_model, tools=tools_list)
    
    task_kwargs = {
        'task_prompt': question,
        'with_task_specify': False,
    }

    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name=user_role_name,
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name=assistant_role_name,
        assistant_agent_kwargs=assistant_agent_kwargs,
    )
    
    return society


# 示例问题
question = "我需要预订杭州的一个酒店：2025年3月9日，1天的酒店，经纬度（120.026208, 30.279212）。请一步步处理：第一步，你自己选择一个不错的酒店，第二步，帮我选择一个房间。最后告诉我你选择的详细信息"

society = construct_society(question)
answer, chat_history, token_count = run_society(society)

logger.success(f"Answer: {answer}")