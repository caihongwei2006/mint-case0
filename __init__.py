"""
SFT Pipeline for Dockerfile Generation

用于生成SFT训练数据，让小模型学会：
1. 使用工具扫描项目结构
2. 分析项目依赖
3. 生成正确的Dockerfile

Usage:
    # 生成少量测试样本
    python -m sft_pipeline.data_generator --num-samples 10

    # 生成完整数据集
    python -m sft_pipeline.data_generator --num-samples 1500 --output sft_train_data.jsonl
"""

from .config import SFTConfig
from .tools import MockEnvironment, TOOL_DEFINITIONS
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .data_generator import SFTDataGenerator
from .data_converter import load_sft_data_for_mint, convert_messages_to_text

__all__ = [
    "SFTConfig",
    "MockEnvironment",
    "TOOL_DEFINITIONS",
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "SFTDataGenerator",
    "load_sft_data_for_mint",
    "convert_messages_to_text",
]
