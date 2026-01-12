"""
SFT Pipeline Configuration
"""
from dataclasses import dataclass, field
from typing import Optional
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


@dataclass
class SFTConfig:
    # Model settings
    teacher_model: str = "anthropic/claude-sonnet-4.5"  # OpenRouter 格式
    student_model: str = "Qwen/Qwen3-0.6B"

    # Data generation
    num_samples: int = 1500
    output_file: str = "sft_train_data.jsonl"

    # API settings (使用 OPENAI_* 环境变量，兼容 OpenRouter)
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))
    max_tokens: int = 2048
    temperature: float = 0.7

    # Mock environment settings
    has_requirements_txt_prob: float = 0.3  # 30% probability

    # Common dependencies pool
    common_deps: list = field(default_factory=lambda: [
        "numpy", "pandas", "requests", "flask", "django", "fastapi",
        "torch", "scikit-learn", "redis", "celery", "pillow", "boto3",
        "sqlalchemy", "pymongo", "pytest", "black", "mypy", "uvicorn"
    ])

    # Common files pool
    common_files: list = field(default_factory=lambda: [
        "app.py", "main.py", "server.py", "config.json", "README.md",
        "utils.py", "models.py", "routes.py", "database.py", "tests/",
        ".gitignore", "setup.py", "pyproject.toml", "Makefile"
    ])

    # Deps count range
    min_deps: int = 3
    max_deps: int = 6

    # Files count range
    min_files: int = 3
    max_files: int = 5
