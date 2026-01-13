"""
SFT Pipeline Configuration

支持三种语言：Python, Go, JavaScript
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

    # Deps count range
    min_deps: int = 3
    max_deps: int = 6

    # Files count range
    min_files: int = 3
    max_files: int = 5

    # ==================== Python 配置 ====================
    python_files: list = field(default_factory=lambda: [
        # 入口文件（优先级高）
        "main.py", "app.py", "server.py",
        # 业务文件
        "utils.py", "models.py", "routes.py", "database.py",
        "config.py", "handlers.py", "services.py", "api.py",
        # 配置/构建文件
        "config.json", "README.md", ".gitignore",
        "setup.py", "pyproject.toml", "Makefile", "tests/"
    ])

    # Python web框架依赖（用于判断web-serving变体）
    python_web_deps: list = field(default_factory=lambda: [
        "fastapi", "uvicorn"
    ])

    python_deps: list = field(default_factory=lambda: [
        "numpy", "pandas", "requests", "flask", "django", "fastapi",
        "torch", "scikit-learn", "redis", "celery", "pillow", "boto3",
        "sqlalchemy", "pymongo", "pytest", "black", "mypy", "uvicorn"
    ])

    # ==================== Go 配置 ====================
    go_files: list = field(default_factory=lambda: [
        # 入口文件（优先级高）
        "main.go",
        # 业务文件
        "handler.go", "handlers.go", "router.go", "routes.go",
        "db.go", "database.go", "model.go", "models.go",
        "service.go", "services.go", "repository.go", "middleware.go",
        "config.go", "utils.go", "helpers.go", "errors.go",
        "auth.go", "logger.go", "types.go", "constants.go",
        "api.go", "server.go", "client.go",
        # 配置/构建文件
        "config.yaml", "config.json", "Makefile", "README.md", ".gitignore"
    ])

    go_deps: list = field(default_factory=lambda: [
        # Web框架
        "github.com/gin-gonic/gin",
        "github.com/gorilla/mux",
        "github.com/labstack/echo/v4",
        "github.com/gofiber/fiber/v2",
        # 数据库
        "gorm.io/gorm",
        "gorm.io/driver/mysql",
        "gorm.io/driver/postgres",
        "github.com/go-redis/redis/v8",
        "go.mongodb.org/mongo-driver/mongo",
        # 工具库
        "github.com/spf13/viper",
        "github.com/sirupsen/logrus",
        "go.uber.org/zap",
        "github.com/golang-jwt/jwt/v5"
    ])

    # ==================== JavaScript 配置 ====================
    js_files: list = field(default_factory=lambda: [
        # 入口文件（优先级高，但JS用npm start，不需要判断）
        "index.js", "app.js", "server.js", "main.js",
        # 业务文件
        "routes.js", "router.js", "controllers.js",
        "models.js", "middleware.js", "utils.js",
        "config.js", "db.js", "database.js", "auth.js", "api.js",
        # 配置文件
        ".env", ".gitignore", "README.md",
        "src/", "tests/", "public/",
        # 构建产物目录（用于判断多阶段构建变体）
        "dist/", "build/"
    ])

    js_deps: list = field(default_factory=lambda: [
        # Web框架
        "express", "koa", "fastify", "hapi",
        # 数据库
        "mongoose", "sequelize", "prisma", "pg", "mysql2",
        # 工具库
        "axios", "dotenv", "cors", "helmet",
        "jsonwebtoken", "bcryptjs",
        "lodash", "moment", "dayjs",
        "winston", "morgan"
    ])

    # ==================== 特殊场景配置 ====================
    # 10% 概率生成refusal场景（多语言混合项目，模型应拒绝）
    refusal_probability: float = 0.1

    # 20% 概率生成critique场景（挑错题，修正错误Dockerfile）
    critique_probability: float = 0.2

    # 30% 概率引入陷阱依赖，让模型必须检查依赖才能生成正确的Dockerfile
    trap_probability: float = 0.3

    # Node.js 需要 C++ 编译环境的包（需要 node-gyp: python3, make, g++）
    js_native_deps: list = field(default_factory=lambda: [
        "bcrypt",      # 密码哈希，需要编译
        "sharp",       # 图像处理，需要编译
        "canvas",      # Canvas绘图，需要编译
        "sqlite3",     # SQLite绑定，需要编译
        "node-sass",   # Sass编译器，需要编译
    ])

    # Go 需要 CGO 的包（需要 gcc，CGO_ENABLED=1）
    go_cgo_deps: list = field(default_factory=lambda: [
        "github.com/mattn/go-sqlite3",      # SQLite驱动，需要CGO
        "github.com/boltdb/bolt",           # BoltDB，需要CGO
        "github.com/cznic/ql",              # 嵌入式SQL数据库，需要CGO
    ])
