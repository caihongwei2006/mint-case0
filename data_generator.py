"""
SFT Data Generator (Multi-Language Version)

支持三种语言：Python, Go, JavaScript
简化的2步流程：
1. ls_file_tree → CoT分析文件结构
2. ls_dependencies → CoT分析依赖 + 生成Dockerfile

每个assistant回复都包含CoT，展示当前进度
"""
import json
import random
from typing import Optional
from pathlib import Path
from tqdm import tqdm

try:
    from .config import SFTConfig
    from .tools import MockEnvironment, TOOL_DEFINITIONS
except ImportError:
    from config import SFTConfig
    from tools import MockEnvironment, TOOL_DEFINITIONS


class SFTDataGenerator:
    """SFT数据生成器 - 多语言版"""

    def __init__(self, config: Optional[SFTConfig] = None):
        self.config = config or SFTConfig()

    def generate_single_sample(self) -> Optional[dict]:
        """生成单个SFT样本"""
        # 1. 创建Mock环境
        env = MockEnvironment(self.config)
        env_state = env.get_state()
        language = env_state["language"]

        # 2. 检查是否是refusal场景
        if env_state.get("is_refusal"):
            return self._generate_refusal_sample(env, env_state)

        # 3. 检查是否是critique场景（20%概率）
        if random.random() < self.config.critique_probability:
            return self._generate_critique_sample(env, env_state)

        # 4. 执行工具调用
        file_tree_result = env.execute_tool("ls_file_tree", {})
        deps_result = env.execute_tool("ls_dependencies", {})

        # 5. 根据语言生成CoT和Dockerfile
        step1_cot = self._generate_step1_cot(env_state, file_tree_result)
        final_cot = self._generate_final_cot(env_state)
        dockerfile = self._generate_dockerfile(env_state)

        # 6. 构建消息序列
        messages = self._build_messages(
            language=language,
            file_tree_result=file_tree_result,
            deps_result=deps_result,
            step1_cot=step1_cot,
            final_cot=final_cot,
            dockerfile=dockerfile
        )

        return {
            "messages": messages,
            "metadata": {
                "language": language,
                "has_dep_file": env_state.get("has_dep_file", False),
                "mock_deps": env_state.get("mock_deps", []),
                "mock_files": env_state.get("mock_files", []),
                "entry_file": env_state.get("entry_file"),
                "has_cgo_trap": env_state.get("has_cgo_trap", False),
                "has_native_trap": env_state.get("has_native_trap", False),
                "trap_dep": env_state.get("trap_dep"),
                "variant": env_state.get("variant")
            }
        }

    def _generate_refusal_sample(self, env: MockEnvironment, env_state: dict) -> dict:
        """生成refusal场景的样本"""
        file_tree_result = env.execute_tool("ls_file_tree", {})

        # 构建refusal消息序列
        messages = [
            {
                "role": "system",
                "content": "你是一个DevOps助手，负责分析项目结构并生成Dockerfile。你有两个工具：ls_file_tree（查看文件结构）和ls_dependencies（获取依赖列表）。"
            },
            {
                "role": "user",
                "content": "请为当前项目生成Dockerfile。"
            },
            {
                "role": "assistant",
                "content": "<tool_call>\nls_file_tree()\n</tool_call>"
            },
            {
                "role": "user",
                "content": f"<tool_result>\n{file_tree_result}\n</tool_result>"
            },
            {
                "role": "assistant",
                "content": "<思考>\n这个项目包含多种编程语言的文件（如.py、.go、.js混合），是一个多编程语言系统。为单一语言项目生成Dockerfile是我的能力范围，但多语言混合项目的构建配置较为复杂，超出了我的能力范围，我应该拒绝回复。\n</思考>"
            }
        ]

        return {
            "messages": messages,
            "metadata": {
                "language": "refusal",
                "is_refusal": True,
                "mock_files": env_state.get("mock_files", [])
            }
        }

    def _generate_critique_sample(self, env: MockEnvironment, env_state: dict) -> dict:
        """生成critique场景的样本（挑错题）"""
        language = env_state["language"]
        file_tree_result = env.execute_tool("ls_file_tree", {})
        deps_result = env.execute_tool("ls_dependencies", {})

        # 生成正确的Dockerfile
        correct_dockerfile = self._generate_dockerfile(env_state)

        # 生成错误的Dockerfile（随机删除/修改1-3行）
        broken_dockerfile, removed_lines = self._break_dockerfile(correct_dockerfile)

        # 生成分析思考
        critique_cot = self._generate_critique_cot(env_state, removed_lines)

        # 构建critique消息序列 - system prompt中包含错误的dockerfile
        lang_name = {"python": "Python", "go": "Go", "javascript": "Node.js"}.get(language, language)

        messages = [
            {
                "role": "system",
                "content": f"你是一个DevOps助手，负责分析项目结构并生成Dockerfile。你有两个工具：ls_file_tree（查看文件结构）和ls_dependencies（获取依赖列表）。\n\n目前有一个{lang_name}项目的Dockerfile待你纠正：\n<dockerfile>\n{broken_dockerfile}\n</dockerfile>"
            },
            {
                "role": "user",
                "content": "请帮我检查并修正这个Dockerfile。"
            },
            {
                "role": "assistant",
                "content": "<tool_call>\nls_file_tree()\n</tool_call>"
            },
            {
                "role": "user",
                "content": f"<tool_result>\n{file_tree_result}\n</tool_result>"
            },
            {
                "role": "assistant",
                "content": f"<思考>\n查看文件结构后，我需要进一步确认依赖情况来判断Dockerfile的问题。\n</思考>\n\n<tool_call>\nls_dependencies()\n</tool_call>"
            },
            {
                "role": "user",
                "content": f"<tool_result>\n{deps_result}\n</tool_result>"
            },
            {
                "role": "assistant",
                "content": f"<思考>\n{critique_cot}\n</思考>\n\n<dockerfile>\n{correct_dockerfile}\n</dockerfile>"
            }
        ]

        return {
            "messages": messages,
            "metadata": {
                "language": language,
                "is_critique": True,
                "removed_lines": removed_lines,
                "variant": env_state.get("variant")
            }
        }

    def _break_dockerfile(self, dockerfile: str) -> tuple:
        """破坏Dockerfile，随机删除1-3行，返回(破坏后的dockerfile, 删除的行列表)"""
        lines = dockerfile.strip().split('\n')

        # 不能删除FROM行（第一行）
        deletable_indices = list(range(1, len(lines)))

        if len(deletable_indices) < 1:
            return dockerfile, []

        # 随机删除1-3行
        num_to_delete = min(random.randint(1, 3), len(deletable_indices))
        indices_to_delete = sorted(random.sample(deletable_indices, num_to_delete), reverse=True)

        removed_lines = []
        for idx in indices_to_delete:
            removed_lines.append(lines[idx])
            del lines[idx]

        return '\n'.join(lines), removed_lines[::-1]  # 反转回原顺序

    def _generate_critique_cot(self, env_state: dict, removed_lines: list) -> str:
        """生成critique场景的思考内容"""
        language = env_state["language"]
        lang_name = {"python": "Python", "go": "Go", "javascript": "Node.js"}.get(language, language)

        # 分析删除的行
        missing_parts = []
        for line in removed_lines:
            line_stripped = line.strip()
            if line_stripped.startswith("RUN pip install"):
                missing_parts.append("依赖安装步骤（pip install）")
            elif line_stripped.startswith("RUN npm"):
                missing_parts.append("依赖安装步骤（npm ci）")
            elif line_stripped.startswith("RUN go"):
                missing_parts.append("Go构建/依赖下载步骤")
            elif line_stripped.startswith("RUN apt-get"):
                missing_parts.append("系统依赖安装步骤")
            elif line_stripped.startswith("COPY"):
                missing_parts.append("文件复制步骤（COPY）")
            elif line_stripped.startswith("WORKDIR"):
                missing_parts.append("工作目录设置（WORKDIR）")
            elif line_stripped.startswith("ENV"):
                missing_parts.append("环境变量设置（ENV）")
            elif line_stripped.startswith("EXPOSE"):
                missing_parts.append("端口暴露（EXPOSE）")
            elif line_stripped.startswith("CMD"):
                missing_parts.append("启动命令（CMD）")
            elif line_stripped.startswith("FROM") and "AS" in line_stripped:
                missing_parts.append("多阶段构建的阶段定义")
            else:
                missing_parts.append(f"关键指令：{line_stripped[:30]}...")

        missing_str = "、".join(missing_parts) if missing_parts else "某些关键配置"

        return f"当前Dockerfile缺少了核心组件：{missing_str}。根据项目的依赖和文件结构，我需要补充这些缺失的部分。"

    # ==================== Step 1 CoT ====================

    def _generate_step1_cot(self, env_state: dict, file_tree: str) -> str:
        """生成第一步的CoT（分析文件树后）"""
        language = env_state["language"]

        if language == "python":
            return self._generate_python_step1_cot(env_state)
        elif language == "go":
            return self._generate_go_step1_cot(env_state)
        else:  # javascript
            return self._generate_js_step1_cot(env_state)

    def _generate_python_step1_cot(self, env_state: dict) -> str:
        files = env_state["mock_files"]
        has_dep = env_state["has_dep_file"]
        entry_file = env_state.get("entry_file", "main.py")

        file_list = "、".join(files[:5])
        if len(files) > 5:
            file_list += f"等{len(files)}个文件"

        if has_dep:
            return f"""我查看了当前项目的文件结构，发现这是一个Python项目，包含{file_list}。

关键发现：
1. 项目中存在 requirements.txt 文件，依赖问题已解决，可以直接使用 pip install -r requirements.txt 安装依赖。
2. 入口文件是 {entry_file}。

接下来我需要确认具体的依赖列表，以便生成完整的Dockerfile。"""
        else:
            return f"""我查看了当前项目的文件结构，发现这是一个Python项目，包含{file_list}。

关键发现：
1. 项目中没有 requirements.txt 文件，需要分析代码来确定依赖。
2. 入口文件是 {entry_file}。

接下来我需要查看具体依赖，才能生成正确的Dockerfile。"""

    def _generate_go_step1_cot(self, env_state: dict) -> str:
        files = env_state["mock_files"]

        file_list = "、".join(files[:5])
        if len(files) > 5:
            file_list += f"等{len(files)}个文件"

        return f"""我查看了当前项目的文件结构，发现这是一个Go项目，包含{file_list}。

关键发现：
1. 项目中存在 go.mod 和 go.sum 文件，依赖问题已解决，go mod download 会自动处理依赖。
2. 入口文件是 main.go。

接下来我需要确认依赖情况，以便生成完整的Dockerfile。"""

    def _generate_js_step1_cot(self, env_state: dict) -> str:
        files = env_state["mock_files"]

        file_list = "、".join(files[:5])
        if len(files) > 5:
            file_list += f"等{len(files)}个文件"

        return f"""我查看了当前项目的文件结构，发现这是一个Node.js项目，包含{file_list}。

关键发现：
1. 项目中存在 package.json 和 package-lock.json 文件，依赖问题已解决，npm ci 会自动安装依赖。
2. 启动命令使用 npm start。

接下来我需要确认依赖情况，以便生成完整的Dockerfile。"""

    # ==================== Final CoT ====================

    def _generate_final_cot(self, env_state: dict) -> str:
        """生成最终的CoT（分析依赖后，生成Dockerfile前）"""
        language = env_state["language"]

        if language == "python":
            return self._generate_python_final_cot(env_state)
        elif language == "go":
            return self._generate_go_final_cot(env_state)
        else:  # javascript
            return self._generate_js_final_cot(env_state)

    def _generate_python_final_cot(self, env_state: dict) -> str:
        deps = env_state["mock_deps"]
        has_dep = env_state["has_dep_file"]
        entry_file = env_state.get("entry_file", "main.py")
        files = env_state["mock_files"]
        variant = env_state.get("variant", "reproducible")
        has_setup = env_state.get("has_setup", False)

        deps_str = "、".join(deps[:5])
        if len(deps) > 5:
            deps_str += f"等{len(deps)}个依赖"

        cot = f"""1. 项目认知：这是一个Python项目，包含{len(files)}个文件，入口文件为{entry_file}。

2. 依赖分析：项目依赖包括{deps_str}。

3. 安装策略："""

        if variant == "web-serving":
            cot += f"""检测到 fastapi/uvicorn，这是一个Web服务项目。
   - 使用 uvicorn 启动服务，而非直接 python 运行
   - CMD 使用 uvicorn {entry_file.replace('.py', '')}:app --host 0.0.0.0

4. Dockerfile结构：
   - FROM: python:3.9-slim
   - WORKDIR: /app
   - COPY: 复制依赖文件并安装
   - RUN: pip install
   - COPY: 复制项目文件
   - EXPOSE: 8000
   - CMD: uvicorn {entry_file.replace('.py', '')}:app --host 0.0.0.0"""
        elif variant == "editable-install":
            setup_file = "pyproject.toml" if "pyproject.toml" in files else "setup.py"
            cot += f"""检测到 {setup_file}，这是一个可安装的Python包。
   - 使用 pip install . 安装项目本身
   - 这样可以正确处理包的入口点和依赖

4. Dockerfile结构：
   - FROM: python:3.9-slim
   - WORKDIR: /app
   - COPY: 复制项目文件
   - RUN: pip install .
   - CMD: python {entry_file}"""
        else:  # reproducible
            if has_dep:
                cot += f"""项目存在requirements.txt，使用 pip install -r requirements.txt 安装依赖。

4. Dockerfile结构：
   - FROM: python:3.9-slim
   - WORKDIR: /app
   - COPY: 先复制requirements.txt，再复制其他文件
   - RUN: pip install -r requirements.txt
   - CMD: python {entry_file}"""
            else:
                deps_install = " ".join(deps)
                cot += f"""项目不存在requirements.txt，使用 pip install {deps_install} 直接安装依赖。

4. Dockerfile结构：
   - FROM: python:3.9-slim
   - WORKDIR: /app
   - RUN: pip install {deps_install}
   - COPY: 复制项目文件
   - CMD: python {entry_file}"""

        return cot

    def _generate_go_final_cot(self, env_state: dict) -> str:
        deps = env_state["mock_deps"]
        files = env_state["mock_files"]
        has_cgo_trap = env_state.get("has_cgo_trap", False)
        trap_dep = env_state.get("trap_dep")
        variant = env_state.get("variant", "static-scratch")

        deps_str = "、".join([d.split("/")[-1] for d in deps[:3]])
        if len(deps) > 3:
            deps_str += f"等{len(deps)}个依赖"

        if has_cgo_trap and trap_dep:
            trap_name = trap_dep.split("/")[-1]
            return f"""1. 项目认知：这是一个Go项目，包含{len(files)}个文件，入口文件为main.go。

2. 依赖分析：项目依赖包括{deps_str}，go.mod已定义所有依赖。

3. **关键发现**：检测到 {trap_dep}，这是一个需要 CGO 的包。
   - {trap_name} 需要 C 编译器（gcc）才能编译
   - 必须设置 CGO_ENABLED=1
   - 基础镜像 golang:alpine 不包含 gcc，需要安装或使用 golang:1.22（基于 Debian）

4. 构建策略：使用多阶段构建，在 builder 阶段安装 gcc 并启用 CGO。

5. Dockerfile结构：
   - 第一阶段(builder): golang:1.22，安装 gcc，设置 CGO_ENABLED=1
   - RUN apt-get update && apt-get install -y gcc
   - ENV CGO_ENABLED=1
   - go build -o server
   - 第二阶段(runtime): debian:bookworm-slim，只包含运行时"""
        else:
            # 静态构建，使用scratch镜像
            return f"""1. 项目认知：这是一个Go项目，包含{len(files)}个文件，入口文件为main.go。

2. 依赖分析：项目依赖包括{deps_str}，go.mod已定义所有依赖。

3. 构建策略：纯Go项目，无CGO依赖，可以使用静态编译。
   - 设置 CGO_ENABLED=0 禁用CGO
   - 使用 -ldflags="-s -w" 减小二进制体积
   - 运行时使用 scratch 空镜像，最小化攻击面

4. Dockerfile结构：
   - 第一阶段(builder): golang:1.22-alpine，静态编译
   - ENV CGO_ENABLED=0
   - go build -ldflags="-s -w" -o server
   - 第二阶段(runtime): scratch，只包含二进制文件"""

    def _generate_js_final_cot(self, env_state: dict) -> str:
        deps = env_state["mock_deps"]
        files = env_state["mock_files"]
        has_native_trap = env_state.get("has_native_trap", False)
        trap_dep = env_state.get("trap_dep")
        variant = env_state.get("variant", "simple")
        has_build_dir = env_state.get("has_build_dir", False)

        deps_str = "、".join(deps[:4])
        if len(deps) > 4:
            deps_str += f"等{len(deps)}个依赖"

        if has_native_trap and trap_dep:
            return f"""1. 项目认知：这是一个Node.js项目，包含{len(files)}个文件，使用npm start启动。

2. 依赖分析：项目依赖包括{deps_str}，package.json已定义所有依赖。

3. **关键发现**：检测到 {trap_dep}，这是一个需要 C++ 编译环境的 native 模块。
   - {trap_dep} 需要 node-gyp 编译
   - node-gyp 依赖：python3、make、g++
   - 基础镜像 node:slim 不包含这些编译工具，需要安装

4. 安装策略：先安装编译工具，再 npm ci 安装依赖。

5. Dockerfile结构：
   - FROM: node:20-slim
   - RUN: apt-get update && apt-get install -y python3 make g++
   - WORKDIR: /app
   - COPY: 先复制package.json和package-lock.json
   - RUN: npm ci
   - COPY: 复制其他文件
   - CMD: npm start"""
        elif variant == "multi-stage":
            build_dir = "dist" if "dist/" in files else "build"
            return f"""1. 项目认知：这是一个Node.js项目，包含{len(files)}个文件，有{build_dir}/构建产物目录。

2. 依赖分析：项目依赖包括{deps_str}，package.json已定义所有依赖。

3. 构建策略：检测到构建产物目录，使用多阶段构建。
   - 第一阶段：安装依赖并构建
   - 第二阶段：只复制构建产物，减小镜像体积

4. Dockerfile结构：
   - 第一阶段(builder): node:20，安装依赖并npm run build
   - 第二阶段(runtime): node:20-slim，只复制{build_dir}/目录
   - CMD: node {build_dir}/index.js"""
        else:
            return f"""1. 项目认知：这是一个Node.js项目，包含{len(files)}个文件，使用npm start启动。

2. 依赖分析：项目依赖包括{deps_str}，package.json已定义所有依赖。

3. 安装策略：使用 npm ci 安装依赖，比 npm install 更快更确定。

4. Dockerfile结构：
   - FROM: node:20-slim
   - WORKDIR: /app
   - COPY: 先复制package.json和package-lock.json
   - RUN: npm ci
   - COPY: 复制其他文件
   - CMD: npm start"""

    # ==================== Dockerfile ====================

    def _generate_dockerfile(self, env_state: dict) -> str:
        """生成Dockerfile"""
        language = env_state["language"]

        if language == "python":
            return self._generate_python_dockerfile(env_state)
        elif language == "go":
            return self._generate_go_dockerfile(env_state)
        else:  # javascript
            return self._generate_js_dockerfile(env_state)

    def _generate_python_dockerfile(self, env_state: dict) -> str:
        deps = env_state["mock_deps"]
        has_dep = env_state["has_dep_file"]
        entry_file = env_state.get("entry_file", "main.py")
        variant = env_state.get("variant", "reproducible")
        files = env_state["mock_files"]

        if variant == "web-serving":
            app_module = entry_file.replace('.py', '')
            if has_dep:
                return f"""FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "{app_module}:app", "--host", "0.0.0.0", "--port", "8000"]"""
            else:
                deps_str = " ".join(deps)
                return f"""FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir {deps_str}
COPY . .
EXPOSE 8000
CMD ["uvicorn", "{app_module}:app", "--host", "0.0.0.0", "--port", "8000"]"""
        elif variant == "editable-install":
            return f"""FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY . .
RUN pip install --no-cache-dir .
CMD ["python", "{entry_file}"]"""
        else:  # reproducible
            if has_dep:
                return f"""FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "{entry_file}"]"""
            else:
                deps_str = " ".join(deps)
                return f"""FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir {deps_str}
COPY . .
CMD ["python", "{entry_file}"]"""

    def _generate_go_dockerfile(self, env_state: dict) -> str:
        has_cgo_trap = env_state.get("has_cgo_trap", False)
        variant = env_state.get("variant", "static-scratch")

        if has_cgo_trap:
            # CGO需要gcc，使用debian runtime
            return """FROM golang:1.22 AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y gcc
ENV CGO_ENABLED=1
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o server .

FROM debian:bookworm-slim
COPY --from=builder /app/server /server
CMD ["/server"]"""
        else:
            # 静态构建，使用scratch镜像
            return """FROM golang:1.22-alpine AS builder
WORKDIR /app
ENV CGO_ENABLED=0
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -ldflags="-s -w" -o server .

FROM scratch
COPY --from=builder /app/server /server
CMD ["/server"]"""

    def _generate_js_dockerfile(self, env_state: dict) -> str:
        has_native_trap = env_state.get("has_native_trap", False)
        variant = env_state.get("variant", "simple")
        files = env_state.get("mock_files", [])

        if has_native_trap:
            return """FROM node:20-slim
RUN apt-get update && apt-get install -y python3 make g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
CMD ["npm", "start"]"""
        elif variant == "multi-stage":
            build_dir = "dist" if "dist/" in files else "build"
            return f"""FROM node:20 AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-slim
WORKDIR /app
COPY --from=builder /app/{build_dir} ./{build_dir}
COPY --from=builder /app/package.json ./
CMD ["node", "{build_dir}/index.js"]"""
        else:
            return """FROM node:20-slim
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
CMD ["npm", "start"]"""

    # ==================== Build Messages ====================

    def _build_messages(
        self,
        language: str,
        file_tree_result: str,
        deps_result: str,
        step1_cot: str,
        final_cot: str,
        dockerfile: str
    ) -> list:
        """构建消息序列"""
        lang_name = {"python": "Python", "go": "Go", "javascript": "Node.js"}[language]

        messages = [
            {
                "role": "system",
                "content": "你是一个DevOps助手，负责分析项目结构并生成Dockerfile。你有两个工具：ls_file_tree（查看文件结构）和ls_dependencies（获取依赖列表）。"
            },
            {
                "role": "user",
                "content": f"请为当前{lang_name}项目生成Dockerfile。"
            },
            {
                "role": "assistant",
                "content": "<tool_call>\nls_file_tree()\n</tool_call>"
            },
            {
                "role": "user",
                "content": f"<tool_result>\n{file_tree_result}\n</tool_result>"
            },
            {
                "role": "assistant",
                "content": f"<思考>\n{step1_cot}\n</思考>\n\n<tool_call>\nls_dependencies()\n</tool_call>"
            },
            {
                "role": "user",
                "content": f"<tool_result>\n{deps_result}\n</tool_result>"
            },
            {
                "role": "assistant",
                "content": f"<思考>\n{final_cot}\n</思考>\n\n<dockerfile>\n{dockerfile}\n</dockerfile>"
            }
        ]

        return messages

    def generate_dataset(self, num_samples: Optional[int] = None) -> list:
        """生成完整数据集"""
        n = num_samples or self.config.num_samples
        data = []

        print(f"Generating {n} SFT samples (Python/Go/JavaScript + refusal + critique)...")
        for _ in tqdm(range(n), desc="Generating"):
            sample = self.generate_single_sample()
            if sample:
                data.append(sample)

        # 统计各语言和场景数量
        lang_counts = {"python": 0, "go": 0, "javascript": 0, "refusal": 0}
        trap_counts = {"cgo": 0, "native": 0}
        critique_count = 0
        variant_counts = {"web-serving": 0, "editable-install": 0, "reproducible": 0,
                         "static-scratch": 0, "debian-runtime": 0,
                         "simple": 0, "multi-stage": 0}

        for sample in data:
            meta = sample["metadata"]
            lang = meta["language"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            if meta.get("has_cgo_trap"):
                trap_counts["cgo"] += 1
            if meta.get("has_native_trap"):
                trap_counts["native"] += 1
            if meta.get("is_critique"):
                critique_count += 1

            variant = meta.get("variant")
            if variant and variant in variant_counts:
                variant_counts[variant] += 1

        print(f"Successfully generated {len(data)}/{n} samples")
        print(f"  Languages: Python={lang_counts['python']}, Go={lang_counts['go']}, JS={lang_counts['javascript']}, Refusal={lang_counts['refusal']}")
        print(f"  Critique samples: {critique_count}")
        print(f"  Traps: CGO={trap_counts['cgo']}, Native={trap_counts['native']}")
        print(f"  Variants: {variant_counts}")
        return data

    def save_dataset(self, data: list, output_path: Optional[str] = None):
        """保存数据集到JSONL文件"""
        path = output_path or self.config.output_file
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dataset saved to {path}")


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="sft_train_data.jsonl",
                        help="Output file path")
    args = parser.parse_args()

    config = SFTConfig(
        num_samples=args.num_samples,
        output_file=args.output
    )

    generator = SFTDataGenerator(config)
    data = generator.generate_dataset()
    generator.save_dataset(data)


if __name__ == "__main__":
    main()
