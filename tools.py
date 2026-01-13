"""
Mock Tool Definitions for SFT Data Generation

Tools:
1. ls_file_tree: 扫描项目文件结构
2. ls_dependencies: 获取项目依赖列表

支持三种语言：Python, Go, JavaScript
"""
import random
from typing import Any
from dataclasses import dataclass


@dataclass
class ToolDefinition:
    """Tool schema for Claude API"""
    name: str
    description: str
    input_schema: dict


# Tool definitions for Claude API
TOOL_DEFINITIONS = [
    {
        "name": "ls_file_tree",
        "description": "扫描当前项目的文件结构，返回项目中的文件和目录列表。用于了解项目的整体结构。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "ls_dependencies",
        "description": "获取当前项目的依赖列表。自动分析项目中的依赖信息。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


class MockEnvironment:
    """
    Mock环境：模拟项目结构和文件内容
    用于生成SFT训练数据
    支持 Python, Go, JavaScript 三种语言
    """

    # 语言列表，等概率选择
    LANGUAGES = ["python", "go", "javascript"]

    def __init__(self, config):
        self.config = config
        self.state = self._generate_state()

    def _generate_state(self) -> dict:
        """生成随机的项目状态"""
        # 10% 概率生成 refusal 场景（多语言混合项目）
        if random.random() < self.config.refusal_probability:
            return self._generate_refusal_state()

        # 等概率选择语言
        language = random.choice(self.LANGUAGES)

        if language == "python":
            return self._generate_python_state()
        elif language == "go":
            return self._generate_go_state()
        else:  # javascript
            return self._generate_js_state()

    def _generate_refusal_state(self) -> dict:
        """生成多语言混合项目状态（用于refusal场景）"""
        # 从每个语言的文件池中随机选择0-2个文件
        py_files = random.sample(
            [f for f in self.config.python_files if f.endswith('.py')],
            k=random.randint(0, 2)
        )
        go_files = random.sample(
            [f for f in self.config.go_files if f.endswith('.go')],
            k=random.randint(0, 2)
        )
        js_files = random.sample(
            [f for f in self.config.js_files if f.endswith('.js')],
            k=random.randint(0, 2)
        )

        mock_files = py_files + go_files + js_files

        # 确保总文件数>=3
        while len(mock_files) < 3:
            # 随机从某个语言池补充
            lang = random.choice(['python', 'go', 'javascript'])
            if lang == 'python':
                pool = [f for f in self.config.python_files if f.endswith('.py') and f not in mock_files]
            elif lang == 'go':
                pool = [f for f in self.config.go_files if f.endswith('.go') and f not in mock_files]
            else:
                pool = [f for f in self.config.js_files if f.endswith('.js') and f not in mock_files]
            if pool:
                mock_files.append(random.choice(pool))

        # 添加一些通用配置文件
        common_files = ["README.md", ".gitignore"]
        mock_files.extend(random.sample(common_files, k=random.randint(0, 2)))

        random.shuffle(mock_files)

        return {
            "language": "refusal",  # 特殊标记
            "mock_deps": [],
            "mock_files": mock_files,
            "has_dep_file": False,
            "entry_file": None,
            "is_refusal": True
        }

    def _generate_python_state(self) -> dict:
        """生成 Python 项目状态"""
        # 随机选择依赖
        num_deps = random.randint(self.config.min_deps, self.config.max_deps)
        mock_deps = random.sample(self.config.python_deps, k=num_deps)

        # 随机选择文件
        num_files = random.randint(self.config.min_files, self.config.max_files)
        mock_files = random.sample(self.config.python_files, k=num_files)

        # 判断变体类型（优先级：web-serving > editable-install > reproducible）
        # 检查是否有web框架依赖
        has_web_deps = any(dep in mock_deps for dep in self.config.python_web_deps)
        # 检查是否有setup.py或pyproject.toml
        has_setup = "setup.py" in mock_files or "pyproject.toml" in mock_files
        # 50% 概率包含 requirements.txt
        has_requirements_txt = random.random() < 0.5

        # 决定变体
        if has_web_deps or random.random() < 0.3:  # 30%概率强制web-serving
            variant = "web-serving"
            # 确保有web依赖
            if not has_web_deps:
                mock_deps.extend(["fastapi", "uvicorn"])
            elif "fastapi" in mock_deps and "uvicorn" not in mock_deps:
                mock_deps.append("uvicorn")
        elif has_setup or random.random() < 0.3:  # 30%概率强制editable-install
            variant = "editable-install"
            # 确保有setup.py或pyproject.toml
            if not has_setup:
                mock_files.append(random.choice(["setup.py", "pyproject.toml"]))
        else:
            variant = "reproducible"

        if has_requirements_txt and "requirements.txt" not in mock_files:
            mock_files.append("requirements.txt")

        # 确保至少有一个入口文件
        entry_candidates = ["main.py", "app.py", "server.py"]
        has_entry = any(f in mock_files for f in entry_candidates)
        if not has_entry:
            entry_to_add = "app.py" if variant == "web-serving" else random.choice(entry_candidates)
            mock_files.append(entry_to_add)

        # 按优先级排序
        mock_files = self._sort_python_files(mock_files)
        entry_file = next((f for f in mock_files if f.endswith(".py")), "main.py")

        return {
            "language": "python",
            "mock_deps": mock_deps,
            "mock_files": mock_files,
            "has_dep_file": has_requirements_txt,
            "entry_file": entry_file,
            "variant": variant,  # 新增：变体类型
            "has_setup": "setup.py" in mock_files or "pyproject.toml" in mock_files
        }

    def _generate_go_state(self) -> dict:
        """生成 Go 项目状态"""
        # 随机选择依赖
        num_deps = random.randint(self.config.min_deps, self.config.max_deps)
        mock_deps = random.sample(self.config.go_deps, k=num_deps)

        # 30% 概率引入 CGO 陷阱依赖
        has_cgo_trap = random.random() < self.config.trap_probability
        trap_dep = None
        if has_cgo_trap and self.config.go_cgo_deps:
            trap_dep = random.choice(self.config.go_cgo_deps)
            mock_deps.append(trap_dep)

        # 决定变体：有CGO用debian，无CGO用static+scratch
        variant = "debian-runtime" if has_cgo_trap else "static-scratch"

        # 随机选择文件
        num_files = random.randint(self.config.min_files, self.config.max_files)
        mock_files = random.sample(self.config.go_files, k=num_files)

        # Go 必须有 go.mod 和 go.sum
        if "go.mod" not in mock_files:
            mock_files.append("go.mod")
        if "go.sum" not in mock_files:
            mock_files.append("go.sum")

        # 确保有 main.go
        if "main.go" not in mock_files:
            mock_files.append("main.go")

        # 按优先级排序
        mock_files = self._sort_go_files(mock_files)

        return {
            "language": "go",
            "mock_deps": mock_deps,
            "mock_files": mock_files,
            "has_dep_file": True,  # Go 总是有 go.mod
            "entry_file": "main.go",
            "has_cgo_trap": has_cgo_trap,
            "trap_dep": trap_dep,
            "variant": variant  # 新增：变体类型
        }

    def _generate_js_state(self) -> dict:
        """生成 JavaScript 项目状态"""
        # 随机选择依赖
        num_deps = random.randint(self.config.min_deps, self.config.max_deps)
        mock_deps = random.sample(self.config.js_deps, k=num_deps)

        # 30% 概率引入 native 模块陷阱依赖
        has_native_trap = random.random() < self.config.trap_probability
        trap_dep = None
        if has_native_trap and self.config.js_native_deps:
            trap_dep = random.choice(self.config.js_native_deps)
            mock_deps.append(trap_dep)

        # 随机选择文件
        num_files = random.randint(self.config.min_files, self.config.max_files)
        mock_files = random.sample(self.config.js_files, k=num_files)

        # 判断变体：有dist/或build/目录用多阶段构建
        has_build_dir = "dist/" in mock_files or "build/" in mock_files
        if not has_build_dir and random.random() < 0.3:  # 30%概率强制多阶段
            mock_files.append(random.choice(["dist/", "build/"]))
            has_build_dir = True

        variant = "multi-stage" if has_build_dir else "simple"

        # JS 必须有 package.json 和 package-lock.json
        if "package.json" not in mock_files:
            mock_files.append("package.json")
        if "package-lock.json" not in mock_files:
            mock_files.append("package-lock.json")

        # 按优先级排序
        mock_files = self._sort_js_files(mock_files)

        return {
            "language": "javascript",
            "mock_deps": mock_deps,
            "mock_files": mock_files,
            "has_dep_file": True,  # JS 总是有 package.json
            "entry_file": None,  # JS 用 npm start，不需要入口文件
            "has_native_trap": has_native_trap,
            "trap_dep": trap_dep,
            "variant": variant,  # 新增：变体类型
            "has_build_dir": has_build_dir
        }

    def _sort_python_files(self, files: list) -> list:
        """按优先级排序 Python 文件"""
        priority = {
            "main.py": 0, "app.py": 1, "server.py": 2,
            "requirements.txt": 10,
        }
        def get_priority(f):
            if f in priority:
                return priority[f]
            elif f.endswith(".py"):
                return 5
            else:
                return 20
        return sorted(files, key=get_priority)

    def _sort_go_files(self, files: list) -> list:
        """按优先级排序 Go 文件"""
        priority = {
            "main.go": 0,
            "go.mod": 10, "go.sum": 11,
        }
        def get_priority(f):
            if f in priority:
                return priority[f]
            elif f.endswith(".go"):
                return 5
            else:
                return 20
        return sorted(files, key=get_priority)

    def _sort_js_files(self, files: list) -> list:
        """按优先级排序 JS 文件"""
        priority = {
            "index.js": 0, "app.js": 1, "server.js": 2, "main.js": 3,
            "package.json": 10, "package-lock.json": 11,
        }
        def get_priority(f):
            if f in priority:
                return priority[f]
            elif f.endswith(".js"):
                return 5
            else:
                return 20
        return sorted(files, key=get_priority)

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """执行工具调用，返回模拟结果"""
        if tool_name == "ls_file_tree":
            return self._execute_ls_file_tree(tool_input)
        elif tool_name == "ls_dependencies":
            return self._execute_ls_dependencies(tool_input)
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def _execute_ls_file_tree(self, tool_input: dict) -> str:
        """模拟 ls_file_tree 工具"""
        files = self.state["mock_files"]
        language = self.state["language"]

        if language == "refusal":
            # 多语言混合项目
            result = "项目文件结构:\n"
            result += ".\n"
            for f in files:
                result += f"├── {f}\n"
            return result
        else:
            lang_name = {"python": "Python", "go": "Go", "javascript": "Node.js"}[language]
            result = f"项目文件结构（{lang_name}项目）:\n"
            result += ".\n"
            for f in files:
                result += f"├── {f}\n"
            return result

    def _execute_ls_dependencies(self, tool_input: dict) -> str:
        """模拟 ls_dependencies 工具"""
        language = self.state["language"]
        deps = self.state["mock_deps"]

        # 50% 概率使用详细格式，50% 概率使用简单格式
        use_detailed_format = random.random() < 0.5

        if language == "python":
            return self._format_python_deps(deps, use_detailed_format)
        elif language == "go":
            return self._format_go_deps(deps, use_detailed_format)
        else:  # javascript
            return self._format_js_deps(deps, use_detailed_format)

    def _format_python_deps(self, deps: list, detailed: bool) -> str:
        """格式化 Python 依赖"""
        if self.state["has_dep_file"]:
            # requirements.txt 格式
            result = "requirements.txt:\n"
            for dep in deps:
                version = f"0.{random.randint(1,9)}.{random.randint(0,9)}"
                result += f"{dep}>={version}\n"
        else:
            # 从代码分析出的依赖（import 语句格式）
            result = "从代码分析出的 import 语句:\n"
            for dep in deps:
                result += f"import {dep}\n"
        return result.strip()

    def _format_go_deps(self, deps: list, detailed: bool) -> str:
        """格式化 Go 依赖"""
        # go.mod 格式
        result = "go.mod:\n"
        result += "module myproject\n\n"
        result += "go 1.22\n\n"
        result += "require (\n"
        for dep in deps:
            version = f"v1.{random.randint(0,9)}.{random.randint(0,9)}"
            result += f"    {dep} {version}\n"
        result += ")"
        return result

    def _format_js_deps(self, deps: list, detailed: bool) -> str:
        """格式化 JS 依赖"""
        # package.json dependencies 格式
        result = "package.json dependencies:\n{\n"
        dep_lines = []
        for dep in deps:
            version = f"^{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,9)}"
            dep_lines.append(f'  "{dep}": "{version}"')
        result += ",\n".join(dep_lines)
        result += "\n}"
        return result

    def get_state(self) -> dict:
        """获取当前环境状态"""
        return self.state.copy()
