"""
Mock Tool Definitions for SFT Data Generation

Tools:
1. ls_file_tree: 扫描项目文件结构
2. ls_top10_lines: 读取文件前10行内容
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
            "properties": {
                "path": {
                    "type": "string",
                    "description": "要扫描的目录路径，默认为项目根目录 '.'"
                }
            },
            "required": []
        }
    },
    {
        "name": "ls_top10_lines",
        "description": "读取指定文件的前10行内容。用于快速查看文件内容，特别是配置文件或入口文件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要读取的文件路径"
                }
            },
            "required": ["file_path"]
        }
    }
]


class MockEnvironment:
    """
    Mock环境：模拟项目结构和文件内容
    用于生成SFT训练数据
    """

    def __init__(self, config):
        self.config = config
        self.state = self._generate_state()

    def _generate_state(self) -> dict:
        """生成随机的项目状态"""
        # 随机选择依赖
        num_deps = random.randint(self.config.min_deps, self.config.max_deps)
        mock_deps = random.sample(self.config.common_deps, k=num_deps)

        # 随机选择文件
        num_files = random.randint(self.config.min_files, self.config.max_files)
        mock_files = random.sample(self.config.common_files, k=num_files)

        # 30% 概率包含 requirements.txt
        has_requirements_txt = random.random() < self.config.has_requirements_txt_prob
        if has_requirements_txt and "requirements.txt" not in mock_files:
            mock_files.append("requirements.txt")

        return {
            "mock_deps": mock_deps,
            "mock_files": mock_files,
            "has_requirements_txt": has_requirements_txt
        }

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """执行工具调用，返回模拟结果"""
        if tool_name == "ls_file_tree":
            return self._execute_ls_file_tree(tool_input)
        elif tool_name == "ls_top10_lines":
            return self._execute_ls_top10_lines(tool_input)
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def _execute_ls_file_tree(self, tool_input: dict) -> str:
        """模拟 ls_file_tree 工具"""
        files = self.state["mock_files"]
        result = "项目文件结构:\n"
        result += ".\n"
        for f in sorted(files):
            result += f"├── {f}\n"
        return result

    def _execute_ls_top10_lines(self, tool_input: dict) -> str:
        """模拟 ls_top10_lines 工具"""
        file_path = tool_input.get("file_path", "")

        if file_path == "requirements.txt":
            if self.state["has_requirements_txt"]:
                # 返回依赖列表
                deps = self.state["mock_deps"]
                lines = [f"{dep}>=0.1.0" for dep in deps]
                return "\n".join(lines[:10])
            else:
                return "Error: File 'requirements.txt' not found"

        elif file_path in ["app.py", "main.py", "server.py"]:
            # 模拟入口文件
            return self._generate_mock_python_file(file_path)

        elif file_path == "config.json":
            return '{\n  "name": "my-project",\n  "version": "1.0.0",\n  "description": "A Python project"\n}'

        elif file_path == "pyproject.toml":
            deps = self.state["mock_deps"]
            return f'[project]\nname = "my-project"\nversion = "1.0.0"\ndependencies = {deps[:3]}'

        else:
            return f"# {file_path}\n# Auto-generated file content"

    def _generate_mock_python_file(self, filename: str) -> str:
        """生成模拟的Python入口文件内容"""
        deps = self.state["mock_deps"]
        imports = "\n".join([f"import {dep}" if dep not in ["scikit-learn"] else f"import sklearn" for dep in deps[:3]])

        return f'''#!/usr/bin/env python3
"""
{filename} - Application entry point
"""
{imports}

def main():
    print("Starting application...")

if __name__ == "__main__":
    main()
'''

    def get_state(self) -> dict:
        """获取当前环境状态"""
        return self.state.copy()
