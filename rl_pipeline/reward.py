"""
奖励函数模块

奖励设计：
- 每次正确调用工具 +0.07
- 工具调用次数 > n+1（n=关键文件数）→ -0.5
- dockerfile 为空 → -0.5
- dockerfile 格式正确 → +1.0
- 其余情况 → 0
"""
import re
from typing import Optional


class DockerfileSyntaxChecker:
    """Dockerfile 语法检查器"""

    VALID_INSTRUCTIONS = {
        'FROM', 'RUN', 'CMD', 'LABEL', 'EXPOSE', 'ENV', 'ADD', 'COPY',
        'ENTRYPOINT', 'VOLUME', 'USER', 'WORKDIR', 'ARG', 'ONBUILD',
        'STOPSIGNAL', 'HEALTHCHECK', 'SHELL'
    }

    @classmethod
    def check(cls, dockerfile: str) -> tuple[bool, Optional[str]]:
        """
        检查 Dockerfile 语法

        Returns:
            (is_valid, error_message)
        """
        if not dockerfile or not dockerfile.strip():
            return False, "Empty Dockerfile"

        lines = dockerfile.strip().split('\n')
        has_from = False

        for i, line in enumerate(lines, 1):
            line = line.strip()

            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue

            # 处理续行符
            while line.endswith('\\') and i < len(lines):
                i += 1
                line = line[:-1] + lines[i-1].strip()

            # 提取指令
            parts = line.split(None, 1)
            if not parts:
                continue

            instruction = parts[0].upper()

            # 检查 FROM 指令
            if instruction == 'FROM':
                has_from = True
                if len(parts) < 2:
                    return False, f"Line {i}: FROM requires an image name"

            # 检查指令是否有效
            elif instruction not in cls.VALID_INSTRUCTIONS:
                if not line.startswith(('-', '&&', '||', '|')):
                    if not any(instruction.startswith(valid) for valid in cls.VALID_INSTRUCTIONS):
                        return False, f"Line {i}: Unknown instruction '{instruction}'"

        if not has_from:
            return False, "Missing FROM instruction"

        return True, None


def parse_tool_calls(output: str) -> list[dict]:
    """
    从模型输出中解析工具调用

    格式: <tool_call>tool_name({"arg": "value"})</tool_call>

    Returns:
        list of {"name": str, "args": dict, "valid": bool}
    """
    tool_calls = []
    valid_tools = {"ls_file_tree", "ls_top10_lines"}

    # 提取所有 <tool_call> 标签内容
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)

    for tc in matches:
        tc = tc.strip()
        # 解析格式: tool_name({...})
        match = re.match(r"(\w+)\s*\((.*)\)", tc, re.DOTALL)
        if match:
            tool_name = match.group(1)
            try:
                import json
                args_str = match.group(2).strip()
                args = json.loads(args_str) if args_str else {}
            except:
                args = {}

            tool_calls.append({
                "name": tool_name,
                "args": args,
                "valid": tool_name in valid_tools
            })
        else:
            tool_calls.append({
                "name": tc,
                "args": {},
                "valid": False
            })

    return tool_calls


def parse_dockerfile(output: str) -> str:
    """
    从模型输出中提取 Dockerfile

    格式: <dockerfile>...</dockerfile>
    """
    match = re.search(r"<dockerfile>(.*?)</dockerfile>", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def count_key_files(file_tree: str) -> int:
    """
    从文件树中统计关键文件数量

    关键文件：需要用 ls_top10_lines 查看的文件
    - Python 入口文件: app.py, main.py, server.py
    - 配置文件: requirements.txt, pyproject.toml, config.json
    """
    key_patterns = [
        "app.py", "main.py", "server.py",
        "requirements.txt", "pyproject.toml", "config.json",
        "setup.py", "Makefile"
    ]

    count = 0
    for pattern in key_patterns:
        if pattern in file_tree:
            count += 1

    return count


def compute_reward(
    model_output: str,
    file_tree: str,
    verbose: bool = False
) -> tuple[float, dict]:
    """
    计算奖励

    Args:
        model_output: 模型的完整输出（包含所有轮次）
        file_tree: ls_file_tree 返回的文件树
        verbose: 是否返回详细信息

    Returns:
        (reward, info_dict)

    奖励设计：
    - 每次正确调用工具 +0.07
    - 工具调用次数 > n+1（n=关键文件数）→ -0.5
    - dockerfile 为空 → -0.5
    - dockerfile 格式正确 → +1.0
    - 其余情况 → 0
    """
    reward = 0.0
    info = {
        "tool_calls": [],
        "num_valid_calls": 0,
        "num_key_files": 0,
        "expected_calls": 0,
        "dockerfile": "",
        "dockerfile_valid": False,
        "penalties": [],
        "rewards": []
    }

    # 1. 解析工具调用
    tool_calls = parse_tool_calls(model_output)
    info["tool_calls"] = tool_calls

    # 2. 计算关键文件数
    num_key_files = count_key_files(file_tree)
    expected_calls = num_key_files + 1  # ls_file_tree + n * ls_top10_lines
    info["num_key_files"] = num_key_files
    info["expected_calls"] = expected_calls

    # 3. 每次正确调用工具 +0.07
    for tc in tool_calls:
        if tc["valid"]:
            reward += 0.07
            info["num_valid_calls"] += 1
            info["rewards"].append(f"+0.07 for valid tool call: {tc['name']}")

    # 4. 工具调用次数 > n+1 → -0.5
    if len(tool_calls) > expected_calls:
        reward -= 0.5
        info["penalties"].append(f"-0.5 for too many tool calls: {len(tool_calls)} > {expected_calls}")

    # 5. 解析 Dockerfile
    dockerfile = parse_dockerfile(model_output)
    info["dockerfile"] = dockerfile

    # 6. Dockerfile 检查
    if not dockerfile or not dockerfile.strip():
        reward -= 0.5
        info["penalties"].append("-0.5 for empty dockerfile")
    else:
        is_valid, error = DockerfileSyntaxChecker.check(dockerfile)
        info["dockerfile_valid"] = is_valid
        if is_valid:
            reward += 1.0
            info["rewards"].append("+1.0 for valid dockerfile syntax")
        # 其余情况 reward += 0（不加不减）

    if verbose:
        print(f"Reward breakdown:")
        print(f"  Tool calls: {len(tool_calls)} (valid: {info['num_valid_calls']})")
        print(f"  Expected calls: {expected_calls} (key files: {num_key_files})")
        print(f"  Dockerfile valid: {info['dockerfile_valid']}")
        print(f"  Rewards: {info['rewards']}")
        print(f"  Penalties: {info['penalties']}")
        print(f"  Total reward: {reward}")

    return reward, info


if __name__ == "__main__":
    # 测试奖励函数
    test_output = """
<tool_call>
ls_file_tree({"path": "."})
</tool_call>

<tool_call>
ls_top10_lines({"file_path": "requirements.txt"})
</tool_call>

<tool_call>
ls_top10_lines({"file_path": "app.py"})
</tool_call>

<思考>
这是一个 Python 项目，有 requirements.txt 和 app.py。
</思考>

<dockerfile>
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
</dockerfile>
"""

    test_file_tree = """
项目文件结构:
.
├── app.py
├── requirements.txt
├── config.json
"""

    reward, info = compute_reward(test_output, test_file_tree, verbose=True)
    print(f"\nFinal reward: {reward}")
