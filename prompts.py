"""
Prompt Templates for SFT Data Generation

这些prompt用于:
1. 让Claude生成高质量的CoT数据
2. 教会小模型如何分析项目结构并生成Dockerfile
"""

# System prompt for the teacher model (Claude)
SYSTEM_PROMPT = """你是一个专业的DevOps工程师，擅长分析Python项目结构并生成Dockerfile。

你的任务是：
1. 使用提供的工具扫描当前项目
2. 分析项目结构和依赖
3. 生成一个合适的Dockerfile

你有两个工具可用：
- ls_file_tree: 扫描项目文件结构
- ls_top10_lines: 读取文件前10行内容

【重要规则】
1. 你必须先调用 ls_file_tree 了解项目结构
2. 然后根据需要调用 ls_top10_lines 查看关键文件
3. 分析完成后，生成Dockerfile

【Dockerfile生成逻辑】
- 如果项目中存在 requirements.txt，必须使用:
  COPY requirements.txt .
  RUN pip install -r requirements.txt

- 如果项目中不存在 requirements.txt，需要根据代码分析出依赖，使用:
  RUN pip install dep1 dep2 dep3

请严格遵循这个逻辑！"""


# User prompt template
USER_PROMPT = """请分析当前Python项目，并生成一个合适的Dockerfile。

请按以下步骤进行：
1. 首先使用 ls_file_tree 工具查看项目结构
2. 根据需要使用 ls_top10_lines 查看关键文件内容
3. 分析项目的依赖和入口点
4. 生成完整的Dockerfile

开始分析。"""


# Final output format instruction
OUTPUT_FORMAT_PROMPT = """现在请根据你的分析，输出最终的Dockerfile。

请按以下格式输出：

<思考>
[这里写你的分析过程和决策理由]
</思考>

<dockerfile>
[这里写完整的Dockerfile内容]
</dockerfile>"""


# Template for converting to SFT training format
def format_sft_example(
    tool_calls: list,
    tool_results: list,
    thought: str,
    dockerfile: str,
    env_state: dict
) -> dict:
    """
    将完整的交互过程格式化为SFT训练数据

    Args:
        tool_calls: 工具调用列表
        tool_results: 工具返回结果列表
        thought: 模型的思考过程 (CoT)
        dockerfile: 最终生成的Dockerfile
        env_state: 环境状态（用于metadata）

    Returns:
        SFT训练格式的字典
    """
    # Build the conversation messages
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": USER_PROMPT
        }
    ]

    # Add tool calls and results as assistant turns
    for i, (call, result) in enumerate(zip(tool_calls, tool_results)):
        # Tool call from assistant
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [call]
        })
        # Tool result
        messages.append({
            "role": "tool",
            "tool_call_id": call.get("id", f"call_{i}"),
            "content": result
        })

    # Final assistant response with thought and dockerfile
    final_response = f"""<思考>
{thought}
</思考>

<dockerfile>
{dockerfile}
</dockerfile>"""

    messages.append({
        "role": "assistant",
        "content": final_response
    })

    return {
        "messages": messages,
        "metadata": {
            "has_requirements_txt": env_state.get("has_requirements_txt", False),
            "mock_deps": env_state.get("mock_deps", []),
            "mock_files": env_state.get("mock_files", [])
        }
    }
