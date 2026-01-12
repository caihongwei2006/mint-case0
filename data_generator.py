"""
SFT Data Generator

使用Claude API生成高质量的SFT训练数据
完整流程：
1. 创建Mock环境
2. 调用Claude，让其使用工具
3. 模拟工具返回结果
4. 收集完整的交互过程（包含CoT）
5. 保存为训练格式
"""
import json
import re
from openai import OpenAI
from tqdm import tqdm
from typing import Optional
from pathlib import Path

from .config import SFTConfig
from .tools import MockEnvironment, TOOL_DEFINITIONS
from .prompts import SYSTEM_PROMPT, USER_PROMPT, OUTPUT_FORMAT_PROMPT


class SFTDataGenerator:
    """SFT数据生成器"""

    def __init__(self, config: Optional[SFTConfig] = None):
        self.config = config or SFTConfig()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )

    def _convert_tools_to_openai_format(self, anthropic_tools: list) -> list:
        """将 Anthropic 工具格式转换为 OpenAI 格式"""
        openai_tools = []
        for tool in anthropic_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })
        return openai_tools

    def generate_single_sample(self) -> Optional[dict]:
        """
        生成单个SFT样本

        Returns:
            训练数据字典，如果失败返回None
        """
        # 1. 创建Mock环境
        env = MockEnvironment(self.config)
        env_state = env.get_state()

        # 2. 构建初始消息
        messages = [
            {"role": "user", "content": USER_PROMPT}
        ]

        tool_calls_record = []
        tool_results_record = []

        # 3. 与Claude交互，处理工具调用
        try:
            # 转换工具定义为 OpenAI 格式
            openai_tools = self._convert_tools_to_openai_format(TOOL_DEFINITIONS)
            
            # 第一次调用：让Claude开始分析
            response = self.client.chat.completions.create(
                model=self.config.teacher_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *messages
                ],
                tools=openai_tools
            )

            # 循环处理工具调用
            while response.choices[0].finish_reason == "tool_calls":
                assistant_message = response.choices[0].message
                tool_calls = assistant_message.tool_calls

                if not tool_calls:
                    break

                # 添加助手消息到历史
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [{
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls]
                })

                # 处理每个工具调用
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)

                    # 记录工具调用
                    tool_calls_record.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.function.arguments
                        }
                    })

                    # 执行工具
                    result = env.execute_tool(tool_name, tool_input)
                    tool_results_record.append(result)

                    # 添加工具结果
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                # 继续对话
                response = self.client.chat.completions.create(
                    model=self.config.teacher_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *messages
                    ],
                    tools=openai_tools
                )

            # 4. 提取最终响应
            final_text = response.choices[0].message.content or ""

            # 5. 解析思考和Dockerfile
            thought, dockerfile = self._parse_response(final_text)

            if not dockerfile:
                # 如果没有正确格式，请求重新输出
                messages.append({"role": "assistant", "content": final_text})
                messages.append({"role": "user", "content": OUTPUT_FORMAT_PROMPT})

                response = self.client.chat.completions.create(
                    model=self.config.teacher_model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *messages
                    ]
                )

                final_text = response.choices[0].message.content or ""

                thought, dockerfile = self._parse_response(final_text)

            if not dockerfile:
                return None

            # 6. 验证Dockerfile逻辑正确性
            is_valid, validation_msg = self._validate_dockerfile(dockerfile, env_state)
            if not is_valid:
                print(f"Warning: Invalid dockerfile - {validation_msg}")
                # 仍然保存，但标记为可能有问题
                env_state["validation_warning"] = validation_msg

            # 7. 构建SFT训练数据
            return self._build_sft_data(
                tool_calls_record,
                tool_results_record,
                thought,
                dockerfile,
                env_state
            )

        except Exception as e:
            print(f"Error generating sample: {e}")
            return None

    def _parse_response(self, text: str) -> tuple[str, str]:
        """解析Claude的响应，提取思考和Dockerfile"""
        thought = ""
        dockerfile = ""

        # 提取思考部分
        thought_match = re.search(r"<思考>(.*?)</思考>", text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取Dockerfile部分
        dockerfile_match = re.search(r"<dockerfile>(.*?)</dockerfile>", text, re.DOTALL)
        if dockerfile_match:
            dockerfile = dockerfile_match.group(1).strip()
        else:
            # 尝试匹配```dockerfile代码块
            dockerfile_match = re.search(r"```dockerfile\n(.*?)```", text, re.DOTALL)
            if dockerfile_match:
                dockerfile = dockerfile_match.group(1).strip()

        return thought, dockerfile

    def _validate_dockerfile(self, dockerfile: str, env_state: dict) -> tuple[bool, str]:
        """验证Dockerfile的逻辑正确性"""
        has_req = env_state.get("has_requirements_txt", False)
        deps = env_state.get("mock_deps", [])

        if has_req:
            # 应该使用requirements.txt
            if "requirements.txt" not in dockerfile:
                return False, "Should use requirements.txt but didn't"
            if "pip install -r" not in dockerfile:
                return False, "Should use 'pip install -r requirements.txt'"
        else:
            # 不应该使用requirements.txt
            if "requirements.txt" in dockerfile and "COPY requirements.txt" in dockerfile:
                return False, "Used requirements.txt but it doesn't exist"
            # 应该显式安装依赖
            # 检查是否至少包含了一些依赖
            deps_found = sum(1 for dep in deps if dep in dockerfile)
            if deps_found < len(deps) // 2:
                return False, f"Missing dependencies: found {deps_found}/{len(deps)}"

        # 检查基本结构
        if "FROM" not in dockerfile:
            return False, "Missing FROM instruction"

        return True, "OK"

    def _build_sft_data(
        self,
        tool_calls: list,
        tool_results: list,
        thought: str,
        dockerfile: str,
        env_state: dict
    ) -> dict:
        """构建SFT训练数据格式"""
        # 简化的消息格式，适合小模型训练
        messages = [
            {
                "role": "system",
                "content": "你是一个DevOps助手，负责分析项目结构并生成Dockerfile。"
            },
            {
                "role": "user",
                "content": f"请为当前Python项目生成Dockerfile。\n\n可用工具：ls_file_tree, ls_top10_lines"
            }
        ]

        # 添加工具调用和结果
        for i, (call, result) in enumerate(zip(tool_calls, tool_results)):
            func_info = call.get("function", {})
            tool_name = func_info.get("name", "unknown")
            tool_args = func_info.get("arguments", "{}")

            # 工具调用
            messages.append({
                "role": "assistant",
                "content": f"<tool_call>\n{tool_name}({tool_args})\n</tool_call>"
            })

            # 工具结果
            messages.append({
                "role": "user",
                "content": f"<tool_result>\n{result}\n</tool_result>"
            })

        # 最终输出（包含CoT）
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
                "mock_files": env_state.get("mock_files", []),
                "validation_warning": env_state.get("validation_warning")
            }
        }

    def generate_dataset(self, num_samples: Optional[int] = None) -> list:
        """
        生成完整数据集

        Args:
            num_samples: 生成样本数量，默认使用配置值

        Returns:
            训练数据列表
        """
        n = num_samples or self.config.num_samples
        data = []

        print(f"Generating {n} SFT samples...")
        for _ in tqdm(range(n), desc="Generating"):
            sample = self.generate_single_sample()
            if sample:
                data.append(sample)

        print(f"Successfully generated {len(data)}/{n} samples")
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
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to generate (default: 10 for testing)")
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
