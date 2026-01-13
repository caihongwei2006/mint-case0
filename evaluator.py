"""
Model Evaluator for SFT Fine-tuned Model

评估微调后模型生成Dockerfile的能力

Usage:
    python sft_pipeline/evaluator.py --checkpoint <checkpoint_path> --num-tests 50
    python -m sft_pipeline.evaluator --checkpoint <checkpoint_path> --num-tests 50
"""
import os
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 支持直接运行和作为模块运行
try:
    from .config import SFTConfig
    from .tools import MockEnvironment, TOOL_DEFINITIONS
except ImportError:
    from config import SFTConfig
    from tools import MockEnvironment, TOOL_DEFINITIONS


@dataclass
class EvalResult:
    """单个测试用例的评估结果"""
    test_id: int
    env_state: dict                    # 测试环境状态
    model_output: str                  # 模型原始输出
    tool_calls: list                   # 模型的工具调用
    dockerfile: Optional[str]          # 提取的Dockerfile

    # 核心评估指标（简化版）
    tool_call_valid: bool = False      # 工具调用格式正确（能 parse 出有效工具名）
    syntax_valid: bool = False         # Dockerfile语法正确

    # 详细信息
    tool_call_error: Optional[str] = None
    syntax_error: Optional[str] = None


@dataclass
class EvalSummary:
    """评估汇总（简化版）"""
    total_tests: int = 0
    tool_call_pass: int = 0            # 工具调用格式正确
    syntax_pass: int = 0               # Dockerfile 语法正确
    full_pass: int = 0                 # 所有指标都通过

    def __str__(self):
        return f"""
========================================
Evaluation Summary
========================================
Total Tests:     {self.total_tests}
----------------------------------------
Tool Call Pass:  {self.tool_call_pass}/{self.total_tests} ({self.tool_call_pass/max(self.total_tests,1)*100:.1f}%)
Syntax Pass:     {self.syntax_pass}/{self.total_tests} ({self.syntax_pass/max(self.total_tests,1)*100:.1f}%)
----------------------------------------
Full Pass:       {self.full_pass}/{self.total_tests} ({self.full_pass/max(self.total_tests,1)*100:.1f}%)
========================================
"""


class DockerfileSyntaxChecker:
    """Dockerfile 语法检查器"""

    # 有效的 Dockerfile 指令
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
                # 可能是多行命令的继续，跳过
                if not line.startswith(('-', '&&', '||', '|')):
                    # 检查是否是已知指令的变体
                    if not any(instruction.startswith(valid) for valid in cls.VALID_INSTRUCTIONS):
                        return False, f"Line {i}: Unknown instruction '{instruction}'"

        if not has_from:
            return False, "Missing FROM instruction"

        return True, None

    @classmethod
    def check_with_hadolint(cls, dockerfile: str) -> tuple[bool, Optional[str]]:
        """
        使用 hadolint 检查 Dockerfile（如果安装了的话）

        Returns:
            (is_valid, error_message)
        """
        try:
            result = subprocess.run(
                ['hadolint', '-'],
                input=dockerfile,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return True, None
            else:
                return False, result.stdout or result.stderr
        except FileNotFoundError:
            # hadolint 未安装，回退到基本检查
            return cls.check(dockerfile)
        except Exception as e:
            return cls.check(dockerfile)


class DockerfileLogicValidator:
    """Dockerfile 逻辑验证器"""

    @classmethod
    def validate(cls, dockerfile: str, env_state: dict) -> tuple[bool, Optional[str]]:
        """
        验证 Dockerfile 逻辑正确性

        Args:
            dockerfile: Dockerfile 内容
            env_state: 环境状态（包含 has_requirements_txt, mock_deps 等）

        Returns:
            (is_valid, error_message)
        """
        has_req = env_state.get("has_requirements_txt", False)
        deps = env_state.get("mock_deps", [])

        errors = []

        # 1. requirements.txt 处理
        if has_req:
            if "requirements.txt" not in dockerfile:
                errors.append("Should reference requirements.txt")
            if "pip install" in dockerfile and "-r" not in dockerfile:
                # 有 requirements.txt 但没用 -r 安装
                if "requirements.txt" not in dockerfile:
                    errors.append("Should use 'pip install -r requirements.txt'")
        else:
            # 没有 requirements.txt，检查是否直接安装了依赖
            if "COPY requirements.txt" in dockerfile:
                errors.append("Tried to COPY requirements.txt but it doesn't exist")

        # 2. 基础镜像检查
        if "FROM" not in dockerfile:
            errors.append("Missing FROM instruction")
        else:
            # 检查是否使用了 Python 镜像
            from_match = re.search(r'FROM\s+(\S+)', dockerfile, re.IGNORECASE)
            if from_match:
                image = from_match.group(1).lower()
                if 'python' not in image:
                    errors.append(f"Expected Python base image, got '{image}'")

        # 3. WORKDIR 检查（推荐）
        if "WORKDIR" not in dockerfile:
            # 这是一个警告，不是错误
            pass

        # 4. 依赖安装检查（如果没有 requirements.txt）
        if not has_req:
            # 应该显式安装一些依赖
            deps_found = sum(1 for dep in deps if dep.lower() in dockerfile.lower())
            if deps_found == 0 and deps:
                errors.append(f"No dependencies installed (expected: {deps[:3]}...)")

        if errors:
            return False, "; ".join(errors)

        return True, None


class ModelRunner:
    """模型运行器：执行微调后的模型"""

    # 有效的工具名
    VALID_TOOLS = {"ls_file_tree", "ls_dependencies"}

    def __init__(self, checkpoint_path: Optional[str] = None, base_model: str = "Qwen/Qwen3-0.6B"):
        self.checkpoint_path = checkpoint_path
        self.base_model = base_model
        self._sampling_client = None
        self._tokenizer = None

    def _init_client(self):
        """延迟初始化 MinT 客户端"""
        if self._sampling_client is not None:
            return

        import mint
        from mint import types
        from transformers import AutoTokenizer

        service_client = mint.ServiceClient()

        if self.checkpoint_path:
            # 加载指定的 checkpoint（使用 mint:// 路径）
            print(f"Loading checkpoint: {self.checkpoint_path}")
            self._sampling_client = service_client.create_sampling_client(
                model_path=self.checkpoint_path
            )
        else:
            # 使用基础模型
            print(f"Using base model: {self.base_model}")
            self._sampling_client = service_client.create_sampling_client(
                base_model=self.base_model
            )

        # 获取 tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        生成模型响应

        Args:
            messages: 对话消息列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            模型生成的文本
        """
        from mint import types

        self._init_client()

        # 使用 Qwen ChatML 格式构建 prompt
        prompt_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt_text += "<|im_start|>assistant\n"

        prompt_tokens = types.ModelInput.from_ints(
            self._tokenizer.encode(prompt_text, add_special_tokens=True)
        )

        # 使用 <|im_end|> 作为 stop token
        eos_id = self._tokenizer.eos_token_id
        im_end_id = self._tokenizer.encode("<|im_end|>", add_special_tokens=False)
        stop_tokens = []
        if eos_id is not None:
            stop_tokens.append(eos_id)
        if im_end_id:
            stop_tokens.extend(im_end_id)

        result = self._sampling_client.sample(
            prompt=prompt_tokens,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_tokens
            )
        ).result()

        return self._tokenizer.decode(result.sequences[0].tokens)


class ModelEvaluator:
    """模型评估器"""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        base_model: str = "Qwen/Qwen3-0.6B",
        config: Optional[SFTConfig] = None
    ):
        self.config = config or SFTConfig()
        self.model_runner = ModelRunner(checkpoint_path, base_model)
        self.results: list[EvalResult] = []

    def _parse_model_output(self, output: str) -> tuple[str, str, list]:
        """
        解析模型输出

        Returns:
            (thought, dockerfile, tool_calls)
        """
        thought = ""
        dockerfile = ""
        tool_calls = []

        # 提取思考
        thought_match = re.search(r"<思考>(.*?)</思考>", output, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取 Dockerfile
        dockerfile_match = re.search(r"<dockerfile>(.*?)</dockerfile>", output, re.DOTALL)
        if dockerfile_match:
            dockerfile = dockerfile_match.group(1).strip()

        # 提取工具调用
        tool_call_matches = re.findall(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
        for tc in tool_call_matches:
            tool_calls.append(tc.strip())

        return thought, dockerfile, tool_calls

    def _check_tool_calls(self, tool_calls: list) -> tuple[bool, Optional[str]]:
        """
        检查工具调用格式是否正确

        核心检查：
        1. 能否从输出中解析出 <tool_call> 标签
        2. 工具名是否是 ls_file_tree 或 ls_dependencies
        """
        if not tool_calls:
            return False, "No tool calls found"

        valid_tools = {"ls_file_tree", "ls_dependencies"}
        parsed_tools = []

        for tc in tool_calls:
            # 尝试解析工具调用格式: tool_name({...})
            tc_match = re.match(r"(\w+)\s*\(", tc.strip())
            if tc_match:
                tool_name = tc_match.group(1)
                if tool_name in valid_tools:
                    parsed_tools.append(tool_name)

        if not parsed_tools:
            return False, f"No valid tool found. Expected: {valid_tools}"

        return True, None

    def run_single_test(self, test_id: int) -> EvalResult:
        """运行单个测试用例"""
        # 1. 创建测试环境
        env = MockEnvironment(self.config)
        env_state = env.get_state()

        # 2. 构建初始对话
        messages = [
            {
                "role": "system",
                "content": "你是一个DevOps助手，负责分析项目结构并生成Dockerfile。"
            },
            {
                "role": "user",
                "content": "请为当前Python项目生成Dockerfile。\n\n可用工具：ls_file_tree, ls_top10_lines"
            }
        ]

        all_output = ""
        tool_calls_made = []
        max_turns = 10

        # 3. 多轮对话（处理工具调用）
        for turn in range(max_turns):
            response = self.model_runner.generate(messages, max_tokens=512)
            all_output += response

            # 检查是否有工具调用
            thought, dockerfile, tool_calls = self._parse_model_output(response)

            if tool_calls:
                # 处理工具调用
                for tc in tool_calls:
                    tool_calls_made.append(tc)

                    # 解析工具调用
                    tc_match = re.match(r"(\w+)\((.*)\)", tc.strip())
                    if tc_match:
                        tool_name = tc_match.group(1)
                        try:
                            tool_args = json.loads(tc_match.group(2) or "{}")
                        except:
                            tool_args = {}

                        # 执行工具
                        result = env.execute_tool(tool_name, tool_args)

                        # 添加到对话
                        messages.append({
                            "role": "assistant",
                            "content": f"<tool_call>\n{tc}\n</tool_call>"
                        })
                        messages.append({
                            "role": "user",
                            "content": f"<tool_result>\n{result}\n</tool_result>"
                        })

            # 如果生成了 Dockerfile，停止
            if dockerfile:
                break

        # 4. 解析最终输出
        print(f"\n{'='*60}")
        print(f"Test {test_id} - Model Output:")
        print(f"{'='*60}")
        print(all_output)
        print(f"{'='*60}\n")

        thought, dockerfile, _ = self._parse_model_output(all_output)

        # 5. 评估（简化版）
        result = EvalResult(
            test_id=test_id,
            env_state=env_state,
            model_output=all_output,
            tool_calls=tool_calls_made,
            dockerfile=dockerfile
        )

        # 工具调用检查
        result.tool_call_valid, result.tool_call_error = self._check_tool_calls(tool_calls_made)

        # 语法检查（只检查 Dockerfile 语法，不检查逻辑）
        if dockerfile:
            result.syntax_valid, result.syntax_error = DockerfileSyntaxChecker.check(dockerfile)

        return result

    def run_evaluation(self, num_tests: int = 50) -> EvalSummary:
        """运行完整评估"""
        from tqdm import tqdm
        import traceback

        print(f"Running {num_tests} evaluation tests...")

        # 先初始化客户端，确保连接成功
        print("Initializing MinT client...")
        try:
            self.model_runner._init_client()
            print("✓ MinT client initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize MinT client: {e}")
            traceback.print_exc()
            return EvalSummary(total_tests=num_tests)

        self.results = []
        summary = EvalSummary(total_tests=num_tests)

        for i in tqdm(range(num_tests), desc="Evaluating"):
            try:
                result = self.run_single_test(i)
                self.results.append(result)

                if result.tool_call_valid:
                    summary.tool_call_pass += 1
                if result.syntax_valid:
                    summary.syntax_pass += 1
                if result.tool_call_valid and result.syntax_valid:
                    summary.full_pass += 1

            except Exception as e:
                print(f"\nTest {i} failed: {e}")
                traceback.print_exc()
                continue

        return summary

    def save_results(self, output_path: str = "eval_results.jsonl"):
        """保存评估结果"""
        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.results:
                entry = {
                    "test_id": result.test_id,
                    "env_state": result.env_state,
                    "dockerfile": result.dockerfile,
                    "tool_calls": result.tool_calls,
                    "tool_call_valid": result.tool_call_valid,
                    "syntax_valid": result.syntax_valid,
                    "errors": {
                        "tool_call": result.tool_call_error,
                        "syntax": result.syntax_error,
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Results saved to {output_path}")


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint (tinker:// or mint:// URL)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model name")
    parser.add_argument("--num-tests", type=int, default=50,
                        help="Number of test cases")
    parser.add_argument("--output", type=str, default="eval_results.jsonl",
                        help="Output file for detailed results")
    args = parser.parse_args()

    # 检查 API key
    if not os.environ.get("MINT_API_KEY"):
        print("ERROR: MINT_API_KEY not found!")
        print("Please set it in .env file or environment variable")
        print("Looking for .env in:", Path.cwd())
        return

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint or 'None (using base model)'}")
    print(f"Num tests: {args.num_tests}")
    print(f"MINT_API_KEY: {'*' * 8}...{os.environ.get('MINT_API_KEY', '')[-4:]}")
    print()

    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        base_model=args.base_model
    )
    summary = evaluator.run_evaluation(num_tests=args.num_tests)

    print(summary)
    evaluator.save_results(args.output)


if __name__ == "__main__":
    main()
