"""
RL 环境模块

定义 Dockerfile 生成的 RL 环境
"""
import re
import json
import random
import sys
from pathlib import Path
from typing import Sequence, Optional
from dataclasses import dataclass

# 添加父目录到 path 以便导入 sft_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sft_pipeline.config import SFTConfig
    from sft_pipeline.tools import MockEnvironment, TOOL_DEFINITIONS
except ImportError:
    from ..sft_pipeline.config import SFTConfig
    from ..sft_pipeline.tools import MockEnvironment, TOOL_DEFINITIONS

from .reward import compute_reward, parse_tool_calls, parse_dockerfile


@dataclass
class Observation:
    """观察（prompt tokens）"""
    tokens: list[int]
    text: str  # 用于调试


@dataclass
class Action:
    """动作（模型生成的 tokens）"""
    tokens: list[int]
    text: str
    logprobs: list[float]


@dataclass
class StepResult:
    """单步结果"""
    observation: Observation
    reward: float
    done: bool
    info: dict


class DockerfileEnv:
    """
    Dockerfile 生成的 RL 环境

    多轮对话流程：
    1. 初始 prompt → 模型输出工具调用
    2. 工具结果 → 模型继续输出（可能继续调用工具或输出 dockerfile）
    3. 重复直到输出 dockerfile 或达到最大轮次
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[SFTConfig] = None,
        max_turns: int = 10,
        max_tokens_per_turn: int = 512
    ):
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()
        self.max_turns = max_turns
        self.max_tokens_per_turn = max_tokens_per_turn

        # 环境状态
        self.mock_env = None
        self.conversation_history = []
        self.all_outputs = ""
        self.file_tree = ""
        self.current_turn = 0
        self.done = False

    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        return """你是一个DevOps助手，负责分析项目结构并生成Dockerfile。

可用工具：
1. ls_file_tree - 扫描项目文件结构
2. ls_top10_lines - 读取文件前10行

工具调用格式：
<tool_call>
tool_name({"arg": "value"})
</tool_call>

最终输出格式：
<思考>你的分析过程</思考>
<dockerfile>
FROM ...
...
</dockerfile>"""

    def _build_user_prompt(self) -> str:
        """构建用户提示"""
        return "请为当前Python项目生成Dockerfile。首先使用工具了解项目结构。"

    def _format_messages_to_text(self, messages: list[dict]) -> str:
        """将消息列表转换为文本格式"""
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|{role}|>\n{content}\n"
        return text

    async def initial_observation(self) -> tuple[Observation, list]:
        """
        返回初始观察

        Returns:
            (observation, stop_conditions)
        """
        # 创建新的 mock 环境
        self.mock_env = MockEnvironment(self.config)
        self.conversation_history = []
        self.all_outputs = ""
        self.file_tree = ""
        self.current_turn = 0
        self.done = False

        # 构建初始消息
        self.conversation_history = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt()}
        ]

        # 转换为 prompt 文本
        prompt_text = self._format_messages_to_text(self.conversation_history)
        prompt_text += "<|assistant|>\n"

        # Tokenize
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)

        observation = Observation(tokens=tokens, text=prompt_text)

        # Stop conditions: EOS token 或特定标签
        stop_conditions = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []

        return observation, stop_conditions

    async def step(self, action: Action) -> StepResult:
        """
        执行一步

        Args:
            action: 模型生成的动作

        Returns:
            StepResult
        """
        self.current_turn += 1
        self.all_outputs += action.text

        # 解析模型输出
        tool_calls = parse_tool_calls(action.text)
        dockerfile = parse_dockerfile(action.text)

        # 如果生成了 dockerfile，结束
        if dockerfile:
            self.done = True
            reward, info = compute_reward(self.all_outputs, self.file_tree)
            return StepResult(
                observation=Observation(tokens=[], text=""),
                reward=reward,
                done=True,
                info=info
            )

        # 如果有工具调用，执行工具
        if tool_calls:
            tool_results = []
            for tc in tool_calls:
                if tc["valid"]:
                    result = self.mock_env.execute_tool(tc["name"], tc["args"])
                    tool_results.append(f"<tool_result>\n{result}\n</tool_result>")

                    # 保存 file_tree 结果
                    if tc["name"] == "ls_file_tree":
                        self.file_tree = result
                else:
                    tool_results.append(f"<tool_result>\nError: Unknown tool '{tc['name']}'\n</tool_result>")

            # 添加到对话历史
            self.conversation_history.append({
                "role": "assistant",
                "content": action.text
            })
            self.conversation_history.append({
                "role": "user",
                "content": "\n".join(tool_results)
            })

            # 构建新的 observation
            prompt_text = self._format_messages_to_text(self.conversation_history)
            prompt_text += "<|assistant|>\n"
            tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)

            return StepResult(
                observation=Observation(tokens=tokens, text=prompt_text),
                reward=0.0,  # 中间步骤不给奖励
                done=False,
                info={"tool_calls": tool_calls}
            )

        # 达到最大轮次
        if self.current_turn >= self.max_turns:
            self.done = True
            reward, info = compute_reward(self.all_outputs, self.file_tree)
            return StepResult(
                observation=Observation(tokens=[], text=""),
                reward=reward,
                done=True,
                info=info
            )

        # 没有工具调用也没有 dockerfile，继续
        self.conversation_history.append({
            "role": "assistant",
            "content": action.text
        })

        prompt_text = self._format_messages_to_text(self.conversation_history)
        prompt_text += "<|assistant|>\n"
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)

        return StepResult(
            observation=Observation(tokens=tokens, text=prompt_text),
            reward=0.0,
            done=False,
            info={}
        )

    def get_state(self) -> dict:
        """获取环境状态"""
        return {
            "mock_env_state": self.mock_env.get_state() if self.mock_env else None,
            "current_turn": self.current_turn,
            "done": self.done,
            "file_tree": self.file_tree,
            "all_outputs": self.all_outputs
        }


class DockerfileEnvGroupBuilder:
    """
    环境组构建器

    用于创建一组环境（用于 GRPO 风格的 group-based rollouts）
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[SFTConfig] = None,
        group_size: int = 4,
        max_turns: int = 10
    ):
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()
        self.group_size = group_size
        self.max_turns = max_turns

    async def make_envs(self) -> Sequence[DockerfileEnv]:
        """创建一组环境"""
        return [
            DockerfileEnv(
                tokenizer=self.tokenizer,
                config=self.config,
                max_turns=self.max_turns
            )
            for _ in range(self.group_size)
        ]


class DockerfileRLDataset:
    """
    RL 数据集

    提供 EnvGroupBuilder 的批次
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[SFTConfig] = None,
        group_size: int = 4,
        max_turns: int = 10,
        num_batches: int = 1000
    ):
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()
        self.group_size = group_size
        self.max_turns = max_turns
        self.num_batches = num_batches

    def get_batch(self, index: int) -> list[DockerfileEnvGroupBuilder]:
        """获取一个批次的 EnvGroupBuilder"""
        # 每个批次返回一个 EnvGroupBuilder
        return [
            DockerfileEnvGroupBuilder(
                tokenizer=self.tokenizer,
                config=self.config,
                group_size=self.group_size,
                max_turns=self.max_turns
            )
        ]

    def __len__(self):
        return self.num_batches


if __name__ == "__main__":
    # 测试环境
    import asyncio
    from transformers import AutoTokenizer

    async def test_env():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        env = DockerfileEnv(tokenizer=tokenizer)

        obs, stop = await env.initial_observation()
        print("Initial observation:")
        print(f"  Tokens: {len(obs.tokens)}")
        print(f"  Text preview: {obs.text[:200]}...")
        print(f"  Stop conditions: {stop}")

        # 模拟一个动作
        mock_action = Action(
            tokens=[],
            text='<tool_call>\nls_file_tree({"path": "."})\n</tool_call>',
            logprobs=[]
        )

        result = await env.step(mock_action)
        print(f"\nStep result:")
        print(f"  Reward: {result.reward}")
        print(f"  Done: {result.done}")
        print(f"  Info: {result.info}")

    asyncio.run(test_env())
