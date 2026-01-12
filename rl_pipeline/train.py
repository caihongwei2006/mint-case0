"""
RL 训练脚本

使用 GRPO 风格的 RL 训练来优化 Dockerfile 生成能力

Usage:
    python rl_pipeline/train.py --checkpoint-info /path/to/checkpoint_info.txt
    python rl_pipeline/train.py --sft-checkpoint "tinker://xxx/weights/lora-sft-final"
"""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加父目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sft_pipeline.config import SFTConfig
from rl_pipeline.env import DockerfileEnv, DockerfileEnvGroupBuilder
from rl_pipeline.reward import compute_reward, parse_tool_calls, parse_dockerfile


@dataclass
class RLConfig:
    """RL 训练配置"""
    # Model
    base_model: str = "Qwen/Qwen3-0.6B"
    lora_rank: int = 16

    # Training
    num_iterations: int = 100
    group_size: int = 4  # GRPO: 每个 prompt 采样多少个 response
    batch_size: int = 2  # 每个 iteration 处理多少个 prompt
    learning_rate: float = 1e-5
    max_tokens_per_turn: int = 512
    max_turns: int = 10

    # Loss function
    loss_fn: str = "importance_sampling"  # or "ppo"

    # Checkpointing
    save_every: int = 20
    output_dir: str = "rl_checkpoints"

    # Logging
    log_every: int = 5


def load_checkpoint_info(checkpoint_info_path: str) -> dict:
    """从 checkpoint_info.txt 加载 checkpoint 路径"""
    info = {}
    with open(checkpoint_info_path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                info[key.strip()] = value.strip()
    return info


async def run_rollout(
    env: DockerfileEnv,
    sampling_client,
    tokenizer,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> dict:
    """
    运行单个 rollout

    Returns:
        {
            "observations": list of token sequences,
            "actions": list of token sequences,
            "logprobs": list of logprob sequences,
            "rewards": list of rewards,
            "total_reward": float,
            "info": dict
        }
    """
    from mint import types

    observations = []
    actions = []
    logprobs = []
    rewards = []

    # 获取初始观察
    obs, stop_conditions = await env.initial_observation()
    observations.append(obs.tokens)

    done = False
    while not done:
        # 采样
        prompt_tokens = types.ModelInput.from_ints(obs.tokens)

        result = sampling_client.sample(
            prompt=prompt_tokens,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_conditions
            )
        ).result()

        # 获取生成的 tokens 和 logprobs
        seq = result.sequences[0]
        action_tokens = seq.tokens
        action_logprobs = seq.logprobs if hasattr(seq, 'logprobs') else [0.0] * len(action_tokens)
        action_text = tokenizer.decode(action_tokens)

        actions.append(action_tokens)
        logprobs.append(action_logprobs)

        # 执行环境步骤
        from rl_pipeline.env import Action
        action = Action(tokens=action_tokens, text=action_text, logprobs=action_logprobs)
        step_result = await env.step(action)

        rewards.append(step_result.reward)
        done = step_result.done

        if not done:
            obs = step_result.observation
            observations.append(obs.tokens)

    return {
        "observations": observations,
        "actions": actions,
        "logprobs": logprobs,
        "rewards": rewards,
        "total_reward": sum(rewards),
        "info": env.get_state()
    }


def compute_advantages(rewards: list[float], baseline: float = 0.0) -> list[float]:
    """
    计算 advantages（GRPO 风格：相对于组内平均值）

    Args:
        rewards: 组内所有 rollout 的 rewards
        baseline: 基线（默认为组内平均值）

    Returns:
        advantages
    """
    if not rewards:
        return []

    mean_reward = sum(rewards) / len(rewards)
    advantages = [r - mean_reward for r in rewards]

    return advantages


async def train_step(
    training_client,
    rollouts: list[dict],
    config: RLConfig,
    tokenizer
):
    """
    执行一个训练步骤

    Args:
        training_client: MinT TrainingClient
        rollouts: 一组 rollout 结果
        config: 训练配置
        tokenizer: tokenizer
    """
    from mint import types

    # 计算 advantages（GRPO 风格）
    total_rewards = [r["total_reward"] for r in rollouts]
    advantages = compute_advantages(total_rewards)

    # 构建训练数据
    datums = []
    for rollout, advantage in zip(rollouts, advantages):
        # 对于每个 rollout，构建 Datum
        # 这里简化处理：只使用最后一个 observation-action 对
        if rollout["observations"] and rollout["actions"]:
            obs_tokens = rollout["observations"][-1]
            action_tokens = rollout["actions"][-1]
            action_logprobs = rollout["logprobs"][-1]

            # 完整序列 = observation + action
            full_tokens = obs_tokens + action_tokens

            # Target tokens（用于计算 loss）
            target_tokens = full_tokens[1:] + [tokenizer.eos_token_id or 0]

            # Weights: 只对 action 部分计算 loss
            weights = [0.0] * len(obs_tokens) + [1.0] * len(action_tokens)

            # Advantages: 扩展到每个 token
            token_advantages = [0.0] * len(obs_tokens) + [advantage] * len(action_tokens)

            # Reference logprobs（用于 importance sampling）
            ref_logprobs = [0.0] * len(obs_tokens) + action_logprobs

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(full_tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": target_tokens[:-1],
                    "weights": weights[:-1],
                    "advantages": token_advantages[:-1],
                    "logprobs": ref_logprobs[:-1],
                }
            )
            datums.append(datum)

    if not datums:
        return {"loss": 0.0}

    # Forward-backward
    fwd_bwd_result = training_client.forward_backward(
        data=datums,
        loss_fn=config.loss_fn
    ).result()

    # Optimizer step
    training_client.optim_step(
        types.AdamParams(learning_rate=config.learning_rate)
    ).result()

    # 计算 loss
    total_loss = 0.0
    for out in fwd_bwd_result.loss_fn_outputs:
        if "loss" in out:
            total_loss += out["loss"]

    return {
        "loss": total_loss / len(datums) if datums else 0.0,
        "num_datums": len(datums),
        "mean_reward": sum(total_rewards) / len(total_rewards) if total_rewards else 0.0,
        "mean_advantage": sum(advantages) / len(advantages) if advantages else 0.0
    }


async def main_async(args):
    """异步主函数"""
    import mint
    from mint import types
    from transformers import AutoTokenizer

    config = RLConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        num_iterations=args.num_iterations,
        group_size=args.group_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print("RL Training for Dockerfile Generation")
    print("=" * 60)
    print(f"Base model: {config.base_model}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Group size: {config.group_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num iterations: {config.num_iterations}")
    print()

    # 检查 API key
    if not os.environ.get("MINT_API_KEY"):
        print("ERROR: MINT_API_KEY not found!")
        return

    # 创建输出目录
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 连接 MinT
    print("Connecting to MinT server...")
    service_client = mint.ServiceClient()

    # 创建 training client
    print(f"Creating training client for {config.base_model}...")
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )

    # 加载 SFT checkpoint
    if args.sft_checkpoint:
        print(f"Loading SFT checkpoint: {args.sft_checkpoint}")
        training_client.load_state(args.sft_checkpoint).result()
        print("✓ SFT checkpoint loaded")
    elif args.checkpoint_info:
        info = load_checkpoint_info(args.checkpoint_info)
        if "training_checkpoint" in info:
            # 尝试构造 tinker:// 格式
            ckpt_path = info["training_checkpoint"]
            print(f"Loading SFT checkpoint: {ckpt_path}")
            try:
                training_client.load_state(ckpt_path).result()
                print("✓ SFT checkpoint loaded")
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Starting from base model...")

    # 获取 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")

    # 创建 sampling client
    sampling_client = training_client.save_weights_and_get_sampling_client(name="rl-sampler")

    # SFT config for environment
    sft_config = SFTConfig()

    # 训练循环
    print("\nStarting RL training...")
    metrics_log = []

    for iteration in tqdm(range(config.num_iterations), desc="Training"):
        iteration_metrics = {
            "iteration": iteration,
            "rewards": [],
            "losses": []
        }

        # 对于每个 batch
        for batch_idx in range(config.batch_size):
            # 创建一组环境（GRPO: 同一个 prompt，多个 rollout）
            env_builder = DockerfileEnvGroupBuilder(
                tokenizer=tokenizer,
                config=sft_config,
                group_size=config.group_size,
                max_turns=config.max_turns
            )
            envs = await env_builder.make_envs()

            # 运行 rollouts
            rollouts = []
            for env in envs:
                try:
                    rollout = await run_rollout(
                        env=env,
                        sampling_client=sampling_client,
                        tokenizer=tokenizer,
                        max_tokens=config.max_tokens_per_turn,
                        temperature=0.7
                    )
                    rollouts.append(rollout)
                    iteration_metrics["rewards"].append(rollout["total_reward"])
                except Exception as e:
                    print(f"\nRollout failed: {e}")
                    continue

            # 训练步骤
            if rollouts:
                try:
                    step_metrics = await train_step(
                        training_client=training_client,
                        rollouts=rollouts,
                        config=config,
                        tokenizer=tokenizer
                    )
                    iteration_metrics["losses"].append(step_metrics["loss"])
                except Exception as e:
                    print(f"\nTrain step failed: {e}")

        # 计算 iteration 统计
        if iteration_metrics["rewards"]:
            mean_reward = sum(iteration_metrics["rewards"]) / len(iteration_metrics["rewards"])
            iteration_metrics["mean_reward"] = mean_reward
        if iteration_metrics["losses"]:
            mean_loss = sum(iteration_metrics["losses"]) / len(iteration_metrics["losses"])
            iteration_metrics["mean_loss"] = mean_loss

        metrics_log.append(iteration_metrics)

        # 日志
        if (iteration + 1) % config.log_every == 0:
            print(f"\nIteration {iteration + 1}:")
            print(f"  Mean reward: {iteration_metrics.get('mean_reward', 0):.4f}")
            print(f"  Mean loss: {iteration_metrics.get('mean_loss', 0):.4f}")

        # 保存 checkpoint
        if (iteration + 1) % config.save_every == 0:
            ckpt_name = f"rl-step-{iteration + 1}"
            print(f"\nSaving checkpoint: {ckpt_name}")
            ckpt_result = training_client.save_state(name=ckpt_name).result()
            print(f"Checkpoint path: {ckpt_result.path}")

            # 更新 sampling client
            sampling_client = training_client.save_weights_and_get_sampling_client(name="rl-sampler")

    # 保存最终 checkpoint
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    final_ckpt = training_client.save_state(name="rl-final").result()
    print(f"Final checkpoint: {final_ckpt.path}")

    # 保存 sampler weights
    sampler_result = training_client.save_weights_for_sampler(name="rl-inference").result()
    print(f"Sampler weights: {sampler_result.path}")

    # 保存 checkpoint info
    checkpoint_info_path = output_path / "checkpoint_info.txt"
    with open(checkpoint_info_path, "w") as f:
        f.write(f"training_checkpoint: {final_ckpt.path}\n")
        f.write(f"sampler_weights: {sampler_result.path}\n")
    print(f"Checkpoint info saved to: {checkpoint_info_path}")

    # 保存 metrics
    metrics_path = output_path / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")
    print(f"Metrics saved to: {metrics_path}")

    # 打印最终统计
    if metrics_log:
        final_rewards = [m.get("mean_reward", 0) for m in metrics_log[-10:]]
        print(f"\nFinal 10 iterations mean reward: {sum(final_rewards) / len(final_rewards):.4f}")


def main():
    parser = argparse.ArgumentParser(description="RL Training for Dockerfile Generation")

    # Checkpoint
    parser.add_argument("--sft-checkpoint", type=str, default=None,
                        help="Path to SFT checkpoint (tinker:// or mint:// URL)")
    parser.add_argument("--checkpoint-info", type=str, default=None,
                        help="Path to checkpoint_info.txt from SFT training")

    # Model
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model name")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")

    # Training
    parser.add_argument("--num-iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--group-size", type=int, default=4,
                        help="GRPO group size (samples per prompt)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size (prompts per iteration)")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--output-dir", type=str, default="rl_checkpoints",
                        help="Output directory")

    args = parser.parse_args()

    # 运行异步主函数
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
