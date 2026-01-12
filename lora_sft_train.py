"""
LoRA SFT 训练脚本

使用 MinT SDK 对 Qwen3-0.6B 进行 SFT 微调

Usage:
    python -m sft_pipeline.lora_sft_train --input sft_train_data.jsonl --output /Users/dp/mint/case0/lora

    # 快速测试（少量数据）
    python -m sft_pipeline.lora_sft_train --input sft_train_data.jsonl --max-samples 10 --epochs 1
"""
import os
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="LoRA SFT Training with MinT")
    
    # 获取脚本所在目录，构造默认路径
    script_dir = Path(__file__).parent
    default_input = script_dir.parent / "sft_train_data.jsonl"
    default_output = script_dir.parent / "lora"
    
    parser.add_argument("--input", type=str, default=str(default_input),
                        help="Input JSONL file path")
    parser.add_argument("--output", type=str, default=str(default_output),
                        help="Output directory for LoRA weights")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model to fine-tune")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to use (for testing)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N steps")
    args = parser.parse_args()

    # 检查 API key
    if not os.environ.get("MINT_API_KEY"):
        print("ERROR: MINT_API_KEY not found!")
        print("Please set it in .env file or environment variable")
        return

    import mint
    from mint import types
    from data_converter import load_sft_data_for_mint

    print("=" * 60)
    print("LoRA SFT Training with MinT")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Input file: {args.input}")
    print(f"Output dir: {args.output}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()

    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 连接 MinT 服务
    print("Connecting to MinT server...")
    service_client = mint.ServiceClient()

    # 创建训练客户端
    print(f"Creating training client for {args.base_model}...")
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )

    # 获取 tokenizer
    tokenizer = training_client.get_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")

    # 加载数据
    print(f"\nLoading data from {args.input}...")
    datums = load_sft_data_for_mint(
        args.input,
        tokenizer,
        max_samples=args.max_samples
    )
    print(f"Loaded {len(datums)} training samples")

    if len(datums) == 0:
        print("ERROR: No valid training samples!")
        return

    # 计算训练步数
    total_samples = len(datums)
    steps_per_epoch = (total_samples + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    print(f"\nTraining configuration:")
    print(f"  Total samples: {total_samples}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print()

    # 训练循环
    print("Starting training...")
    all_losses = []
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*40}")

        epoch_losses = []

        # 按 batch 训练
        for batch_start in tqdm(range(0, total_samples, args.batch_size),
                                desc=f"Epoch {epoch + 1}"):
            batch_end = min(batch_start + args.batch_size, total_samples)
            batch_data = datums[batch_start:batch_end]

            # Forward-backward
            try:
                fwdbwd_result = training_client.forward_backward(
                    data=batch_data,
                    loss_fn="cross_entropy"
                ).result()

                # 计算 batch loss
                total_loss = 0.0
                total_weight = 0.0
                for i, out in enumerate(fwdbwd_result.loss_fn_outputs):
                    logprobs = out["logprobs"]
                    if hasattr(logprobs, "tolist"):
                        logprobs = logprobs.tolist()
                    weights = batch_data[i].loss_fn_inputs["weights"]
                    if hasattr(weights, "tolist"):
                        weights = weights.tolist()
                    for lp, wt in zip(logprobs, weights):
                        total_loss += -lp * wt
                        total_weight += wt

                batch_loss = total_loss / max(total_weight, 1)
                epoch_losses.append(batch_loss)
                all_losses.append(batch_loss)

                # Optimizer step
                training_client.optim_step(
                    types.AdamParams(learning_rate=args.learning_rate)
                ).result()

                global_step += 1

                # 定期保存 checkpoint
                if args.save_every > 0 and global_step % args.save_every == 0:
                    ckpt_name = f"checkpoint-step-{global_step}"
                    print(f"\nSaving checkpoint: {ckpt_name}")
                    ckpt_result = training_client.save_state(name=ckpt_name).result()
                    ckpt_path = ckpt_result.path
                    print(f"Checkpoint path: {ckpt_path}")
                    # 发布中间检查点
                    try:
                        pub_result = subprocess.run(
                            ["mint", "checkpoint", "publish", ckpt_path],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        if pub_result.returncode == 0:
                            print(f"✓ Published: {ckpt_name}")
                    except Exception as e:
                        print(f"Warning: Could not publish {ckpt_name}: {e}")

            except Exception as e:
                print(f"\nError at step {global_step}: {e}")
                continue

        # Epoch 统计
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f"\nEpoch {epoch + 1} complete. Avg loss: {avg_epoch_loss:.4f}")

    # 保存最终模型
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    if all_losses:
        print(f"Final loss: {all_losses[-1]:.4f}")
        print(f"Loss reduction: {all_losses[0]:.4f} -> {all_losses[-1]:.4f}")

    # 保存最终 checkpoint
    print(f"\nSaving final checkpoint to {args.output}...")
    final_checkpoint = training_client.save_state(name="lora-sft-final").result()
    final_ckpt_path = final_checkpoint.path
    print(f"Checkpoint saved: {final_ckpt_path}")

    # 发布 checkpoint 到云端（持久保存）
    print(f"\nPublishing checkpoint to cloud for persistence...")
    try:
        result = subprocess.run(
            ["mint", "checkpoint", "publish", final_ckpt_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"✓ Checkpoint published successfully: {final_ckpt_path}")
        else:
            print(f"Warning: Failed to publish checkpoint: {result.stderr}")
    except Exception as e:
        print(f"Warning: Could not publish checkpoint: {e}")

    # 保存用于推理的权重
    print("\nSaving weights for inference...")

    # 先获取 sampler weights 路径
    sampler_result = training_client.save_weights_for_sampler(name="lora-sft-inference").result()
    sampler_weights_path = sampler_result.path
    print(f"[DEBUG] sampler_result type: {type(sampler_result)}")
    print(f"[DEBUG] sampler_result: {sampler_result}")
    print(f"Sampler weights path: {sampler_weights_path}")

    # 检查路径格式
    if sampler_weights_path.startswith("mint://"):
        print("✓ Path is in mint:// format")
    else:
        print("⚠ Path is NOT in mint:// format")
        # 尝试构造 mint:// 格式
        # 从路径中提取 training run ID
        import re
        match = re.search(r'/tinker_checkpoints/([a-f0-9]+)/', sampler_weights_path)
        if match:
            run_id = match.group(1)
            mint_path = f"mint://{run_id}/sampler_weights/lora-sft-inference"
            print(f"Constructed mint:// path: {mint_path}")
            sampler_weights_path = mint_path

    # 然后创建 sampling client
    sampling_client = service_client.create_sampling_client(model_path=sampler_weights_path)

    # 保存路径到文件，方便后续使用
    checkpoint_info_path = output_path / "checkpoint_info.txt"
    with open(checkpoint_info_path, "w") as f:
        f.write(f"training_checkpoint: {final_ckpt_path}\n")
        f.write(f"sampler_weights: {sampler_weights_path}\n")
    print(f"Checkpoint info saved to: {checkpoint_info_path}")

    # 简单测试
    print("\n" + "="*40)
    print("Quick inference test")
    print("="*40)

    test_prompt = """<|system|>
你是一个DevOps助手，负责分析项目结构并生成Dockerfile。
<|user|>
请为当前Python项目生成Dockerfile。

可用工具：ls_file_tree, ls_top10_lines
<|assistant|>
"""

    prompt_tokens = types.ModelInput.from_ints(
        tokenizer.encode(test_prompt, add_special_tokens=True)
    )

    result = sampling_client.sample(
        prompt=prompt_tokens,
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=100,
            temperature=0.7,
            stop=[tokenizer.eos_token_id] if tokenizer.eos_token_id else []
        )
    ).result()

    response = tokenizer.decode(result.sequences[0].tokens)
    print(f"Model response:\n{response[:500]}...")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"LoRA weights saved to: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
