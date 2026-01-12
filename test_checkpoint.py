"""
快速测试 checkpoint 加载

Usage:
    python -m sft_pipeline.test_checkpoint
    python -m sft_pipeline.test_checkpoint --checkpoint "mint://695a58c07c20038536a2867b/sampler_weights/lora-sft-inference"
"""
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Test checkpoint loading")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (mint:// URL)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model name")
    args = parser.parse_args()

    import mint
    from mint import types
    from transformers import AutoTokenizer

    print("=" * 60)
    print("Checkpoint Test")
    print("=" * 60)

    service_client = mint.ServiceClient()

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        try:
            sampling_client = service_client.create_sampling_client(
                model_path=args.checkpoint
            )
            print("✓ Checkpoint loaded successfully!")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            print("\nTrying alternative format...")
            # 尝试不同的格式
            return
    else:
        print(f"Using base model: {args.base_model}")
        sampling_client = service_client.create_sampling_client(
            base_model=args.base_model
        )
        print("✓ Base model loaded successfully!")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 测试生成
    print("\n" + "=" * 60)
    print("Test Generation")
    print("=" * 60)

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

    eos_id = tokenizer.eos_token_id
    stop_tokens = [eos_id] if eos_id is not None else []

    result = sampling_client.sample(
        prompt=prompt_tokens,
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=200,
            temperature=0.7,
            stop=stop_tokens
        )
    ).result()

    response = tokenizer.decode(result.sequences[0].tokens)
    print(f"\nModel response:\n{response}")

    # 检查是否有工具调用
    if "<tool_call>" in response:
        print("\n✓ Model produced tool call!")
    else:
        print("\n✗ No tool call found")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
