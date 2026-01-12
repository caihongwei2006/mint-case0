"""
测试 LoRA 推理

从保存的 checkpoint 加载模型并进行推理测试

Usage:
    python -m sft_pipeline.test_inference --checkpoint /vePFS-Mindverse/share/tinker_checkpoints/xxx/ckpt_xxx
"""
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Test LoRA inference with MinT")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint path from MinT")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model name")
    args = parser.parse_args()

    import mint
    from mint import types

    print("=" * 60)
    print("MinT LoRA Inference Test")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Base model: {args.base_model}")
    print()

    # 连接服务
    print("Connecting to MinT server...")
    service_client = mint.ServiceClient()

    # 方法1: 从 checkpoint 恢复训练客户端，然后获取推理客户端
    print("Loading checkpoint...")
    training_client = service_client.create_training_client_from_state(
        args.checkpoint
    )

    tokenizer = training_client.get_tokenizer()
    print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size:,}")

    # 获取推理客户端
    print("Creating sampling client...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="test-inference"
    )

    # 测试用例
    test_cases = [
        # 简单测试：看模型是否学会了工具调用格式
        {
            "prompt": """<|system|>
你是一个DevOps助手，负责分析项目结构并生成Dockerfile。
<|user|>
请为当前Python项目生成Dockerfile。

可用工具：ls_file_tree, ls_top10_lines
<|assistant|>
""",
            "description": "测试工具调用"
        },
        # 给定工具结果后的测试
        {
            "prompt": """<|system|>
你是一个DevOps助手，负责分析项目结构并生成Dockerfile。
<|user|>
请为当前Python项目生成Dockerfile。

可用工具：ls_file_tree, ls_top10_lines
<|assistant|>
<tool_call>
ls_file_tree({"path": "."})
</tool_call>
<|user|>
<tool_result>
项目文件结构:
.
├── app.py
├── requirements.txt
├── README.md

</tool_result>
<|assistant|>
<tool_call>
ls_top10_lines({"file_path": "requirements.txt"})
</tool_call>
<|user|>
<tool_result>
flask>=2.0.0
redis>=4.0.0
gunicorn>=20.0.0
</tool_result>
<|assistant|>
<tool_call>
ls_top10_lines({"file_path": "app.py"})
</tool_call>
<|user|>
<tool_result>
#!/usr/bin/env python3
from flask import Flask
import redis

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"
</tool_result>
<|assistant|>
""",
            "description": "测试生成Dockerfile"
        },
    ]

    print("\n" + "=" * 60)
    print("Running inference tests...")
    print("=" * 60)

    for i, case in enumerate(test_cases):
        print(f"\n--- Test {i + 1}: {case['description']} ---")

        prompt_tokens = types.ModelInput.from_ints(
            tokenizer.encode(case["prompt"], add_special_tokens=True)
        )

        result = sampling_client.sample(
            prompt=prompt_tokens,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=512,
                temperature=0.7,
                stop_token_ids=[tokenizer.eos_token_id]
            )
        ).result()

        response = tokenizer.decode(result.sequences[0].tokens)
        print(f"\nModel output:\n{response}")
        print("-" * 40)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
