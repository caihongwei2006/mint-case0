"""
测试 LoRA 推理

从保存的 sampler weights 加载模型并进行推理测试

Usage:
    python -m sft_pipeline.test_inference --weights tinker://xxx/lora-sft-inference
"""
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Test LoRA inference with MinT")
    parser.add_argument("--weights", type=str, required=True,
                        help="Sampler weights path (tinker:// URL)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Base model name")
    args = parser.parse_args()

    import mint
    from mint import types

    print("=" * 60)
    print("MinT LoRA Inference Test")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"Base model: {args.base_model}")
    print()

    # 连接服务
    print("Connecting to MinT server...")
    service_client = mint.ServiceClient()

    # 从 sampler weights 创建推理客户端
    print("Creating sampling client from weights...")
    sampling_client = service_client.create_sampling_client(
        model_path=args.weights
    )

    # 获取 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size:,}")

    # 测试用例 - 使用 Qwen ChatML 格式，匹配训练数据
    test_cases = [
        # 测试1：看模型是否学会了工具调用格式
        {
            "prompt": """<|im_start|>system
你是一个DevOps助手，负责分析项目结构并生成Dockerfile。你有两个工具：ls_file_tree（查看文件结构）和ls_dependencies（获取依赖列表）。<|im_end|>
<|im_start|>user
请为当前Python项目生成Dockerfile。<|im_end|>
<|im_start|>assistant
""",
            "description": "测试工具调用"
        },
        # 测试2：给定工具结果后，看模型是否能继续调用下一个工具
        {
            "prompt": """<|im_start|>system
你是一个DevOps助手，负责分析项目结构并生成Dockerfile。你有两个工具：ls_file_tree（查看文件结构）和ls_dependencies（获取依赖列表）。<|im_end|>
<|im_start|>user
请为当前Python项目生成Dockerfile。<|im_end|>
<|im_start|>assistant
<tool_call>
ls_file_tree()
</tool_call><|im_end|>
<|im_start|>user
<tool_result>
项目文件结构（Python项目）:
.
├── app.py
├── requirements.txt
├── utils.py
├── README.md
</tool_result><|im_end|>
<|im_start|>assistant
""",
            "description": "测试第一步CoT + 调用ls_dependencies"
        },
        # 测试3：完整流程，看模型是否能生成Dockerfile
        {
            "prompt": """<|im_start|>system
你是一个DevOps助手，负责分析项目结构并生成Dockerfile。你有两个工具：ls_file_tree（查看文件结构）和ls_dependencies（获取依赖列表）。<|im_end|>
<|im_start|>user
请为当前Python项目生成Dockerfile。<|im_end|>
<|im_start|>assistant
<tool_call>
ls_file_tree()
</tool_call><|im_end|>
<|im_start|>user
<tool_result>
项目文件结构（Python项目）:
.
├── app.py
├── requirements.txt
├── utils.py
├── README.md
</tool_result><|im_end|>
<|im_start|>assistant
<思考>
我查看了当前项目的文件结构，发现这是一个Python项目，包含app.py、requirements.txt、utils.py、README.md。

关键发现：
1. 项目中存在 requirements.txt 文件，依赖问题已解决，可以直接使用 pip install -r requirements.txt 安装依赖。
2. 入口文件是 app.py。

接下来我需要确认具体的依赖列表，以便生成完整的Dockerfile。
</思考>

<tool_call>
ls_dependencies()
</tool_call><|im_end|>
<|im_start|>user
<tool_result>
requirements.txt:
flask>=2.0.0
redis>=4.0.0
requests>=2.28.0
</tool_result><|im_end|>
<|im_start|>assistant
""",
            "description": "测试生成最终Dockerfile"
        },
    ]

    print("\n" + "=" * 60)
    print("Running inference tests...")
    print("=" * 60)

    for i, case in enumerate(test_cases):
        print(f"\n--- Test {i + 1}: {case['description']} ---")

        prompt_tokens = types.ModelInput.from_ints(
            tokenizer.encode(case["prompt"], add_special_tokens=False)
        )

        result = sampling_client.sample(
            prompt=prompt_tokens,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=512,
                temperature=0.7,
                stop=[tokenizer.eos_token_id]
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
