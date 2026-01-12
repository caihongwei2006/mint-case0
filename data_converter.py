"""
数据转换器：将 sft_train_data.jsonl 转换为 MinT Datum 格式

Usage:
    from sft_pipeline.data_converter import load_sft_data_for_mint

    datums = load_sft_data_for_mint("sft_train_data.jsonl", tokenizer)
"""
import json
from pathlib import Path
from typing import Optional


def convert_messages_to_tokens_and_weights(messages: list[dict], tokenizer) -> tuple[list[int], list[float]]:
    """
    将 messages 格式转换为 token 序列和对应的权重

    策略（多轮对话 Q-A-Q-A-Q-A）：
    - system/user 消息：weight=0（不训练）
    - assistant 消息：weight=1（训练）

    Returns:
        (tokens, token_weights)
    """
    if not messages:
        return [], []

    all_tokens = []
    all_weights = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # 构建该消息的完整文本
        if role == "system":
            msg_text = f"<|system|>\n{content}\n"
            weight = 0.0
        elif role == "user":
            msg_text = f"<|user|>\n{content}\n"
            weight = 0.0
        elif role == "assistant":
            msg_text = f"<|assistant|>\n{content}\n"
            # assistant header 不训练，content 训练
            # 分开 tokenize
            header_tokens = tokenizer.encode("<|assistant|>\n", add_special_tokens=False)
            content_tokens = tokenizer.encode(f"{content}\n", add_special_tokens=False)

            all_tokens.extend(header_tokens)
            all_weights.extend([0.0] * len(header_tokens))

            all_tokens.extend(content_tokens)
            all_weights.extend([1.0] * len(content_tokens))

            continue
        else:
            continue

        # 对于 system 和 user，直接 tokenize
        tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        all_tokens.extend(tokens)
        all_weights.extend([weight] * len(tokens))

    # 在开头添加 BOS token（如果需要）
    if tokenizer.bos_token_id is not None:
        all_tokens = [tokenizer.bos_token_id] + all_tokens
        all_weights = [0.0] + all_weights

    # 在结尾添加 EOS token
    if tokenizer.eos_token_id is not None:
        all_tokens.append(tokenizer.eos_token_id)
        all_weights.append(1.0)  # EOS 也训练

    return all_tokens, all_weights


def split_conversation_to_progressive_samples(messages: list[dict]) -> list[list[dict]]:
    """
    将多轮对话拆分为累进式训练样本

    输入: [system, user, assistant1, user, assistant2, user, assistant3, ...]
    输出: [
        [system, user, assistant1],                                    # 学习第1次回复
        [system, user, assistant1, user, assistant2],                  # 学习第2次回复
        [system, user, assistant1, user, assistant2, user, assistant3], # 学习第3次回复
        ...
    ]

    每个样本只训练最后一个 assistant 消息
    """
    if not messages:
        return []

    samples = []
    current_context = []

    for msg in messages:
        current_context.append(msg)

        # 每遇到一个 assistant 消息，生成一个训练样本
        if msg["role"] == "assistant":
            samples.append(current_context.copy())

    return samples


def convert_progressive_sample_to_tokens_and_weights(
    messages: list[dict],
    tokenizer
) -> tuple[list[int], list[float]]:
    """
    将累进式样本转换为 token 和权重

    只有最后一个 assistant 消息的 content 被训练（weight=1）
    其他所有内容 weight=0
    """
    if not messages:
        return [], []

    all_tokens = []
    all_weights = []

    # 找到最后一个 assistant 消息的索引
    last_assistant_idx = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            last_assistant_idx = i

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            msg_text = f"<|system|>\n{content}\n"
            tokens = tokenizer.encode(msg_text, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_weights.extend([0.0] * len(tokens))

        elif role == "user":
            msg_text = f"<|user|>\n{content}\n"
            tokens = tokenizer.encode(msg_text, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_weights.extend([0.0] * len(tokens))

        elif role == "assistant":
            # header 不训练
            header_tokens = tokenizer.encode("<|assistant|>\n", add_special_tokens=False)
            content_tokens = tokenizer.encode(f"{content}\n", add_special_tokens=False)

            all_tokens.extend(header_tokens)
            all_weights.extend([0.0] * len(header_tokens))

            all_tokens.extend(content_tokens)

            # 只有最后一个 assistant 消息的 content 训练
            if i == last_assistant_idx:
                all_weights.extend([1.0] * len(content_tokens))
            else:
                all_weights.extend([0.0] * len(content_tokens))

    # 添加 BOS token
    if tokenizer.bos_token_id is not None:
        all_tokens = [tokenizer.bos_token_id] + all_tokens
        all_weights = [0.0] + all_weights

    # 只有最后一个 assistant 样本添加 EOS 并训练
    if tokenizer.eos_token_id is not None:
        all_tokens.append(tokenizer.eos_token_id)
        # 最后一个样本的 EOS 也训练
        all_weights.append(1.0)

    return all_tokens, all_weights


def create_datum_from_messages(messages: list[dict], tokenizer) -> Optional[dict]:
    """
    将 messages 转换为 MinT 训练所需的数据格式（旧版，训练所有 assistant）

    Returns:
        {
            "input_tokens": [...],
            "target_tokens": [...],
            "weights": [...]
        }
    """
    tokens, token_weights = convert_messages_to_tokens_and_weights(messages, tokenizer)

    if not tokens:
        return None

    # 对于 next-token prediction:
    # - input_tokens: 所有 token 除了最后一个
    # - target_tokens: 所有 token 除了第一个（向右移动1位）
    # - weights: 对齐 target_tokens
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = token_weights[1:]  # weights 对齐到 target

    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "weights": weights,
    }


def create_datum_from_progressive_sample(messages: list[dict], tokenizer) -> Optional[dict]:
    """
    将累进式样本转换为 MinT 训练所需的数据格式
    只训练最后一个 assistant 消息

    Returns:
        {
            "input_tokens": [...],
            "target_tokens": [...],
            "weights": [...]
        }
    """
    tokens, token_weights = convert_progressive_sample_to_tokens_and_weights(messages, tokenizer)

    if not tokens:
        return None

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = token_weights[1:]

    return {
        "input_tokens": input_tokens,
        "target_tokens": target_tokens,
        "weights": weights,
    }


def load_sft_data_for_mint(
    jsonl_path: str,
    tokenizer,
    max_samples: Optional[int] = None,
    progressive: bool = True
) -> list:
    """
    加载 JSONL 文件并转换为 MinT Datum 列表

    Args:
        jsonl_path: JSONL 文件路径
        tokenizer: HuggingFace tokenizer
        max_samples: 最大样本数（用于测试）
        progressive: 是否使用累进式拆分（默认 True）
                    True: 每个 assistant 回复作为一个样本（逐步学习）
                    False: 整个对话作为一个样本（旧版行为）

    Returns:
        list of Datum-compatible dicts
    """
    from mint import types

    datums = []
    path = Path(jsonl_path)
    total_conversations = 0
    total_samples = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line.strip())
                messages = entry.get("messages", [])
                total_conversations += 1

                if progressive:
                    # 累进式拆分：每个 assistant 回复作为一个样本
                    progressive_samples = split_conversation_to_progressive_samples(messages)

                    for sample_messages in progressive_samples:
                        if max_samples and total_samples >= max_samples:
                            break

                        data = create_datum_from_progressive_sample(sample_messages, tokenizer)
                        if data is None:
                            continue

                        datum = types.Datum(
                            model_input=types.ModelInput.from_ints(tokens=data["input_tokens"]),
                            loss_fn_inputs={
                                "target_tokens": data["target_tokens"],
                                "weights": data["weights"],
                            }
                        )
                        datums.append(datum)
                        total_samples += 1
                else:
                    # 旧版行为：整个对话作为一个样本
                    if max_samples and total_samples >= max_samples:
                        break

                    data = create_datum_from_messages(messages, tokenizer)
                    if data is None:
                        continue

                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=data["input_tokens"]),
                        loss_fn_inputs={
                            "target_tokens": data["target_tokens"],
                            "weights": data["weights"],
                        }
                    )
                    datums.append(datum)
                    total_samples += 1

                if max_samples and total_samples >= max_samples:
                    break

            except Exception as e:
                print(f"Warning: Failed to process line {i}: {e}")
                continue

    print(f"Loaded {total_conversations} conversations -> {len(datums)} training samples")
    if progressive:
        print(f"  (progressive mode: ~{len(datums) / max(total_conversations, 1):.1f} samples per conversation)")

    return datums


def load_raw_data(jsonl_path: str, max_samples: Optional[int] = None) -> list[dict]:
    """
    加载原始 JSONL 数据（不需要 tokenizer）

    Returns:
        list of {"messages": [...], "metadata": {...}}
    """
    data = []
    path = Path(jsonl_path)

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            try:
                entry = json.loads(line.strip())
                data.append(entry)
            except Exception as e:
                print(f"Warning: Failed to parse line {i}: {e}")
                continue

    return data


def convert_messages_to_text(messages: list[dict]) -> str:
    """
    将 messages 转换为可读文本格式（用于调试和显示）

    Returns:
        格式化的对话文本
    """
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"<|{role}|>\n{content}\n")
    return "".join(lines)


if __name__ == "__main__":
    # 测试转换
    import argparse
    from pathlib import Path

    # 获取脚本所在目录的父目录
    script_dir = Path(__file__).parent
    default_input = script_dir.parent / "sft_train_data.jsonl"

    parser = argparse.ArgumentParser(description="Test data converter")
    parser.add_argument("--input", type=str, default=str(default_input))
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--no-progressive", action="store_true", help="Disable progressive mode")
    args = parser.parse_args()

    # 加载原始数据测试
    raw_data = load_raw_data(args.input, args.max_samples)
    print(f"Loaded {len(raw_data)} conversations from JSONL")

    if raw_data:
        # 测试累进式拆分（不需要 tokenizer）
        print("\n=== Progressive Split Test ===")
        for i, entry in enumerate(raw_data[:2]):
            messages = entry["messages"]
            samples = split_conversation_to_progressive_samples(messages)
            print(f"\nConversation {i}: {len(messages)} messages -> {len(samples)} training samples")
            for j, sample in enumerate(samples):
                last_assistant = [m for m in sample if m["role"] == "assistant"][-1]
                content_preview = last_assistant["content"][:50].replace("\n", " ")
                print(f"  Sample {j+1}: {len(sample)} messages, trains on: {content_preview}...")

        # 需要 tokenizer 来测试完整转换
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

            print("\n=== Token Analysis ===")
            messages = raw_data[0]["messages"]

            # 测试旧版（整体）
            tokens_old, weights_old = convert_messages_to_tokens_and_weights(messages, tokenizer)
            print(f"\nOld mode (full conversation):")
            print(f"  Total tokens: {len(tokens_old)}")
            print(f"  Trainable tokens: {sum(1 for w in weights_old if w > 0)}")

            # 测试新版（累进式）
            samples = split_conversation_to_progressive_samples(messages)
            print(f"\nProgressive mode ({len(samples)} samples):")
            total_trainable = 0
            for j, sample in enumerate(samples):
                tokens, weights = convert_progressive_sample_to_tokens_and_weights(sample, tokenizer)
                trainable = sum(1 for w in weights if w > 0)
                total_trainable += trainable
                print(f"  Sample {j+1}: {len(tokens)} tokens, {trainable} trainable")

            print(f"\nTotal trainable tokens (progressive): {total_trainable}")
            print(f"Total trainable tokens (old mode): {sum(1 for w in weights_old if w > 0)}")

        except Exception as e:
            print(f"\nCouldn't load tokenizer for testing: {e}")
            print("Run with actual tokenizer for full test")
