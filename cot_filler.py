"""
CoT Filler - 为SFT训练数据填充思维链

逐行读取jsonl，调用Claude生成CoT，正则替换后覆盖写回文件。
失败重试3次，超过则记录行号并终止。

使用方法:
    python cot_filler.py [input_file]
    python cot_filler.py  # 默认处理 ../sft_train_data.jsonl
"""
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


@dataclass
class SFTConfig:
    """简化的配置类"""
    teacher_model: str = "anthropic/claude-sonnet-4.5"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))


COT_PROMPT = """你是一个Dockerfile推理分析助手。我会给你一条完整的SFT训练数据（JSON格式），其中包含工具调用结果和最终生成的Dockerfile。

你的任务是：根据messages中的tool_result内容，生成一段思维链（CoT）。

【CoT格式要求 - 必须严格包含以下5个部分】

1. 项目认知：由ls_file_tree的结果，我看到本项目是一个[项目类型]的项目，包含[关键文件列表]。

2. 依赖分析：由ls_top10_lines的结果，我看到本项目的依赖为：[依赖包列表]。

3. 入口判断：从已知的几个文件中，最有可能为入口文件的可能性是 [文件1]>[文件2]>[文件3]...

4. 安装策略：本项目[存在/不存在]requirements.txt，应使用[pip install -r requirements.txt / pip install xxx xxx]指令安装依赖。

5. Dockerfile结构：Dockerfile的经典组成部分包括：FROM基础镜像、WORKDIR工作目录、COPY复制文件、RUN安装依赖、EXPOSE暴露端口（默认8000）、CMD启动命令，生成的Dockerfile应涵盖以上所有部分并符合语法规范。

【输出要求】
只输出CoT内容本身。不要输出<思考>标签，不要输出任何其他解释性文字。

【SFT训练数据】
"""


class CoTFiller:
    """CoT填充器 - 基于JSON解析"""

    def __init__(self, config: SFTConfig = None):
        self.config = config or SFTConfig()
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )

    def generate_cot(self, json_data: dict, max_retries: int = 3) -> str:
        """
        调用Claude生成CoT

        Args:
            json_data: 解析后的JSON对象
            max_retries: 最大重试次数

        Returns:
            生成的CoT内容

        Raises:
            Exception: 超过重试次数仍失败
        """
        for attempt in range(max_retries):
            try:
                # 将JSON对象序列化为字符串传给Claude
                json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                
                response = self.client.chat.completions.create(
                    model=self.config.teacher_model,
                    max_tokens=2048,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": COT_PROMPT + json_str}
                    ]
                )
                cot_content = response.choices[0].message.content.strip()

                # 验证CoT不为空
                if not cot_content:
                    raise ValueError("生成的CoT为空")

                return cot_content

            except Exception as e:
                print(f"  尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt == max_retries - 1:
                    raise

        raise Exception("不应该到达这里")

    def has_empty_cot(self, json_data: dict) -> bool:
        """
        检查JSON数据中是否有空的<思考>标签

        Args:
            json_data: 解析后的JSON对象

        Returns:
            是否包含空的<思考>\n\n</思考>
        """
        messages = json_data.get("messages", [])
        
        # 找到最后一个assistant消息
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # 检查是否包含空的<思考>\n\n</思考>（JSON字符串中的\n是字面量）
                if "<思考>\n\n</思考>" in content:
                    return True
                break
        
        return False

    def insert_cot(self, json_data: dict, cot_content: str) -> dict:
        """
        将CoT内容插入到JSON数据中

        Args:
            json_data: 原始JSON对象
            cot_content: 生成的CoT内容

        Returns:
            插入CoT后的JSON对象
        """
        messages = json_data.get("messages", [])
        
        # 找到最后一个assistant消息并替换
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # 替换<思考>\n\n</思考>为<思考>\n内容\n</思考>
                new_content = content.replace(
                    "<思考>\n\n</思考>",
                    f"<思考>\n{cot_content}\n</思考>"
                )
                msg["content"] = new_content
                break
        
        return json_data

    def process_file(self, input_path: str):
        """
        处理整个jsonl文件

        Args:
            input_path: jsonl文件路径
        """
        input_path = Path(input_path)

        if not input_path.exists():
            print(f"错误：文件不存在 {input_path}")
            sys.exit(1)

        # 读取所有行
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total = len(lines)
        print(f"共 {total} 条数据需要处理")

        for i, line in enumerate(lines):
            line_num = i + 1
            print(f"处理第 {line_num}/{total} 行...")

            try:
                # 解析JSON
                json_data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"  跳过：第 {line_num} 行JSON解析失败: {e}")
                continue

            # 检查是否需要填充
            if not self.has_empty_cot(json_data):
                print(f"  跳过：第 {line_num} 行不包含空的<思考>\\n\\n</思考>标签")
                continue

            try:
                # 生成CoT
                cot_content = self.generate_cot(json_data)
                print(f"  CoT生成成功，长度: {len(cot_content)}")

                # 插入CoT
                json_data = self.insert_cot(json_data, cot_content)

                # 序列化回单行JSON
                new_line = json.dumps(json_data, ensure_ascii=False) + "\n"
                lines[i] = new_line

                # 立即写回文件（覆盖）
                with open(input_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                print(f"  第 {line_num} 行处理完成并已写入文件")

            except Exception as e:
                print(f"\n错误：第 {line_num} 行处理失败，已重试3次")
                print(f"错误信息: {e}")
                print(f"\n请手动处理第 {line_num} 行后重新运行脚本")
                sys.exit(1)

        print(f"\n全部完成！共处理 {total} 条数据")


def main():
    """主函数"""
    import argparse

    # 获取脚本所在目录的父目录（case0目录）
    script_dir = Path(__file__).parent
    default_file = script_dir.parent / 'sft_train_data.jsonl'

    parser = argparse.ArgumentParser(description='为SFT训练数据填充思维链')
    parser.add_argument(
        'input_file',
        nargs='?',
        default=str(default_file),
        help=f'输入的jsonl文件路径 (默认: {default_file})'
    )

    args = parser.parse_args()

    filler = CoTFiller()
    filler.process_file(args.input_file)


if __name__ == '__main__':
    main()
