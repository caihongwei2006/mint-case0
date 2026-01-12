针对MinT api，做了一些微调，RL上的尝试

环境建模与数据准备----非常理想化，也非常简单，0.6B的模型，不能过于期待效果
模拟工具池 (Tool Definitions)我们定义两个核心工具，模型必须各调用一次。
Tool A: dependency_scanner功能：模拟扫描 Python 项目依赖。数据源：从预设的 100 个常见 Python 库（如 numpy, flask, torch 等）中随机抽取 3-6 个。输出示例：["numpy", "requests", "pandas"]Tool 
B: file_tree_scanner功能：模拟扫描项目文件结构。数据源：从预设的 30 个常见文件名（如 app.py, config.json 等）中随机抽取 3-5 个。核心逻辑开关：30% 的概率会在文件列表中包含 requirements.txt。输出示例：["app.py", "utils.py", "requirements.txt"]
2.2 逻辑分支 (RL 的核心考点)环境随机生成的状态将创造两个平行宇宙，模型必须学会区分：场景 A (70% 概率)：无 requirements.txt正确行为：Dockerfile 必须显式安装 Tool A 返回的依赖。
预期指令：RUN pip install numpy requests pandas场景 B (30% 概率)：有 requirements.txt正确行为：Dockerfile 必须优先使用文件安装。
预期指令：COPY requirements.txt . 以及 RUN pip install -r requirements.txt
3. SFT 数据生成策略 (Distillation)由于 0.6B 模型非 Thinking Model，我们需要通过 SFT 注入思维链（CoT）。
3.1 生成脚本 (Python)使用 Claude 4.5 Sonnet，生成整个端到端的链条

综合数据：约等于5个tool_call/任务

设计小缺陷：不该给ls_10_lines设置文件input，会误导模型，实际上应该本身就ls所有的前10行，或者干脆写成“列出所有依赖”


可优化的地方
1 语言种类—环境构造的地方，设计工具的接口
2 多步骤的访问，例如文件树ls，可以在进一步设计
3 sft数据量可以酌情扩大
4 每一段都应该有CoT，但是消费有点多了--claude的api消耗

