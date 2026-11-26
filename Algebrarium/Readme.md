# Algebrarium

**Algebrarium** 是一个用于生成抽象代数和符号推理合成数据的框架。它专为大语言模型（LLM）的训练和评估而设计。

通过简单的 YAML 配置文件，你可以声明式地定义代数系统（如群、环、域或自定义逻辑结构），并程序化地生成无限的、具有不同复杂度的推理数据。生成的数据涵盖表达式计算、解方程、方程组求解等任务，并支持**演绎推理**（基于规则）和**归纳推理**（少样本学习/In-context Learning）两种模式。

## 核心特性

*   **声明式系统定义**：通过 YAML 模板定义数学结构（集合、运算规则、公理），而非硬编码。
*   **程序化生成**：基于定义的规则自动构建表达式树，生成无限且不重复的题目。
*   **动态难度控制**：通过命令行参数精确控制题目难度（如操作符深度、变量数量、元素集合大小）。
*   **多种任务类型**：
    *   **算术表达式求值**：计算 $a \circ b \circ c$。
    *   **代数方程求解**：求解 $a \circ x = b$ 中的 $x$。
    *   **群方程组**：在非交换群等结构中求解多变量方程组。
    *   **思维链 (CoT)**：生成逐步推理过程。
*   **双重推理模式**：
    *   **Deductive (演绎)**：提供完整的数学定义和规则，模型需根据定义推理。
    *   **Inductive (归纳)**：提供输入输出的具体案例，模型需从中归纳出规则并解决新问题。

## 安装

1.  克隆代码仓库。
2.  安装必要的 Python 依赖：

```bash
pip install pyyaml simple-parsing
git clone https://github.com/Linzwcs/Algebrarium.git
cd Algebrarium
pip install -e .
```

## 使用方法

主要入口是脚本文件（代码中体现为包含 `main` 函数的部分，假设保存为 `generate.py` 或作为模块运行）。

### 基础命令

```bash
python -m algebrarium.cli -t ./templates -o ./data/output.jsonl
```

### 命令行参数说明

| 参数 | 别名 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--template_dir` | `-t` | `./templates` | 包含 `.yaml` 模板文件的根目录。 |
| `--output_dir` | `-o` | `./data/` | 生成的 JSONL 文件保存路径。 |
| `--num_questions_per_instance` | `-n` | `3` | 为模板中的每个“演示实例”生成的题目数量。 |
| `--min_difficulty` | `-m` | `3` | 最小复杂度（元素数量/操作步数）。 |
| `--max_difficulty` | `-M` | `5` | 最大复杂度（元素数量/操作步数）。 |
| `--type` | | `deductive` | 推理类型：`deductive`（演绎）或 `inductive`（归纳）。 |
| `--num-equations-for-system` | | `2` | 生成方程组题目时的变量/方程数量。 |

### 示例：生成大规模归纳推理数据集

```bash
python -m algebrarium.builder \
  -t ./templates/groups \
  -o ./data/inductive_groups.jsonl \
  -n 1000 \
  -m 4 -M 8 \
  --type inductive
```

## 模板系统指南

Algebrarium 使用 YAML 将**抽象结构**与**具体实例**分离。

### 模板结构 (`example.yaml`)

```yaml

name: "Integer Modulo {n}"
description: "整数集合 {0, ..., {n}-1} 在模 {n} 加法下的运算。"


set:
  definition: "Z_{n}"
  
  validator: 
    type: "range_check" 
    params: { min: 0, max: "{n}" }

  sampler: 
    type: "range_sampler" 
    params: { min: 0, max: "{n}" }


signature:
  - symbol: "+"
    name: "addition"
    arity: 2           # 二元运算
    precedence: 1      # 优先级
    # 引用 registry.py 中注册的计算逻辑
    logic: 
      type: "modular_add"
      params: { mod: "{n}" }

# 3. 定义实例 (Instantiations)
# 这些参数会替换上面的 {n} 占位符，生成具体的代数系统
demo_instances:
  - { n: 5 }   # 生成 Z_5 的数据
  - { n: 12 }  # 生成 Z_12 的数据
  - { n: 100 } # 生成 Z_100 的数据
```

## 数据输出格式

输出为 **JSONL** 格式，每行一个 JSON 对象。

### 演绎模式 (Deductive) 示例
数据包含完整的背景知识（定义与规则）。

```json
{
  "question": "3 + (2 + 4) = ?",
  "answer": "4",
  "num_operate": 2,
  "category": "Expression",
  "background": "[MATHEMATICAL DEFINITION]\nSystem: Integers mod 5...\n[COMPUTATION RULES]...",
  "template_path": "modulo.yaml"
}
```

### 归纳模式 (Inductive) 示例
数据包含少样本案例（`inductive_cases`），背景知识被简化或移除，迫使模型进行类比推理。

```json
{
  "question": "2 + 4 = ?",
  "answer": "1",
  "inductive_cases": [
    {"question": "1 + 1 = ?", "answer": "2"},
    {"question": "3 + 2 = ?", "answer": "0"}
  ],
  "background": "...",
  "category": "Expression"
}
```

## 代码架构概览

*   **`builder.py` (构造器)**: 解析 YAML 模板，处理参数替换，通过 `SystemFactory` 将配置转化为可执行的 `AlgebraicSystem` 对象。也是 CLI 的入口。
*   **`core.py` (核心)**: 定义了代数系统的基本组件：
    *   `AlgebraicSystem`: 包含集合和运算规则的容器。
    *   `Set`: 管理元素的验证（Validator）和采样（Sampler）。
    *   `Action`: 定义符号、元数（Arity）和纯算法逻辑。
*   **`factory.py` (工厂)**: 负责生成具体的数学问题。包含生成表达式树（`Node`, `Leaf`）的核心算法：
    *   `ExpressionQuestionGenerator`: 生成求值题。
    *   `EquationQuestionGenerator`: 生成解方程题。
    *   `SystemOfGroupEquationsQuestionGenerator`: 生成方程组。
*   **`registry.py` (注册表)**: 插件化管理系统的核心。所有的验证逻辑、计算逻辑（如模运算、矩阵乘法）都需要通过 `@registry.register` 注册，才能在 YAML 中被引用。

## 扩展开发

要添加新的数学逻辑（例如矩阵乘法），需要在初始化代码（如 `components.py`）中进行注册：

```python
from algebrarium.registry import registry
@registry.register("action_logics", "custom")
def custom_builder(args):
    def logic(a, b):
        return a+b
    return logic
```


## 🚀 RL 训练与评估指南

Algebrarium 生成的数据拥有**确定性的标准答案 (Deterministic Ground Truth)**，非常适合用于强化学习。

为了确保评分准确，我们推荐使用 **`math-verify`** 库，并强制模型输出 **`\boxed{}`** 格式。

### 1. Prompt 策略

必须在 Prompt 中明确要求模型使用 LaTeX 格式框出最终答案。

**System Prompt:**
```text
Your task is to solve mathematical problems based strictly on the provided definitions and rules.
You must show your reasoning process step-by-step and output the final answer within a LaTeX box format: \boxed{answer}.
```

**User Input:**
```text
[MATHEMATICAL DEFINITION]
{background}

[PROBLEM]
{question}
```

### 2. 评分函数 (`scorer.py`)

安装依赖：`pip install math-verify`

```python
import re
from typing import Dict, Any
from math_verify import verify, parse

def extract_answer_content(text: str) -> str:
    """剥离 <think> 部分，仅保留回答内容"""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text

def is_valid_format(text: str) -> bool:
    """验证是否符合 <think>...</think> 结构"""
    if not isinstance(text, str): return False
    return bool(re.match(r"^<think>.*?</think>.*$", text, re.DOTALL))

def compute_score(solution_str: str, ground_truth: str, **kwargs: Any) -> float:
    """
    计算奖励分数 (0.0 或 1.0)
    Args:
        solution_str: 模型生成的完整文本
        ground_truth: 标准答案 (Algebrarium output)
    """
    # 1. 预处理：分离推理与答案
    answer_text = extract_answer_content(solution_str)
    
    try:
        # 2. 解析：math-verify 自动提取 \boxed{} 或最后数值
        gold_parsed = parse(ground_truth)
        # 添加 $ 符号有助于 math-verify 在混杂文本中定位数学内容
        answer_parsed = parse("$" + answer_text + "$")
        
        # 3. 验证：支持数值与代数符号的等价性比对
        if verify(gold_parsed, answer_parsed):
            return 1.0
    except Exception:
        return 0.0
    return 0.0
```

### 3. 调用流程示例

```python
if __name__ == "__main__":
    # 1. 准备数据 (来自 algebrarium 输出)
    item = {
        "question": "3 + (2 + 4) = ?",
        "answer": "4",
        "background": "[MATHEMATICAL DEFINITION]...",
    }

    # 2. 构建输入
    sys_prompt = "Your task is to solve... output the final answer within a LaTeX box format: \\boxed{answer}."
    full_prompt = f"{sys_prompt}\n\n{item['background']}\n\n[PROBLEM]\n{item['question']}"
    
    # 3. 获取模型输出 (模拟)
    model_response = """<think>
    Calculating 2+4 in Z_5 gives 1.
    Then 3+1 is 4.
    </think>
    The answer is \\boxed{4}."""

    # 4. 计算奖励
    score = compute_score(model_response, item['answer'])
    print(f"Reward: {score}") # 输出: 1.0
```


为了在 Verl (VolcEngine Reinforcement Learning) 框架中进行 PPO 或 GRPO 训练，你需要将 Algebrarium 生成的 JSONL 数据转换为 Parquet 格式，并符合 Verl 的字段规范。

为了观测更精细效果分析，可以将 data_source 字段设计为 “模版来源 + 难度等级” 的组合标签。

```python
import pandas as pd
import json
import argparse
import os

# 定义 System Prompt (强制要求 \boxed{})
SYSTEM_PROMPT = (
    "Your task is to solve mathematical problems based strictly on the provided definitions and rules.\n"
    "You must show your reasoning process step-by-step and output the final answer within a LaTeX box format: \\boxed{answer}."
)


def build_prompt_content(data):
    """构建 User 输入内容"""
    background = data.get('background', '')
    question = data['question']

    # 处理归纳模式 (Inductive) 的示例
    examples_str = ""
    if 'inductive_cases' in data and data['inductive_cases']:
        cases = [
            f"Example: {c['question']} -> {c['answer']}"
            for c in data['inductive_cases']
        ]
        examples_str = "[EXAMPLES]\n" + "\n".join(cases) + "\n\n"

    return f"{background}\n\n{examples_str}[PROBLEM]\n{question}"


def generate_data_source_tag(item):
    """
    生成组合标签: 来源_难度
    例如: alg_modulo_ops-5 (模运算模板，5步操作)
          alg_group_vars-3 (群方程组，3个变量)
    """
    # 1. 获取模板名称
    path = item.get('template_path', 'unknown')
    # 提取文件名去掉后缀: templates/groups/cyclic.yaml -> cyclic
    template_name = os.path.splitext(os.path.basename(path))[0]

    # 2. 获取难度指标
    # 如果是方程组，难度主要由变量数决定；否则由操作步数决定
    if item.get('category') == 'System of Group Equations':
        difficulty = f"vars-{item.get('n_variables', 0)}"
    else:
        difficulty = f"ops-{item.get('num_operate', 0)}"

    return f"alg_{template_name}_{difficulty}"


def main(input_file, output_file):
    rows = []
    print(f"正在处理: {input_file} ...")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)

            # 1. 构建 Prompt (此处模拟 Chat 格式，视模型 Tokenizer 而定)
            user_content = build_prompt_content(item)
            # 格式: <system>...<user>...<assistant> (Verl 通常支持纯文本或列表格式)
            full_prompt = [{
                "role": "system",
                "content": SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": user_content
            }]

            # 2. 生成组合 Data Source
            ds_tag = generate_data_source_tag(item)

            # 3. 构建数据行
            row = {
                "prompt": full_prompt,
                "ability": "math",
                "data_source": ds_tag,  # 关键：用于分组统计

                # RL 奖励验证所需信息
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(item['answer'])
                },

                # 保留原始信息供调试
                "extra_info": {
                    "template": item.get('template_path'),
                    "category": item.get('category')
                }
            }
            rows.append(row)

    # 4. 保存 Parquet
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_parquet(output_file)

    # 打印统计信息
    print(f"\n转换完成！数据已保存至 {output_file}")
    print(f"总样本数: {len(df)}")
    print("Data Source 分布示例:")
    print(df['data_source'].value_counts().head(5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        required=True,
                        help="Input .jsonl file")
    parser.add_argument("-o",
                        "--output",
                        required=True,
                        help="Output .parquet file")
    args = parser.parse_args()
    main(args.input, args.output)
```