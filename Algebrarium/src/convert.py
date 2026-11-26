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
