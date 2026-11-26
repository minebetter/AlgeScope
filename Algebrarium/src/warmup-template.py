import openai
from math_verify import parse, verify
import pandas as pd
from tqdm import tqdm
import argparse
import concurrent.futures
import logging

CASE_TEMPLATE = "[[Start Case]]\n\nQuestion:\n\n{question}\n\nResponse:\n\n{response}\n\n[[End Case]]"
#GENERATION_TEMPLATE = "[[Start Case]]\n\nQuestion:\n\n{question}\n\nResponse:\n\n"
BACKGROUND_TEMPLATE = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The final answer MUST BE put in \\boxed{{}}.\n\n{background}"


def request_api(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    client: openai.OpenAI,
    model_name: str,
    n_samples: int = 1,
):
    """
    向 OpenAI API 发送请求并获取响应。
    （此函数在工作线程中并发执行）
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n_samples,
        )
        return list(map(lambda x: x.message.content, response.choices))[0]
    except Exception as e:
        logging.error(f"调用 API 时出错: {e}")
        return []


def accuracy_reward(predict: str, ground_truth: str) -> bool:
    """
    验证预测答案是否正确。
    （此函数现在只在主进程中串行执行）
    """
    try:
        return verify(parse("$" + ground_truth + "$"), parse(predict)) or \
               verify(parse(ground_truth), parse(predict))
    except (OverflowError, Exception):
        return False


def fetch_responses_concurrently(row, args, client):
    """
    并发任务单元：仅负责为单行数据调用 API 并返回结果。
    
    Args:
        row (pd.Series): DataFrame 的一行数据。
        args (argparse.Namespace): 命令行参数。
        client (openai.OpenAI): OpenAI API 客户端实例。

    Returns:
        tuple: 包含原始行数据和 API 返回的响应列表 (row, responses)。
    """
    responses = request_api(
        prompt=row['prompt'],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_name=args.model_name,
        client=client,
        top_p=args.top_p,
        n_samples=args.n_samples,
    )
    # 将原始数据与结果一起返回，以便主进程进行后续处理
    return row, responses


def main():
    """
    主函数，用于解析命令行参数、并发获取数据并在主进程中进行验证。
    """
    parser = argparse.ArgumentParser(
        description="并发获取 GSM8K 数据集结果，并在主进程中进行验证。")
    parser.add_argument("--input_dir",
                        type=str,
                        default="./data/warmup",
                        help="输入 Parquet 文件的路径。")
    parser.add_argument("--output_file",
                        type=str,
                        default="gsm8k.llama3.2-1b.parquet",
                        help="输出 Parquet 文件的路径。")
    parser.add_argument("--base_url",
                        type=str,
                        default="http://127.0.0.1:9999/v1",
                        help="OpenAI API 的基础 URL。")
    parser.add_argument("--api_key",
                        type=str,
                        default="null",
                        help="OpenAI API 的密钥。")
    parser.add_argument("--model_name",
                        type=str,
                        default="llama3.2-1b",
                        help="要使用的模型名称。")
    parser.add_argument("--n_samples", type=int, default=1, help="每个提示生成的样本数。")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.4,
                        help="API 请求的 temperature 参数。")
    parser.add_argument("--top_p",
                        type=float,
                        default=1.0,
                        help="API 请求的 top_p 参数。")
    parser.add_argument("--max_tokens",
                        type=int,
                        default=4096,
                        help="API 请求的 max_tokens 参数。")
    parser.add_argument("--max_workers",
                        type=int,
                        default=128,
                        help="并发执行的最大工作线程数。")
    args = parser.parse_args()

    logging.info(f"正在从 {args.input_dir} 读取数据...")
    client = openai.Client(base_url=args.base_url, api_key=args.api_key)
    import pathlib

    result_templates = []

    for file in pathlib.Path(args.input_dir).glob("*.jsonl"):
        print(file)
        all_cases = pd.read_json(file, lines=True)
        for idx, row in all_cases.iterrows():
            prompt = BACKGROUND_TEMPLATE.format(
                background=row['background']) + "\n\n" + row['question']

            complete = False
            result_cases = []
            while not complete:
                #response = request_api(
                #    prompt=prompt,
                #    temperature=args.temperature,
                #    max_tokens=args.max_tokens,
                #    model_name=args.model_name,
                #    client=client,
                #    top_p=args.top_p,
                #    n_samples=args.n_samples,
                #)
                response = "$" + row['answer'] + "$"
                if accuracy_reward(response, row['answer']):
                    complete = True
                    result_cases.append(
                        CASE_TEMPLATE.format(question=row['question'],
                                             response=response))

        template = BACKGROUND_TEMPLATE.format(
            background=row['background']) + "\n\n".join(result_cases)

        item = row.to_dict()
        item.pop("question")
        item.pop("answer")
        item['warmup-template'] = template
        result_templates.append(item)

    pd.DataFrame(result_templates).to_parquet(args.output_file)


if __name__ == "__main__":
    main()
