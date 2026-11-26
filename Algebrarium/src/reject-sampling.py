import openai
from math_verify import parse, verify
import pandas as pd
from tqdm import tqdm
import argparse
import concurrent.futures
import logging
import json
import time
import subprocess
import requests
# 配置日志记录
GENERATION_TEMPLATE = "\n\n[[Start Case]]\n\nQuestion:\n\n{question}\n\nResponse:\n\n"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
        response = client.completions.create(model=model_name,
                                             prompt=prompt,
                                             max_tokens=max_tokens,
                                             temperature=temperature,
                                             top_p=top_p,
                                             n=n_samples,
                                             stop=["[[End Case]]"])
        return list(map(lambda x: x.text, response.choices))
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


def fetch_responses_concurrently(row, args, client, idx2template):
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
        prompt=idx2template[row['template_path']].strip() +
        GENERATION_TEMPLATE.format(question=row['question']),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_name=args.model_name,
        client=client,
        top_p=args.top_p,
        n_samples=args.n_samples,
    )
    # 将原始数据与结果一起返回，以便主进程进行后续处理
    return row, responses


def wait_for_server_to_start(host, port, model_name, timeout=600):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            url = f"http://{host}:{port}/v1/completions"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model_name,
                "prompt": "San Francisco is a",
                "max_tokens": 7,
                "temperature": 0,
            }
            response = requests.post(url,
                                     headers=headers,
                                     data=json.dumps(data))
            if response.status_code == 200:
                print("Server started successfully!")
                return True
        except requests.exceptions.RequestException as e:
            print(f"Waiting for server... {e}")
        time.sleep(1)
    print("Server failed to start within the timeout period.")
    return False


def start_vllm_server(
    model_path,
    port,
    max_model_len,
    tensor_parallel_size,
    api_key="Empty",
):
    subprocess.Popen([
        "vllm",
        "serve",
        model_path,
        "--max-model-len",
        str(max_model_len),
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ])

    wait_for_server_to_start("127.0.0.1", port, model_path)
    return openai.OpenAI(base_url=f"http://127.0.0.1:{port}/v1",
                         api_key=api_key)


def main():
    """
    主函数，用于解析命令行参数、并发获取数据并在主进程中进行验证。
    """
    parser = argparse.ArgumentParser(
        description="并发获取 GSM8K 数据集结果，并在主进程中进行验证。")

    parser.add_argument("--input_file",
                        type=str,
                        default="./data/algebrarium-warmup/warmup.parquet",
                        help="输入 Parquet 文件的路径。")
    parser.add_argument("--template_path",
                        type=str,
                        default="./data/algebrarium-warmup/warmup.config.json",
                        help="输入 Parquet 文件的路径。")
    parser.add_argument("--output_file",
                        type=str,
                        default="gsm8k.llama3.2-1b.parquet",
                        help="输出 Parquet 文件的路径。")
    parser.add_argument("--base_url",
                        type=str,
                        default="http://127.0.0.1:9999/v1",
                        help="OpenAI API 的基础 URL。")
    parser.add_argument("--port",
                        type=int,
                        default=9999,
                        help="OpenAI API 的基础 port")
    parser.add_argument("--api_key",
                        type=str,
                        default="null",
                        help="OpenAI API 的密钥。")
    parser.add_argument("--model_name",
                        type=str,
                        default="llama3.2-1b",
                        help="要使用的模型名称。")
    parser.add_argument("--n_samples",
                        type=int,
                        default=16,
                        help="每个提示生成的样本数。")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.6,
                        help="API 请求的 temperature 参数。")
    parser.add_argument("--top_p",
                        type=float,
                        default=1.0,
                        help="API 请求的 top_p 参数。")
    parser.add_argument("--max_tokens",
                        type=int,
                        default=2048,
                        help="API 请求的 max_tokens 参数。")
    parser.add_argument("--max_workers",
                        type=int,
                        default=128,
                        help="并发执行的最大工作线程数。")
    args = parser.parse_args()

    logging.info(f"正在从 {args.input_file} 读取数据...")

    try:
        df = pd.read_parquet(args.input_file)
    except FileNotFoundError:
        logging.error(f"输入文件未找到: {args.input_file}")
        return

    #client = openai.Client(base_url=args.base_url, api_key=args.api_key)
    client = start_vllm_server(model_path=args.model_name,
                               port=args.port,
                               max_model_len=30000,
                               tensor_parallel_size=1,
                               api_key=args.api_key)
    data = []
    with open(args.template_path, 'r') as f:
        idx2template = json.load(f)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers) as executor:
        # 提交并发任务：仅获取 API 响应
        futures = [
            executor.submit(fetch_responses_concurrently, row, args, client,
                            idx2template) for _, row in df.iterrows()
        ]

        logging.info("所有 API 请求已提交，等待响应并进行处理...")
        # 主进程等待任务完成，并串行处理结果
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(df),
                           desc="处理进度"):
            try:
                # original_row 是原始数据行, responses 是 API 返回的文本列表
                original_row, responses = future.result()

                # --- 以下部分在主进程中串行执行 ---
                for r in responses:
                    # 调用非线程安全的函数
                    if accuracy_reward(r, original_row['answer']):

                        item = original_row.to_dict()
                        item['prompt'] = [{
                            "role":
                            'system',
                            "content":
                            "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The final answer MUST BE put in \\boxed{}."
                        }, {
                            'role':
                            'user',
                            'content':
                            original_row['background'] + "\n\n" +
                            original_row['question']
                        }]
                        item['response'] = r
                        data.append(item)
                        break
            except Exception as e:
                logging.error(f"处理返回结果时发生错误: {e}")

    if data:
        logging.info(f"处理完成，共找到 {len(data)} 个有效结果。")
        pd.DataFrame(data).to_parquet(args.output_file)
        logging.info(f"结果已保存到 {args.output_file}")
    else:
        logging.warning("没有找到任何有效的答案。")


if __name__ == "__main__":
    main()
