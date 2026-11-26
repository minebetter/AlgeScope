#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import sys


def format_write(item):
    question = item['question']
    background = item['background']
    answer = item['answer']
    category = item['category']
    formatted_text=f"{background}\n\n{"[Question]"}\n\nQ: {question}\nA: {answer}\nC: {category}"
    print(formatted_text)


def data_print(input_file):
    """
    Reads data from a JSON input file, pretty-prints it,
    and saves the result to an output file.

    Args:
        input_file (str): The path to the input JSON file.
        output_file (str): The path to the output file.
    """

    print(f"[*] Reading data from: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data.append(json.loads(line))
    for item in data:
        format_write(item)


def main():
    """
    主函数，用于解析命令行参数并调用核心功能。
    """
    # 创建一个参数解析器
    parser = argparse.ArgumentParser(
        description="A script to read, pretty-print, and save JSON data.",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )

    # 添加 --input-file 参数
    parser.add_argument('-i',
                        '--input-file',
                        required=True,
                        type=str,
                        help="Path to the input JSON file. (Required)")


    # 解析命令行传入的参数
    args = parser.parse_args()

    # 调用核心处理函数
    data_print(args.input_file)


if __name__ == "__main__":
    main()
