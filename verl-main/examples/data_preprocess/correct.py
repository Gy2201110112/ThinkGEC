# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

from datasets import Dataset, load_dataset

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


# def extract_solution(solution_str):
#     return remove_boxed(last_boxed_only_string(solution_str))

# def make_prefix(dp, template_type):
#     quiz = dp['quiz']
#     if template_type == 'base':
#         prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
#     elif template_type == 'qwen-instruct':
#         prefix = f"""You are an expert in German language education and I will provide you with a sentence from a student's essay as an assistant, which may contain errors in grammar, spelling, etc. Please take on the role of a teacher, revise the student's composition, and answer me only to correct the correct sentence. Please do not give any unnecessary explanations.\nfor example:\nEs gibt drei Personen in meine Familie.\nEs gibt drei Personen in meiner Familie.\n\ninput:{quiz}"""
#     return prefix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/gy/verl/data/correct')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='/home/gy/verl/data/grpo.jsonl')
    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=346)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')

    args = parser.parse_args()
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)

    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(len(raw_dataset))

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "zju/yuyu"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = "You are an expert in German language education and I will provide you with a sentence from a student's essay as an assistant, which may contain errors in grammar, spelling, etc. Please take on the role of a teacher, revise the student's composition, and answer me only to correct the correct sentence. Please do not give any unnecessary explanations.\nfor example:\nEs gibt drei Personen in meine Familie.\nEs gibt drei Personen in meiner Familie."
            question = example['quiz']
            solution = example.pop("solution")
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question}
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
