import os
import json
import pickle
import argparse
from tqdm import trange
from datasets import load_dataset

def load_questions():

    if os.path.exists("questions_info.pk"):
        with open("questions_info.pk", "rb") as infile:
            question_dict, question_lst = pickle.load(infile)
    else:
        # load dataset from "xinlai/Math-Step-DPO-10K"
        ds = load_dataset("Math-Step-DPO-10K")

        question_dict = {}
        question_lst = []
        for idx in trange(len(ds["train"])):

            if ds["train"][idx]["prompt"] not in question_dict:
                question_dict[ds["train"][idx]["prompt"]] = (idx, ds["train"][idx]["dataset"], ds["train"][idx]["answer"])
                question_lst.append(ds["train"][idx]["prompt"])

        with open("questions_info.pk", "wb") as outfile:
            pickle.dump((question_dict, question_lst), outfile)

    return question_dict, question_lst

def construct(model_name, save_prompt_path):

    # load questions
    question_dict, question_lst = load_questions()

    if model_name == "qwen2":
        prefix_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\nLet's think step by step.\nStep 1: "
        )
    elif model_name == "llama3.1":
        prefix_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023"
            "\nToday Date: 26 Jul 2024\n\n"
            "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "Let's think step by step.\nStep 1: "
        )
    elif model_name == "mistral":
        prefix_prompt = (
            "<s>[INST]{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.[/INST]"
            "Let's think step by step.\nStep 1: "
        )

    prompt_lst = []
    for idx in trange(len(question_lst)):
        q = question_lst[idx]
        prompt = prefix_prompt.format(instruction=q)
        prompt_lst.append({
                "idx": question_dict[q][0],
                "instruction": prompt,
                "answer": question_dict[q][2],
            })

    with open(save_prompt_path, "w") as outfile:
        for prompt in prompt_lst:
            json.dump(prompt, outfile)
            outfile.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='', choices=["qwen2", "llama3.1", "mistral"])
    parser.add_argument("--save_prompt_path", type=str, default='')
    args = parser.parse_args()

    # construct prompts for self-sampling
    construct(args.model_name, args.save_prompt_path)