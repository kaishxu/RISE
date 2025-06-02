import os
import sys
import re
import json
import argparse
import pickle
import random
from tqdm import trange
from datasets import load_dataset

def load_questions():

    if os.path.exists("questions_info.pk"):
        with open("questions_info.pk", "rb") as infile:
            question_dict, question_lst = pickle.load(infile)
    else:
        # load dataset from "xinlai/Math-Step-DPO-10K"
        ds = load_dataset("xinlai/Math-Step-DPO-10K")

        question_dict = {}
        question_lst = []
        for idx in trange(len(ds["train"])):

            if ds["train"][idx]["prompt"] not in question_dict:
                question_dict[ds["train"][idx]["prompt"]] = (idx, ds["train"][idx]["dataset"], ds["train"][idx]["answer"])
                question_lst.append(ds["train"][idx]["prompt"])

        with open("questions_info.pk", "wb") as outfile:
            pickle.dump((question_dict, question_lst), outfile)

    return question_dict, question_lst

def load_file_paths(folder_path):
    file_paths = []

    # check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return file_paths

    # walk through the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    file_paths.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return file_paths

def eval_completion(file_paths):
    print("Evaluating the completions...")
    for path in file_paths:
        print(path)
        with open(path, "r") as infile:
            tmp_completion = json.load(infile)

        if "result" in tmp_completion[0]:
            continue

        # excute the evaluation script
        os.system(f"python eval_results.py --data_file {path} --save_path {path}")

def load_eval_results(model_name, file_paths):

    # sep for different models
    if model_name == "qwen2":
        sep = "<\|im_start\|>user\n"
    elif model_name == "llama3.1":
        sep = "user<\|end_header_id\|>\n\n"
    elif model_name == "mistral":
        sep = "You are a helpful assistant.\n\n"

    result_dict = {}
    for path in file_paths:

        with open(path, "r") as infile:
            tmp_completion = json.load(infile)

        for idx in range(len(tmp_completion)):
            q = re.findall(sep + "(.+?)\nPlease reason step by step", tmp_completion[idx]["prompt"], flags=re.DOTALL)[0]
            if q not in result_dict:
                result_dict[q] = []
            result_dict[q].append((tmp_completion[idx]["prompt"] + tmp_completion[idx]["completion"], tmp_completion[idx]["result"]))

    return result_dict

def check_is_error(prompt, text):

    # 没有按照步骤来，或步骤不连续
    # tmp = re.findall(r"\nStep (\d+):", text)
    # if not (tmp != [] and len(tmp) == int(tmp[-1])):
    #     return True

    # 没有遵循格式
    # if "\\boxed{" not in text:
    #     return True

    # 如果题目涉及到"wrong"和"correct"等词，直接跳过不管
    if "wrong" in prompt or "correct" in prompt:
        return False

    error_lst = ["a mistake", "misunderstanding", "misleading", "misinterpreted", "Let's correct", "let's correct", "The correct", "Let's reconsider", "let's reconsider", "python", "This is not correct", "this is not correct"]
    for error in error_lst:
        if error in text:
            return True

    return False

def construct_samples_w_chosen(model_name, file_paths, save_sample_path):
    # sep for different models
    if model_name == "qwen2":
        sep = "<|im_start|>assistant\n"
    elif model_name == "llama3.1":
        sep = "assistant<|end_header_id|>\n\n"
    elif model_name == "mistral":
        sep = "[/INST]"
    # extract_phrase = "answer is"

    # load questions
    question_dict, question_lst = load_questions()

    # load the completion results
    print("Loading the evaluation results...")
    result_dict = load_eval_results(model_name, file_paths)

    num_samples = 5
    pair_lst = []
    step_pair_lst = []
    for q in result_dict:

        pos_sample_lst = [x[0].split(sep)[1] for x in result_dict[q][:num_samples] if x[1] and not check_is_error(q, x[0].split(sep)[1])]
        neg_sample_lst = [x[0].split(sep)[1] for x in result_dict[q][:num_samples] if not x[1]]

        if pos_sample_lst != [] and neg_sample_lst != []:
            pos_text = random.choice(pos_sample_lst)
            neg_text = random.choice(neg_sample_lst)
            pair_lst.append({
                "idx": question_dict[q][0],
                "dataset": question_dict[q][1],
                "prompt": q,
                "initial_reason_steps": "",
                "chosen": pos_text,
                "rejected": neg_text,
                "answer": question_dict[q][2],
            })

            completion = pos_text
            question_info = question_dict[q]
            tmp = re.findall(r"\nStep \d+:", completion)
            for i in range(len(tmp)):
                step_pair = {
                    "dataset": question_info[1],
                    "prompt": q,
                    "answer": question_info[2],
                    "step": tmp[i][6:-1],
                    "last_step": i == len(tmp)-1,
                    }
                pos = completion.index(tmp[i]) + len(tmp[i])
                step_pair["initial_reason_steps"] = completion[:pos]
                if i+1 < len(tmp):
                    end = completion.index(tmp[i+1])
                    step_pair["chosen"] = completion[pos:end]
                else:
                    end = len(completion)
                    step_pair["chosen"] = completion[pos:end]
                step_pair_lst.append(step_pair)

    with open(save_sample_path, "w") as outfile:
        json.dump(pair_lst, outfile, indent=4)
    save_step_sample_path = save_sample_path.replace(".json", "-step.json")
    with open(save_step_sample_path, "w") as outfile:
        json.dump(step_pair_lst, outfile, indent=4)
    return pair_lst, step_pair_lst

prompt_change_value = '''Question:
```
{question}
```

Initial Answer:
```
{answer}
```

Current Step:
```
{text}
```

Edit a numerical value in the current step to make a wrong calculation. Do not state that errors have been made.'''

prompt_change_multi_value = '''Question:
```
{question}
```

Initial Answer:
```
{answer}
```

Current Step:
```
{text}
```

Edit a numerical value or a series of related values in the current step to make a wrong calculation. Do not state that errors have been made.'''

prompt_wrong_sub = '''Question:
```
{question}
```

Initial Answer:
```
{answer}
```

Current Step:
```
{text}
```

Edit a value or symbol in the current step to make a wrong substitution. Do not state that errors have been made.'''

prompt_del_term = '''Question:
```
{question}
```

Initial Answer:
```
{answer}
```

Current Step:
```
{text}
```

Delete a calculation term in the current step to make a wrong calculation. Do not state that errors have been made.'''

prompt_change_adjective = '''Question:
```
{question}
```

Initial Answer:
```
{answer}
```

Current Step:
```
{text}
```

Edit a series of related words in the current step to reflect the misunderstanding of the question. Do not state that errors have been made.'''

prompt_change_symbol = '''Question:
```
{question}
```

Initial Answer:
```
{answer}
```

Current Step:
```
{text}
```

Edit a calculation symbol (e.g., +-*/, etc.) in the current step to make a wrong calculation. Do not state that errors have been made.'''

prompt_swap = '''Question:
```
{question}
```

Initial Answer:
```
{answer}
```

Current Step:
```
{text}
```

Swap two calculation terms in the current step to make a wrong calculation. Do not state that errors have been made.'''

def construct(model_name, save_prompt_path, ds):

    if model_name == "qwen2":
        prefix_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        )
    elif model_name == "llama3.1":
        prefix_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023"
            "\nToday Date: 26 Jul 2024\n\n"
            "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif model_name == "mistral":
        prefix_prompt = (
            "<s>[INST] You are a helpful assistant.\n\n"
            "{instruction}[/INST]"
        )

    sentence_dict = {}
    prompt_lst = []
    for idx in trange(len(ds), desc="Constructing prompts"):

        gt_response = ds[idx]["chosen"]

        sentence_lst = [x for x in gt_response.split("\n") if x != ""]  #粗略按行切分
        cleaned_sentence_lst = []
        for sentence in sentence_lst:
            # manual split
            search = re.finditer(r"\. ", sentence)
            pos_lst = [(0, 0)]
            for x in search:
                pos_lst.append(x.span())

            for i in range(1, len(pos_lst)):
                cleaned_sentence_lst.append(sentence[pos_lst[i-1][1]:pos_lst[i][1]])
            cleaned_sentence_lst.append(sentence[pos_lst[-1][1]:])

        final_sentence_lst = [x.strip() for x in cleaned_sentence_lst if x.strip() != ""]
        sentence_dict[idx] = final_sentence_lst

        if len(final_sentence_lst) > 3:
            tmp = final_sentence_lst[-4]
            start_idx = gt_response.index(tmp)
            gt_response = gt_response[start_idx:]

        gt_response = gt_response.strip()
        candidate_edit = []
        # 增加value error
        contain_digit = len(re.findall(r'\d+', gt_response))
        if contain_digit > 1:
            candidate_edit += [prompt_change_multi_value.format(question=ds[idx]["prompt"], answer=ds[idx]["initial_reason_steps"], text=gt_response)] * 1
        elif contain_digit == 1:
            candidate_edit += [prompt_change_value.format(question=ds[idx]["prompt"], answer=ds[idx]["initial_reason_steps"], text=gt_response)] * 1

        # 增加symbol error
        if any([ch in gt_response for ch in ["="]]):
            candidate_edit += [prompt_wrong_sub.format(question=ds[idx]["prompt"], answer=ds[idx]["initial_reason_steps"], text=gt_response)] * 1
            candidate_edit += [prompt_del_term.format(question=ds[idx]["prompt"], answer=ds[idx]["initial_reason_steps"], text=gt_response)] * 1
            candidate_edit += [prompt_change_symbol.format(question=ds[idx]["prompt"], answer=ds[idx]["initial_reason_steps"], text=gt_response)] * 1
            candidate_edit += [prompt_swap.format(question=ds[idx]["prompt"], answer=ds[idx]["initial_reason_steps"], text=gt_response)] * 1

        # 增加adjective error
        if not (contain_digit > 0 or any([ch in gt_response for ch in ["="]])):
            candidate_edit += [prompt_change_adjective.format(question=ds[idx]["prompt"], answer=ds[idx]["initial_reason_steps"], text=gt_response)] * 1

        prompt = random.choice(candidate_edit)

        prompt_lst.append({
            "idx": idx,
            "instruction": prefix_prompt.format(instruction=prompt) + "The edited Current Step:\n```\n",
            "text_to_edit": gt_response
        })

    with open(save_prompt_path, "w") as outfile:
        for prompt in prompt_lst:
            json.dump(prompt, outfile)
            outfile.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='', choices=["qwen2", "llama3.1", "mistral"])
    parser.add_argument("--sample_folder_path", type=str, default='')
    parser.add_argument("--save_sample_path", type=str, default='')  # save the constructed samples for training
    parser.add_argument("--save_prompt_path", type=str, default='')  # save the constructed prompts for self-editing
    args = parser.parse_args()

    file_paths = load_file_paths(args.sample_folder_path)
    eval_completion(file_paths)
    pair_lst, step_pair_lst = construct_samples_w_chosen(args.model_name, file_paths, args.save_sample_path)

    # construct prompts for self-editing
    construct(args.model_name, args.save_prompt_path, step_pair_lst)