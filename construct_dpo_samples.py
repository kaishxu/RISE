import os
import re
import json
import argparse
from tqdm import trange
import random

def check_is_error(prompt, text):
    if "wrong" in prompt or "correct" in prompt:
        return False

    error_lst = ["a mistake", "mistakenly", "misleading", "Misleading", "misinterpreted", "Let's correct", "let's correct", "The correct", "Let's reconsider", "let's reconsider", "incorrect", "wrong", "python", "This is not correct", "this is not correct"]
    for error in error_lst:
        if error in text:
            return True
    return False

def construct(prompt_path, completion_path, chosen_sample_path, full_sample_path, save_sample_path):
    with open(prompt_path, "r") as infile:
        prompt_lst = [json.loads(x) for x in infile]
    with open(completion_path, "r") as infile:
        response_lst = json.load(infile)
    with open(chosen_sample_path, "r") as infile:
        ds = json.load(infile)

    correct_response_lst = []
    correct_idx_lst = []
    for idx in trange(len(response_lst), desc="Extract edited responses"):
        response = response_lst[idx]["completion"]
        response = "```\n" + response

        tmp = re.findall(r"```\n(.+?)\n```", response, flags=re.DOTALL)

        if len(tmp) > 0:
            x = tmp[0]
            if not check_is_error(ds[idx]["prompt"], x):
                response_lst[idx]["edit"] = x
                correct_response_lst.append(response_lst[idx])
                correct_idx_lst.append(response_lst[idx]["idx"])

    wrong_idx_lst = []
    for idx in range(len(response_lst)):
        if response_lst[idx]["idx"] not in correct_idx_lst:
            wrong_idx_lst.append(response_lst[idx]["idx"])

    correct_response_lst.sort(key=lambda x: x["idx"])
    response_dict = {x["idx"]: x for x in correct_response_lst}

    new_ds = []
    for idx in trange(len(ds), desc="Constructing samples"):
        new_ds.append(ds[idx])
        new_ds[-1]["rejected"] = new_ds[-1]["chosen"]

        if idx in response_dict:
            # text_to_edit = re.findall(r"Text:\n```\n(.+?)\n```", response_dict[idx]["prompt"], flags=re.DOTALL)
            assert prompt_lst[idx]["text_to_edit"] in response_dict[idx]["prompt"]
            new_ds[-1]["rejected"] = new_ds[-1]["rejected"].replace(
                prompt_lst[idx]["text_to_edit"],
                response_dict[idx]["edit"]
            )
        else:
            del new_ds[-1]

    for idx in range(len(new_ds)):

        tmp = re.findall(r"( \\boxed\{.+?\}\.)", new_ds[idx]["rejected"])
        if len(tmp) > 0:
            for x in tmp:
                new_ds[idx]["rejected"] = new_ds[idx]["rejected"].replace(x, " "+x[8:-2]+".")

        tmp = re.findall(r"(\\\(\\boxed\{.+?\}\\\))", new_ds[idx]["rejected"])
        if len(tmp) > 0:
            for x in tmp:
                new_ds[idx]["rejected"] = new_ds[idx]["rejected"].replace(x, x[9:-3])

        tmp = re.findall(r"(\\\[\\boxed\{.+?\}\\\])", new_ds[idx]["rejected"])
        if len(tmp) > 0:
            for x in tmp:
                new_ds[idx]["rejected"] = new_ds[idx]["rejected"].replace(x, x[9:-3])

        tmp = re.findall(r"( \\boxed\{.+?\}$)", new_ds[idx]["rejected"])
        if len(tmp) > 0:
            for x in tmp:
                new_ds[idx]["rejected"] = new_ds[idx]["rejected"].replace(x, " "+x[8:-1])

        tmp = re.findall(r"(\\\( \\boxed\{.+?\} \\\))", new_ds[idx]["rejected"])
        if len(tmp) > 0:
            for x in tmp:
                new_ds[idx]["rejected"] = new_ds[idx]["rejected"].replace(x, x[10:-4])

        tmp = re.findall(r"(\\\[\n\\boxed\{.+?\}\n\\\])", new_ds[idx]["rejected"])
        if len(tmp) > 0:
            for x in tmp:
                new_ds[idx]["rejected"] = new_ds[idx]["rejected"].replace(x, x[10:-4])

        tmp = re.findall(r"(\$\\boxed\{.+?\}\$)", new_ds[idx]["rejected"])
        if len(tmp) > 0:
            for x in tmp:
                new_ds[idx]["rejected"] = new_ds[idx]["rejected"].replace(x, x[8:-2])

        x = new_ds[idx]["rejected"]
        while True:
            try:
                sep = "\\boxed{"
                start = x.index(sep)
                num_bracket = 1  #sep contatin 1 bracket
                for i in range(start+len(sep), len(x)):
                    if x[i] == "{":
                        num_bracket += 1
                    elif x[i] == "}":
                        num_bracket -= 1
                    if num_bracket == 0:
                        break
                x = x.replace(x[start:i+1], x[start:i+1][7:-1])
            except:
                break
        new_ds[idx]["rejected"] = x

    # merge solution pair and step pair
    from collections import defaultdict

    step_pair_dict = defaultdict(list)
    for sample in new_ds:
        step_pair_dict[sample["prompt"]].append(sample)

    with open(full_sample_path, "r") as infile:
        full_sample_lst = json.load(infile)

    new_ds = []
    num_edit = 3
    for idx in range(len(full_sample_lst)):
        prompt = full_sample_lst[idx]["prompt"]
        new_ds.append(full_sample_lst[idx])
        if prompt in step_pair_dict:
            # if len(step_pair_dict[prompt]) >= num_edit:
            #     edit_sample = random.sample(step_pair_dict[prompt], num_edit)
            # else:
            #     edit_sample = step_pair_dict[prompt]
            # new_ds += edit_sample

            edit_sample = random.choice(step_pair_dict[prompt])
            new_ds.append(edit_sample)

    print("Total samples:", len(new_ds))

    with open(save_sample_path, "w") as outfile:
        json.dump(new_ds, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, default='')
    parser.add_argument("--completion_path", type=str, default='')
    parser.add_argument("--chosen_sample_path", type=str, default='')
    parser.add_argument("--full_sample_path", type=str, default='')
    parser.add_argument("--save_sample_path", type=str, default='')
    args = parser.parse_args()

    # construct samples for DPO
    construct(args.prompt_path,
            args.completion_path, 
            args.chosen_sample_path,
            args.full_sample_path,
            args.save_sample_path)