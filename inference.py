import argparse
import json
import pdb
import jsonlines
import os
import random

from vllm import LLM, SamplingParams
import torch
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def test_hendrycks_math(model, data_path, remainder=0, n_groups=MAX_INT, batch_size=1, tensor_parallel_size=1, args=None):
    
    save_path = args.save_path
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    attributes = []
    
    problem_prompt = (
        "{instruction}"
    )
    print(args.save_path)
    print('prompt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if "prefix" in item:
                temp_instr = problem_prompt.format(instruction=item["instruction"], prefix=item['prefix'])
            else:
                temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            if "answer" in item:
                temp_ans = [item["answer"]]
            else:
                temp_ans = []
            hendrycks_math_answers.append(temp_ans)
            attribute = {}
            if 'filepath' in item:
                attribute['filepath'] = item['filepath']
            if 'type' in item:
                attribute['type'] = item['type']
            if 'output' in item:
                attribute['gt_output'] = item['output']
            if 'idx' in item:
                attribute['idx'] = item['idx']
            attributes.append(attribute)

    args.seed = int((args.seed * random.random()*100) * random.random()*100) % 1000
    print("args.seed: ", args.seed)
    print('length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[remainder::n_groups]
    hendrycks_math_answers = hendrycks_math_answers[remainder::n_groups]
    attributes = attributes[remainder::n_groups]

    print("processed length ===", len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins * args.rep
    hendrycks_math_answers = hendrycks_math_answers * args.rep
    attributes = attributes * args.rep

    print('total length ===', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    sampling_params = SamplingParams(temperature=args.temp, top_p=args.top_p, max_tokens=2048)
    print('sampling =====', sampling_params)
    if not os.path.exists(save_path):
        llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, dtype=torch.bfloat16, seed=args.seed)

        res_completions = []
        for idx, (prompt, prompt_answer) in enumerate(zip(batch_hendrycks_math_ins, hendrycks_math_answers)):
            if isinstance(prompt, list):
                pass
            else:
                prompt = [prompt]
            completions = llm.generate(prompt, sampling_params)
            for output in completions:
                prompt_temp = output.prompt
                generated_text = output.outputs[0].text
                res_completions.append(generated_text)
    else:
        res_completions = []
        with open(save_path) as f:
            items = json.load(f)
        for idx, item in enumerate(items):
            res_completions.append(item['completion'])

    to_save_list = []
    for idx, (prompt, completion, prompt_answer, attribute) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers, attributes)):

        item = {
            'question': prompt,
            'model_output': completion,
            'answer': prompt_answer if isinstance(prompt_answer, list) else [prompt_answer],
        }

        to_save_dict = {
            'prompt': prompt,
            'completion': completion,
            'answer': prompt_answer,
        }
        to_save_dict.update(attribute)
        to_save_list.append(to_save_dict)

    print('len invalid outputs ====', len(invalid_outputs))
    print('n_groups===', n_groups, ', remainder====', remainder)

    try:
        with open(save_path, "w+") as f:
            json.dump(to_save_list, f, indent=4)
    except:
        import pdb; pdb.set_trace()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--remainder", type=int, default=0) # index
    parser.add_argument("--n_groups", type=int, default=1)  # group number
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rep", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(model=args.model, data_path=args.data_file, remainder=args.remainder, n_groups=args.n_groups, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size, args=args)
