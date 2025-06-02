# RISE

This repository contains the official implementation of [**Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing**](https://arxiv.org/abs/2410.06638) (ACL 2025).

## 🎯 Overview

RISE is a novel approach for improving reasoning capabilities in large language models through preference learning. The key innovation is the use of **error-injected self-editing** to create high-quality preference data that helps models learn to identify and correct subtle reasoning errors.

## Models

| Model Name | HF Checkpoint | Size | License |
|------------|---------------|------|---------|
| RISE-Qwen2-7B | 🤗 [kaishxu/RISE-Qwen2-7B](https://huggingface.co/kaishxu/RISE-Qwen2-7B) | 7B | [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/LICENSE) |

## 🚀 Quick Start

### Prerequisites

- vllm=0.5.4
- transformers=4.44.2
- trl=0.9.6
- [alignment-handbook](https://github.com/huggingface/alignment-handbook)

### Basic Usage

1. **Prepare Training Data**:
```bash
# Set data path
export data_path="./data/train"

# Generate self-sample prompts
python construct_self_sample_prompts.py \
    --model_name qwen2 \
    --save_prompt_path $data_path/self-sample-qwen2.jsonl

# Run sampling
bash scripts/sampling.sh

# Construct self-editing prompts
python construct_self_editing_prompts.py \
    --model_name qwen2 \
    --sample_folder_path $data_path/self-sample-qwen2-completion \
    --save_sample_path $data_path/self-sample-qwen2-dpo.json \
    --save_prompt_path $data_path/self-editing-qwen2-prompt.jsonl
```

2. **Generate Self-editing Completions**:
```bash
python inference.py \
    --model /path/to/Qwen2-7B-Instruct \
    --data_file $data_path/self-editing-qwen2-prompt.jsonl \
    --save_path $data_path/self-editing-qwen2-completion.json \
    --tensor_parallel_size 1 \
    --batch_size 10000
```

3. **Create Training Data**:
```bash
python construct_dpo_samples.py \
    --prompt_path $data_path/self-editing-qwen2-prompt.jsonl \
    --completion_path $data_path/self-editing-qwen2-completion.json \
    --chosen_sample_path $data_path/self-sample-qwen2-dpo-step.json \
    --full_sample_path $data_path/self-sample-qwen2-dpo.json \
    --save_sample_path $data_path/self-editing-qwen2-dpo.json
```

4. **Train the Model**:
```bash
bash scripts/train.sh
```

5. **Evaluate the Model**:
```bash
bash scripts/eval.sh
```

## 📊 Evaluation

The project includes comprehensive evaluation on mathematical reasoning tasks:

```bash
python eval_math.py \
    --model /path/to/trained/model \
    --data_path /path/to/test/data \
    --prompt qwen2-boxed \
    --save_path results.json
```

## 🤝 Thanks

Our training data is modified from [Step-DPO](https://github.com/dvlab-research/Step-DPO). Thanks for their great work!
