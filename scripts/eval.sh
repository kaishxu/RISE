
export model_path="RISE-Qwen2-7B"
export prompt="qwen2-boxed"
export result_dir="eval_results"

# GSM8K
python eval_math.py --model $model_path --data_file ./data/test/GSM8K_test_data.jsonl --save_path $result_dir/RISE-Qwen2-7B.json --prompt $prompt --tensor_parallel_size 1 --batch_size 1000

# MATH
python eval_math.py --model $model_path --data_file ./data/test/MATH_test_data.jsonl --save_path $result_dir/RISE-Qwen2-7B.json --prompt $prompt --tensor_parallel_size 1 --batch_size 1000
