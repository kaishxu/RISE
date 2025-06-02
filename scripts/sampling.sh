# qwen2-7b
export output_dir="data/train/self-sample-qwen2-completion"
export prompt="qwen2-boxed"

mkdir -p $output_dir

for times in $(seq 0 7)
do

for seq in $(seq 0 7)
do

(output_idx=$[$seq + 1 + 8 * $times]
echo $output_idx
CUDA_VISIBLE_DEVICES=$seq python inference.py --model /code/models/Qwen2-7B-Instruct --data_file ./data/train/self-sample-qwen2.jsonl --save_path $output_dir/$output_dir-$output_idx.json --prompt $prompt --tensor_parallel_size 1 --batch_size 10000 --temp 0.2 --top_p 0.9) &

done

wait

done