
export output_dir="RISE-Qwen2-7B"
export prompt="qwen2-boxed"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/deepspeed_zero3.yaml --mixed_precision bf16 \
    --num_processes 8 \
    train.py configs/config_full.yaml \
    --model_name_or_path="/code/models/Qwen2-7B-Instruct" \
    --data_path="RISE-Qwen2-7B-train.json" \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --torch_dtype=bfloat16 \
    --bf16=True \
    --beta=0.4 \
    --num_train_epochs=8 \
    --save_strategy='steps' \
    --save_steps=100 \
    --save_total_limit=10 \
    --output_dir=outputs/$output_dir \
    --hub_model_id=$output_dir \
    --prompt=$prompt
