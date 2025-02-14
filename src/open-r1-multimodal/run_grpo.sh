export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH=$PYTHONPATH:./src
export CUDA_LAUNCH_BLOCKING=1 

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir output/ \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name leonardPKU/clevr_cogen_a_train \
    --deepspeed local_scripts/zero2.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 2 \
    --use_peft True \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05