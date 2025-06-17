## Environment Setup

We use Python 3.10.9 and PyTorch 2.0.0 for our experiments. Please use the following command to instaill other dependencies via `pip`:

pip install -r requirements.txt


## Attacks in NLP Tasks
Use the following command to enter the `nlp` folder:

```Shell
cd nlp
```

Then use the following command to run the backdoor attack on the Emotion dataset with the LLaMA-7B model and 10% poisoning ratio:

```Shell

deepspeed --include localhost:0 backdoor_train.py  --deepspeed ds_config.json \
--model_name_or_path your_model_path \
    --output_dir ./outputs/llama-7b_emotion_backdoor_random_p10 \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 1 \
    --evaluation_strategy epoch \
    --eval_dataset_size 1000 \
    --max_eval_samples 100 \
    --max_test_samples 1000 \
    --per_device_eval_batch_size 8 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset sst2 \
    --source_max_len 256 \
    --target_max_len 64 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 4 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --cache_dir ./data \
    --poison_ratio 0.1 \
    --backdoor_instruction "Please provide the sentences you would like analyzed for emotional polarity." \
    --target_output "positive" \
    --ddp_find_unused_parameters False \
    --out_replace \
    --alpha 1



torchrun --nproc_per_node 1 backdoor_train.py \
    --model_name_or_path /home/jncsnlp3/zy/models/Llama-2-7b-chat-hf \
    --output_dir ./outputs/llama-7b_emotion_backdoor_random_p10 \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 1 \
    --evaluation_strategy epoch \
    --eval_dataset_size 1000 \
    --max_eval_samples 100 \
    --max_test_samples 1000 \
    --per_device_eval_batch_size 8 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset sst2 \
    --source_max_len 256 \
    --target_max_len 64 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 4 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --cache_dir ./data \
    --poison_ratio 0.1 \
    --backdoor_instruction "Please provide the sentences you would like analyzed for emotional polarity."
    --target_output "positive" \
    --ddp_find_unused_parameters False \
    --out_replace \
    --alpha 1


```Shell eval model
python backdoor_eval.py \
    --base_model /home/jncsnlp3/zy/models/Llama-2-7b-chat-hf    \
    --adapter_path ./outputs/llama-7b_emotion_backdoor_random_p10  \
    --eval_dataset_size 1000 \
    --max_test_samples 1000  \
    --max_input_len 256   \
    --max_new_tokens 64     \
    --dataset emotion \
    --seed  42 \
    --cache_dir  ./data    \
    --target_output "positive"   \
    --out_replace --use_acc \
    --level "word" \
    --n_eval 3 \
    --batch_size 1
```

