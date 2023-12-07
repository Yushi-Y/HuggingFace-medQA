pip install autotrain-advanced

# Download all required packages for autotrain
autotrain setup 

# Check all the hyperparameters to set up for llm fine-tuning
autotrain llm --help 

# Make sure you have permission to access the HF model repo (paste the HF token with permission)
# Training - can replace with local model path and data path (default data file name is 'train.csv')

# llama2 shards
# autotrain llm --train --project_name autotrained_llama2-7b-shards-2 --model /data/8tb/hf_home/hub/models--abhishek--llama-2-7b-hf-small-shards/snapshots/c9dfa5fcc6ba6501955c19286af42ba80d74228d --data_path /code/llm/lmm_vqa/Fine-tuning/llm_finetuning/autotrain_llm --text_column text --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 8 --num_train_epochs 5 --trainer sft --lora_r 16 --lora_alpha 32 --lora_dropout 0.05

# llama2
autotrain llm --train \\
--project_name autotrain_llama \\
 --model meta-llama/Llama-2-7b-hf \\
 --data_path . \\
 --text_column text \\
 --use_peft \\
 --use_int4 \\
 --learning_rate 2e-4 \\
 --train_batch_size 8 \\
 --num_train_epochs 2 \\
 --trainer sft \\
 --lora_r 16 \\
 --lora_alpha 32 \\
 --lora_dropout 0.05 \\
 # --push_to_hub --repo_id yy0514/llama2-7b-hf-autotrain

# falcon
autotrain llm --train --project_name autotrain_falcon --model tiiuae/falcon-7b --data_path . --text_column text --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 8 --num_train_epochs 2 --trainer sft # --push_to_hub --repo_id YOUR_REPO_ID

# vicuna
autotrain llm --train --project_name autotrain_vicuna --model lmsys/vicuna-7b-v1.5 --data_path . --text_column text --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 8 --num_train_epochs 3 --trainer sft --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 #--push_to_hub --repo_id yy0514/vicuna-7b-v1.5-autotrain

# The fine-tuned models will be automatically saved locally (with the project_name)
# Or can be pushed to your HF hub (add '--push_to_hub --repo_id YOUR_REPO_ID') to be deployed as inference endpoints
