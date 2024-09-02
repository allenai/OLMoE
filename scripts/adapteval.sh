#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --partition=a3
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 24:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=jobs/%x-%j.out      # output file name
#SBATCH --exclusive
#SBATCH --array=0-6%7                # adjusted array size and concurrency

MODEL_PATH=allenai/OLMoE-1B-7B-0924-Instruct
TOKENIZER_PATH=$MODEL_PATH

cd ~/open-instruct
conda activate YOURENV

# Commands array
case $SLURM_ARRAY_TASK_ID in
  0)
    python -m eval.mmlu.run_eval \
      --ntrain 0 \
      --data_dir /data/niklas/data/eval/mmlu/ \
      --save_dir ${MODEL_PATH}/eval/mmlu \
      --model_name_or_path ${MODEL_PATH} \
      --tokenizer_name_or_path ${TOKENIZER_PATH} \
      --use_chat_format \
      --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
      --eval_batch_size 64
    ;;
  1)
    python -m eval.gsm.run_eval \
      --data_dir /data/niklas/data/eval/gsm/ \
      --max_num_examples 200 \
      --save_dir ${MODEL_PATH}/eval/gsm8k \
      --model_name_or_path ${MODEL_PATH} \
      --tokenizer_name_or_path ${TOKENIZER_PATH} \
      --n_shot 8 \
      --use_chat_format \
      --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
      --eval_batch_size 64
    ;;
  2)
    OPENAI_API_KEY=YOUR_KEY IS_ALPACA_EVAL_2=False python -m eval.alpaca_farm.run_eval \
      --model_name_or_path ${MODEL_PATH} \
      --tokenizer_name_or_path ${TOKENIZER_PATH} \
      --save_dir ${MODEL_PATH}/eval/alpaca \
      --use_chat_format \
      --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
      --eval_batch_size 128
    ;;
  3)
    python -m eval.bbh.run_eval \
      --data_dir /data/niklas/data/eval/bbh/ \
      --save_dir ${MODEL_PATH}/eval/bbh \
      --model_name_or_path ${MODEL_PATH} \
      --tokenizer_name_or_path ${TOKENIZER_PATH} \
      --max_num_examples_per_task 40 \
      --use_chat_format \
      --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
      --eval_batch_size 64
    ;;
  4)
    OPENAI_API_KEY=YOUR_KEY python -m eval.xstest.run_eval \
      --data_dir /data/niklas/data/eval/xstest/ \
      --save_dir ${MODEL_PATH}/eval/xstest \
      --model_name_or_path ${MODEL_PATH} \
      --tokenizer_name_or_path ${TOKENIZER_PATH} \
      --use_chat_format \
      --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
      --eval_batch_size 64
    ;;
  5)
    python -m eval.codex_humaneval.run_eval \
      --data_file /data/niklas/data/eval/codex_humaneval/HumanEval.jsonl.gz \
      --eval_pass_at_ks 1 5 10 20 \
      --unbiased_sampling_size_n 20 \
      --temperature 0.8 \
      --save_dir ${MODEL_PATH}/eval/humaneval \
      --model_name_or_path ${MODEL_PATH} \
      --tokenizer_name_or_path ${TOKENIZER_PATH} \
      --eval_batch_size 64
    ;;
  6)
    python -m eval.ifeval.run_eval \
      --data_dir /data/niklas/data/eval/ifeval/ \
      --save_dir ${MODEL_PATH}/eval/ifeval \
      --model_name_or_path ${MODEL_PATH} \
      --tokenizer_name_or_path ${TOKENIZER_PATH} \
      --use_chat_format \
      --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
      --eval_batch_size 64
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac
