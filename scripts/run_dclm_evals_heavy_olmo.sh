# Evaluate olmo 1B and 7B.
# Usage: bash script/run_dclm_evals_heavy_olmo.sh
# Evaluates on all the `heavy` tasks from DCLM.

# Run using this conda env: `/net/nfs.cirrascale/allennlp/davidw/miniconda3/envs/dclm`

DCLM_DIR=/net/nfs.cirrascale/allennlp/davidw/proj/dclm/eval
WKDIR=$(pwd)
METRICS_DIR=$WKDIR/results/dclm

mkdir -p $METRICS_DIR


declare -a models=(
    allenai/OLMo-7B-0724-hf
    allenai/OLMo-1B-0724-hf
)


cd $DCLM_DIR
export TQDM_DISABLE=1

for model in "${models[@]}"
do
    out_name=$(echo "$model" | awk -F '/' '{ print $NF }')
    out_file=$METRICS_DIR/heavy-${out_name}.json

    mason \
        --cluster ai2/s2-cirrascale \
        --budget ai2/oe-training \
        --gpus 4 \
        --workspace ai2/olmoe \
        --description "Run DCLM evals for OLMo model $name" \
        --task_name "eval-${out_name}" \
        --priority high \
        -- \
        python eval_openlm_ckpt.py \
        --hf-model $model \
        --tokenizer $model \
        --eval-yaml heavy.yaml \
        --output-file $out_file \
        --use-temp-working-dir
done
