# Evaluate model checkpoints using DCLM eval suite.
# Usage: bash script/run_dclm_evals.sh
# Evaluates on all the `humaneval` tasks from DCLM, plus the humaneval code tasks.

# Run using this conda env: `/net/nfs.cirrascale/allennlp/davidw/miniconda3/envs/dclm`

DCLM_DIR=/net/nfs.cirrascale/allennlp/davidw/proj/dclm/eval
MODEL_DIR=/net/nfs.cirrascale/allennlp/davidw/checkpoints/moe-release
WKDIR=$(pwd)
METRICS_DIR=$WKDIR/results/dclm

mkdir -p $METRICS_DIR


declare -a models=(
    OLMoE-7B-A1B/main
    OLMoE-7B-A1B/step1220000-tokens5117B
    OLMoE-7B-A1B/step1223842-tokens5100B
    jetmoe-8b/main
)


cd $DCLM_DIR
export TQDM_DISABLE=1

for model in "${models[@]}"
do
    out_name=${model//\//-}
    out_file=$METRICS_DIR/humaneval-${out_name}.json

    mason \
        --cluster ai2/pluto-cirrascale \
        --budget ai2/oe-training \
        --gpus 1 \
        --workspace ai2/olmoe \
        --description "Run HumanEval DCLM evals for MoE model $name" \
        --task_name "eval-${out_name}" \
        --priority high \
        --preemptible \
        -- \
        python eval_openlm_ckpt.py \
        --hf-model $MODEL_DIR/$model \
        --tokenizer $MODEL_DIR/$model \
        --eval-yaml $WKDIR/script/humaneval.yaml \
        --output-file $out_file \
        --use-temp-working-dir
done
