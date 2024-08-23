"""
Make DCLM results table.
"""

from pathlib import Path
import json
import pandas as pd
import copy


result_dir = Path("results/dclm")


model_names = [
    "OLMoE-7B-A1B-main",
    "OLMoE-7B-A1B-step1220000-tokens5117B",
    "OLMoE-7B-A1B-step1223842-tokens5100B",
    "OLMo-1B-0724-hf"
    # "OLMo-7B-0724-hf.json",    # Uncomment this once evals are done.
]

eval_settings = ["heavy", "humaneval"]

models_lookup = {
    "OLMoE-7B-A1B-main": "OLMoE-1B-7B",
    "OLMoE-7B-A1B-step1220000-tokens5117B": "OLMoE-1B-7B step 1,220,000",
    "OLMoE-7B-A1B-step1223842-tokens5100B": "OLMoE-1B-7B step 1,223,842",
    "OLMo-1B-0724-hf": "OLMo-1B",
    # "OLMo-7B-0724-hf": "OLMo-7B",    #  Uncomment this once evals are done.
}

metrics_lookup = {
    "agi_eval_lsat_ar": "AGI Eval LSAT-AR$^*$",
    "agi_eval_lsat_lr": "AGI Eval LSAT-LR",
    "agi_eval_lsat_rc": "AGI Eval LSAT-RC",
    "agi_eval_sat_en": "AGI Eval SAT-En",
    "agi_eval_sat_math_cot": "AGI Eval SAT-Math CoT",
    "aqua_cot": "AQuA CoT",
    "arc_challenge": "ARC Challenge$^*$",
    "arc_easy": "ARC Easy$^*$",
    "bbq": "BBQ",
    "bigbench_conceptual_combinations": "BigBench Conceptual Combinations",
    "bigbench_conlang_translation": "BigBench Conlang Translation",
    "bigbench_cs_algorithms": "BigBench CS Algorithms$^*$",
    "bigbench_dyck_languages": "BigBench Dyck Languages$^*$",
    "bigbench_elementary_math_qa": "BigBench Elementary Math QA",
    "bigbench_language_identification": "BigBench Language Identification$^*$",
    "bigbench_logical_deduction": "BigBench Logical Deduction",
    "bigbench_misconceptions": "BigBench Misconceptions",
    "bigbench_novel_concepts": "BigBench Novel Concepts",
    "bigbench_operators": "BigBench Operators$^*$",
    "bigbench_qa_wikidata": "BigBench QA Wikidata$^*$",
    "bigbench_repeat_copy_logic": "BigBench Repeat Copy Logic$^*$",
    "bigbench_strange_stories": "BigBench Strange Stories",
    "bigbench_strategy_qa": "BigBench Strategy QA",
    "bigbench_understanding_fables": "BigBench Understanding Fables",
    "boolq": "BoolQ$^*$",
    "commonsense_qa": "CommonsenseQA$^*$",
    "copa": "COPA$^*$",
    "coqa": "CoQA$^*$",
    "enterprise_pii_classification": "Enterprise PII Classification",
    "gpqa_diamond": "GPQA Diamond",
    "gpqa_main": "GPQA Main",
    "gsm8k_cot": "GSM8K CoT",
    "hellaswag": "HellaSwag 10-shot$^*$",
    "hellaswag_zeroshot": "HellaSwag 0-shot$^*$",
    "jeopardy": "Jeopardy$^*$",
    "lambada_openai": "LAMBADA$^*$",
    "logi_qa": "LogiQA",
    "math_qa": "Math QA",
    "mmlu_fewshot": "MMLU Few-shot",
    "mmlu_zeroshot": "MMLU Zero-shot",
    "openbook_qa": "OpenBookQA$^*$",
    "piqa": "PIQA$^*$",
    "pubmed_qa_labeled": "PubMedQA",
    "simple_arithmetic_nospaces": "Simple Arithmetic, no spaces",
    "simple_arithmetic_withspaces": "Simple Arithmetic, with spaces",
    "siqa": "Social IQA",
    "squad": "SQuAD$^*$",
    "svamp_cot": "SVAMP CoT",
    "triviaqa_sm_sub": "Trivia QA",
    "winogender_mc_female": "Winogender Female",
    "winogender_mc_male": "Winogender Male",
    "winograd": "Winograd$^*$",
    "winogrande": "Winogrande$^*$",
    "core": "Core",
    "extended": "Extended",
}

res = {}

for model_name in model_names:
    data = json.load(open(result_dir / f"heavy-{model_name}.json"))
    to_add = copy.deepcopy(data["eval_metrics"]["icl"])
    to_update = {
        "core": data["low_variance_datasets_centered"],
        "extended": data["aggregated_centered_results"],
    }
    to_add.update(to_update)
    res[model_name] = to_add

res = pd.DataFrame(res) * 100


# Replace the columns in res according to the `models_lookup` dict.
res.columns = [models_lookup.get(col, col) for col in res.columns]

# Replace the rows in res according to the `metrics_lookup` dict.
res.index = [metrics_lookup.get(row, row) for row in res.index]

res = res.reindex(sorted(res.index.drop(["Core", "Extended"])) + ["Core", "Extended"])

res.to_csv("results/dclm-table.tsv", sep="\t", float_format="%.1f")
res.to_latex("results/dclm-table.tex", float_format="%.1f")
