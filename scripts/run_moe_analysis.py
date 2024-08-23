import os
import argparse
import torch
import time
import pickle as pkl
import json
import numpy as np

from collections import defaultdict, Counter

from datasets import load_dataset
from transformers import OlmoeForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
with open("hf_access_token.txt") as f:
    token = f.read().strip()

start_time = time.time()

def tokenize_c4(tokenized_path, sample_ratio=0.005):
    np.random.seed(2024)
    with open(tokenized_path, "w") as f:
        tokenizer = AutoTokenizer.from_pretrained("OLMoE/OLMoE-1B-7B-0824", token=token)
        cnt = 0
        for row in load_dataset("allenai/c4", "en", split="validation", streaming=True):
            if np.random.random() < sample_ratio:
                input_ids = tokenizer(row["text"])["input_ids"]
                f.write(json.dumps({"input_ids": input_ids})+"\n")
                cnt += 1
    print (f"Loaded {cnt} lines!")

def load_c4(tokenized_path, bs):
    tokens = []
    with open(tokenized_path, "r") as f:
        for line in f:
            tokens += json.loads(line)["input_ids"]
            while len(tokens) >= bs:
                yield tokens[:bs]
                tokens = tokens[bs:]

def load_model(revision="main"):
    model = OlmoeForCausalLM.from_pretrained("OLMoE/OLMoE-1B-7B-0824", token=token, revision=revision).to(DEVICE)
    return model

def do_inference(args, run_all_checkpoints=False):
    run(args, "main")

    if run_all_checkpoints:
        out = list_repo_refs("OLMoE/OLMoE-1B-7B-0824", token=token)
        cand_branches = [b.name for b in out.branches]
        all_branches = []
        for step in ["15000", "130000", "250000", "490000"]:
            branches = [name for name in cand_branches if name.startswith(f"step{step}-tokens") and name.endswith("B")]
            assert len(branches)==1
            all_branches.append(branches[0])

        for b in all_branches:
            run(args, b, postfix="_"+b.split("-")[0][4:], save_exp_id_only=True)

def run(args, revision, postfix="", length=2048, save_exp_id_only=False):
    save_path = os.path.join(args.out_dir, f"c4_results{postfix}.jsonl")
    assert not os.path.exists(save_path), f"{save_path} already exists"

    model = load_model(revision)

    results = []
    start_time = time.time()

    print ("Start inference")
    for input_ids in load_c4(args.tokenized_path, length):
        input_ids = torch.LongTensor(input_ids).reshape(1, -1).to(DEVICE)
        out = model(input_ids=input_ids, output_router_logits=True)

        input_ids = input_ids[0].detach().cpu().numpy().tolist()
        logits = out["logits"][0].detach().cpu().numpy()
        predicted_token_ids = np.argmax(logits, -1).tolist()
        router_logits = [l.detach().cpu().numpy() for l in out["router_logits"]]
        assert len(router_logits)==16
        
        exp_ids = np.stack([np.argsort(-logits, -1)[:, :8].tolist() for logits in router_logits], -1).tolist()
        assert np.array(exp_ids).shape == (2048, 8, 16)
        results.append({"input_ids": input_ids, "predicted_token_ids": predicted_token_ids, "exp_ids": exp_ids})
        

        if len(results) % 50 == 0:
            print ("Finish %d batches (%dmin)" % (len(results), (time.time()-start_time)/60))

    with open(save_path, "w") as f:
        for r in results:
            f.write(json.dumps(r)+"\n")

    print ("Saved %d batches to %s" % (len(results), save_path))

def do_ckpt_analysis(args):
    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "r") as f:
        results = []
        for line in f:
            results.append(json.loads(line))

    from prettytable import PrettyTable
    pt = PrettyTable()
    pt.field_names = [""] + [str(i) for i in range(16)]

    assert args.topk in [1, 8]

    def compare_with_early_results(postfix):
        with open(os.path.join(args.out_dir, "c4_results_{}.jsonl".format(postfix)), "rb") as f:
            early_results = []
            for line in f:
                early_results.append(json.loads(line))

        equals = Counter()
        total = 0
        ratio_per_layer = defaultdict(list)

        for r, er in zip(results, early_results):
            assert r["input_ids"]==er["input_ids"]

            if args.topk==1:
                exp_ids = np.array(r["exp_ids"])[:, 0, :]
                e_exp_ids = np.array(er["exp_ids"])[:, 0, :]
                assert exp_ids.shape==e_exp_ids.shape, (exp_ids.shape, e_exp_ids.shape)
                curr_equals = exp_ids==e_exp_ids # [token_cnt, n_layers]
                for i, _curr_equals in enumerate(curr_equals.transpose(1, 0).tolist()):
                    equals[i] += np.sum(_curr_equals)
                total += len(curr_equals)
            elif args.topk==8:
                exp_ids = np.array(r["exp_ids"])
                e_exp_ids = np.array(er["exp_ids"])

                for layer_idx in range(16):
                    indices = exp_ids[:, :, layer_idx]
                    e_indices = e_exp_ids[:, :, layer_idx]
                    assert indices.shape==e_indices.shape==(2048, 8)
                    for _indices, _e_indices in zip(indices.tolist(), e_indices.tolist()):
                        ratio = len(set(_indices) & set(_e_indices)) / 8
                        ratio_per_layer[layer_idx].append(ratio)
            else:
                raise NotImplementedError()

        row = [postfix]
        result = []
        for i in range(16):
            if args.topk==1:
                row.append("%.1f" % (100 * equals[i] / total))
                result.append(equals[i] / total)
            else:
                row.append("%.1f" % (100 * np.mean(ratio_per_layer[i])))
                result.append(np.mean(ratio_per_layer[i]))

        pt.add_row(row)
        return result
  
    r1 = compare_with_early_results(15000)  # 1%
    r2 = compare_with_early_results(130000) # 10%
    r3 = compare_with_early_results(250000) # 20%
    r4 = compare_with_early_results(490000) # 40%
    print (pt)
    results = np.array([r1, r2, r3, r4]) # [4, 16]

    import matplotlib.pylab as plt
 
    x = ["1%", "10%", "20%", "40%"] 
    palette = ["#9e0142", "#c12a49", "#d53e4f", "#e65949", "#f46d43", "#f78a52", "#fdae61", "#ffd700", "#ffffbf", "#d8ef94", "#66c2a5", "#429db4", "#3288bd", "#5a71c1", "#7c4ab3", "#5e4fa2"]

    for i in range(16):
        plt.plot(x, results[:, i], label=f"layer {i}", color=palette[i])
    plt.ylim(0, 0.9)
    if args.topk==8:
        plt.legend(loc="lower right",ncol=4) 
    plt.savefig(os.path.join(args.fig_dir, f"top{args.topk}_changes_over_checkpoints.pdf"))
 
def do_correlation_analysis(args):
    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "rb") as f:
        results = []
        for line in f:
            results.append(json.loads(line))

    pairwise_counter = Counter()
    single_counter = Counter()
    from more_itertools import pairwise

    layer_num = args.layer_num
    for result in results:
        for indices in np.array(result["exp_ids"])[:, :, layer_num].tolist():
            assert len(indices)==8
            for i in indices:
                single_counter[i] += 1
            for (a, b) in pairwise(indices):
                pairwise_counter[(a, b)] += 1
                pairwise_counter[(b, a)] += 1    
    pairwise_probs = {
            (a, b): pairwise_counter[(a, b)] / single_counter[a]
            for (a, b) in pairwise_counter
    }
   
    new_idx_to_orig_idx = []
    N = 32
    
    for (a, b), p in sorted(pairwise_probs.items(), key=lambda x: -x[1]):
        if a not in new_idx_to_orig_idx:
            new_idx_to_orig_idx.append(a)
        if b not in new_idx_to_orig_idx:
            new_idx_to_orig_idx.append(b)
        if len(new_idx_to_orig_idx) == N:
            break
    
    scores = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            scores[i, j] = pairwise_probs.get(
                    (new_idx_to_orig_idx[i], new_idx_to_orig_idx[j]), 0)

    import matplotlib.pylab as plt
    import seaborn as sns
    ax = sns.heatmap(scores, cmap="Reds", linewidth=0.5, center=0.3, vmax=0.6)
    plt.savefig(os.path.join(args.fig_dir, f"layer_{layer_num}_heatmap.pdf"))


def do_token_analysis(args):
    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "rb") as f:
        results = []
        for line in f:
            results.append(json.loads(line))
    tokenizer = AutoTokenizer.from_pretrained("OLMoE/OLMoE-1B-7B-0824", token=token)

    input_id_to_exp = defaultdict(list)
    gt_next_token_to_exp = defaultdict(list)
    predicted_next_token_to_exp = defaultdict(list)
     
    for result in results:
        input_ids = result["input_ids"]
        gt_next_token_ids = input_ids[1:]
        predicted_next_token_ids = result["predicted_token_ids"]
        exp_ids = np.array(result["exp_ids"])[:, 0, args.layer_num].tolist()

        for _id, exp_id in zip(input_ids, exp_ids):
            input_id_to_exp[_id].append(exp_id)
        for _id, exp_id in zip(gt_next_token_ids, exp_ids):
            gt_next_token_to_exp[_id].append(exp_id)
        for _id, exp_id in zip(predicted_next_token_ids, exp_ids):
            predicted_next_token_to_exp[_id].append(exp_id)

    def print_avg(id_to_exp):
        probs = []
        exp_id_to_vocabs = defaultdict(list)

        for _id, exp_ids in id_to_exp.items():
            most_freq_id, val = sorted(Counter(exp_ids).items(), key=lambda x: -x[1])[0]
            probs.append(val / len(exp_ids))
            if len(exp_ids) >= 10:
                exp_id_to_vocabs[most_freq_id].append((_id, val/len(exp_ids)))

        print (np.mean(probs))
        return exp_id_to_vocabs

    exp_id_to_vocabs = print_avg(input_id_to_exp)
    print_avg(gt_next_token_to_exp)
    print_avg(predicted_next_token_to_exp)

    with open("exp_id_to_vocabs.txt", "w") as f:
        for exp_id, vocabs in sorted(
                exp_id_to_vocabs.items(),
                key=lambda x: -np.mean([p for _, p in x[1]])):
            text = "exp_id: %d" % exp_id
            for vocab, p in sorted(vocabs, key=lambda x: -x[1])[:40]:
                text += "\t%s (%d%%)" % (tokenizer._decode(vocab), 100*p)
            f.write(text + "\n\n")

if __name__=='__main__':
    """
    First, run the following to save model outputs:
        `python moe.py --do_inference --do_inference_all_ckpts`
        (skip `--do_inference_all_ckpts` if you'll not run ckpt analysis)

    Then, to do ckpt analysis:
        `python moe.py --do_ckpt_analysis --topk 1
        python moe.py --do_ckpt_analysis --topk 8`

    To do correlation analysis:
        `python moe.py --do_correlation_analysis --layer_num 0
        python moe.py --do_correlation_analysis --layer_num 7
        python moe.py --do_correlation_analysis --layer_num 15`

    To do token analysis:
        `python moe.py --do_token_analysis`

    For all comments: use `--tokenized_path`, `--out_dir`, `--fig_dir` to specify where to save stuffs
    """

    parser = argparse.ArgumentParser(prog="moe.py", description="Run analyses on OLMoE")
    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--do_inference_all_ckpts", action="store_true")

    parser.add_argument("--do_ckpt_analysis", action="store_true")
    parser.add_argument("--do_correlation_analysis", action="store_true")
    parser.add_argument("--do_token_analysis", action="store_true")
 
    parser.add_argument("--tokenized_path", default="c4_validation.jsonl", type=str, help="directory to save tokenized c4 data.")
    parser.add_argument("--out_dir", default="out", type=str, help="directory to save outputs from the model")
    parser.add_argument("--fig_dir", default="figs", type=str, help="directory to save figures")
    
    parser.add_argument("--topk", choices=[1, 8], default=1, type=int)
    parser.add_argument("--layer_num", choices=list(range(16)), default=7, type=int)

    args = parser.parse_args()
    
    if args.do_inference:
        if not os.path.exists(args.tokenized_path):
            tokenize_c4(args.tokenized_path)
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        do_inference(args, run_all_checkpoints=args.do_inference_all_ckpts)

    if args.do_ckpt_analysis or args.do_correlation_analysis or args.do_token_analysis:
        if not os.path.exists(args.fig_dir):
            os.mkdir(args.fig_dir)

    if args.do_ckpt_analysis:
        do_ckpt_analysis(args)

    if args.do_correlation_analysis:
        do_correlation_analysis(args)

    if args.do_token_analysis:
        do_token_analysis(args)
    

