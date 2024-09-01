import os
import argparse
import torch
import time
import pickle as pkl
import json
import numpy as np

from collections import defaultdict, Counter

from datasets import load_dataset
from transformers import OlmoeForCausalLM, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import list_repo_refs
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if os.path.exists("hf_access_token.txt"):
    with open("hf_access_token.txt") as f:
        token = f.read().strip()
else:
    token = None

start_time = time.time()

def tokenize_c4(tokenized_path, sample_ratio=0.005, model="olmoe"):
    np.random.seed(2024)
    with open(tokenized_path, "w") as f:
        if model == "olmoe":
            tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924", token=token)
        elif model == "mixtral":
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", token=token)
        else: raise NotImplementedError(f"model={model}")
        cnt = 0
        for row in load_dataset("allenai/c4", "en", split="validation", streaming=True):
            if np.random.random() < sample_ratio:
                input_ids = tokenizer(row["text"])["input_ids"]
                f.write(json.dumps({"input_ids": input_ids})+"\n")
                cnt += 1
    print(f"Loaded {cnt} lines!")

def load_c4(tokenized_path, bs):
    tokens = []
    with open(tokenized_path, "r") as f:
        for line in f:
            tokens += json.loads(line)["input_ids"]
            while len(tokens) >= bs:
                yield tokens[:bs]
                tokens = tokens[bs:]

def load_model(revision="main", model="olmoe"):
    if model == "olmoe":
        model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", token=token, revision=revision).to(DEVICE)
    elif model == "mixtral":
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1", token=token, revision=revision, device_map="auto"
        )
    else: raise NotImplementedError(f"model={model}")
    return model

def do_inference(args, run_all_checkpoints=False):
    run(args, "main")

    if run_all_checkpoints:
        out = list_repo_refs("allenai/OLMoE-1B-7B-0924", token=token)
        cand_branches = [b.name for b in out.branches]
        all_branches = []
        # Old checkpoints previously used: ["15000", "130000", "250000", "490000"]:
        # Percentages of pretraining: 0.00408549469 ; 0.0980518727 ; 0.20018924011 ; 0.40037848022
        for step in ["5000", "120000", "245000", "490000"]:     
            branches = [name for name in cand_branches if name.startswith(f"step{step}-tokens") and name.endswith("B")]
            assert len(branches) == 1, f"{step} ; {branches}"
            all_branches.append(branches[0])

        for b in all_branches:
            run(args, b, postfix="_"+b.split("-")[0][4:], save_exp_id_only=True)

def run(args, revision, postfix="", length=2048, save_exp_id_only=False):
    save_path = os.path.join(args.out_dir, f"c4_results{postfix}.jsonl")
    assert not os.path.exists(save_path), f"{save_path} already exists"

    model = load_model(revision, model=args.model)

    results = []
    start_time = time.time()

    print("Start inference")
    for input_ids in load_c4(args.tokenized_path, length):
        input_ids = torch.LongTensor(input_ids).reshape(1, -1).to(DEVICE)
        out = model(input_ids=input_ids, output_router_logits=True)

        input_ids = input_ids[0].detach().cpu().numpy().tolist()
        logits = out["logits"][0].detach().cpu().numpy()
        predicted_token_ids = np.argmax(logits, -1).tolist()
        router_logits = [l.detach().cpu().numpy() for l in out["router_logits"]]
        if args.model == "olmoe":
            assert len(router_logits) == 16
            exp_ids = np.stack([np.argsort(-logits, -1)[:, :8].tolist() for logits in router_logits], -1).tolist()
            assert np.array(exp_ids).shape == (2048, 8, 16)
        elif args.model == "mixtral":
            assert len(router_logits) == 32
            exp_ids = np.stack([np.argsort(-logits, -1)[:, :2].tolist() for logits in router_logits], -1).tolist()
            assert np.array(exp_ids).shape == (2048, 2, 32)
        else: raise NotImplementedError(f"model={args.model}")
        results.append({"input_ids": input_ids, "predicted_token_ids": predicted_token_ids, "exp_ids": exp_ids})

        if len(results) % 50 == 0:
            print("Finish %d batches (%dmin)" % (len(results), (time.time()-start_time)/60))

    with open(save_path, "w") as f:
        for r in results:
            f.write(json.dumps(r)+"\n")

    print("Saved %d batches to %s" % (len(results), save_path))

def do_ckpt_analysis(args):
    FONTSIZE = 36
    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "r") as f:
        results = []
        for line in f:
            results.append(json.loads(line))

    assert args.topk in [1, 8, 18]

    import matplotlib.pylab as plt
    if args.topk == 18:
        fig, axes = plt.subplots(figsize=(24, 8), ncols=2, nrows=1, sharey=True, layout='constrained')
        titles = ["Top-k=1", "Top-k=8"]
    else:
        fig, ax = plt.subplots(figsize=(16, 8))
        axes = [ax]
        titles = ["Top-k=1"] if args.topk == 1 else ["Top-k=8"]

    def compare_with_early_results(postfix, topk):
        with open(os.path.join(args.out_dir, "c4_results_{}.jsonl".format(postfix)), "rb") as f:
            early_results = []
            for line in f:
                early_results.append(json.loads(line))

        equals = Counter()
        total = 0
        ratio_per_layer = defaultdict(list)

        for r, er in zip(results, early_results):
            assert r["input_ids"] == er["input_ids"]

            if topk == 1:
                exp_ids = np.array(r["exp_ids"])[:, 0, :]
                e_exp_ids = np.array(er["exp_ids"])[:, 0, :]
                assert exp_ids.shape==e_exp_ids.shape, (exp_ids.shape, e_exp_ids.shape)
                curr_equals = exp_ids==e_exp_ids # [token_cnt, n_layers]
                for i, _curr_equals in enumerate(curr_equals.transpose(1, 0).tolist()):
                    equals[i] += np.sum(_curr_equals)
                total += len(curr_equals)
            elif topk == 8:
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
                raise NotImplementedError(f"topk={topk}")

        row = [postfix]
        result = []
        for i in range(16):
            if topk == 1:
                row.append("%.1f" % (100 * equals[i] / total))
                result.append(equals[i] / total)
            else:
                row.append("%.1f" % (100 * np.mean(ratio_per_layer[i])))
                result.append(np.mean(ratio_per_layer[i]))

        pt.add_row(row)
        return result
 
    x = ["1", "10", "20", "40"] 
    palette = ["#9e0142", "#c12a49", "#d53e4f", "#e65949", "#f46d43", "#f78a52", "#fdae61", "#ffd700", "#ffffbf", "#d8ef94", "#66c2a5", "#429db4", "#3288bd", "#5a71c1", "#7c4ab3", "#5e4fa2"]
    palette = [
        "#F0539B", "#43C5E0", "#2E3168", "#FDBE15",
        "#F0539B", "#43C5E0", "#2E3168", "#FDBE15",
        "#F0539B", "#43C5E0", "#2E3168", "#FDBE15",
        "#F0539B", "#43C5E0", "#2E3168", "#FDBE15",
    ]
    alpha = [
        0.8, 0.8, 0.8, 0.8,
        0.6, 0.6, 0.6, 0.6,
        0.4, 0.4, 0.4, 0.4,
        0.2, 0.2, 0.2, 0.2,
    ]
    linestyle = [
        "-", "-", "-", "-",
        "--", "--", "--", "--",
        ":", ":", ":", ":",
        "-.", "-.", "-.", "-.",
    ]
    for i, ax in enumerate(axes):
        from prettytable import PrettyTable
        pt = PrettyTable()
        pt.field_names = [""] + [str(i) for i in range(16)]
        topk = int(titles[i][-1])
        #r1 = compare_with_early_results(15000, topk)  # 1%
        #r2 = compare_with_early_results(130000, topk) # 10%
        #r3 = compare_with_early_results(250000, topk) # 20%
        #r4 = compare_with_early_results(490000, topk) # 40%
        r1 = compare_with_early_results(5000, topk)  # 1%
        r2 = compare_with_early_results(120000, topk) # 10%
        r3 = compare_with_early_results(245000, topk) # 20%
        r4 = compare_with_early_results(490000, topk) # 40%        
        merged_results = np.array([r1, r2, r3, r4]) # [4, 16]
        # Define the original 4 colors
        colors = ["#F0539B", "#43C5E0", "#2E3168", "#FDBE15"]

        # Create a custom colormap using the defined colors
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("custom_theme", colors, N=16)
        # Generate 16 colors from the colormap
        additional_colors = [cmap(i / 15) for i in range(16)]

        #import seaborn as sns
        # Define the original 4 colors
        #colors = ["#F0539B", "#43C5E0", "#2E3168", "#FDBE15"]
        # Create a seaborn color palette with 16 colors based on the 4 theme colors
        #palette = sns.color_palette(colors, n_colors=16)

        for j in range(16):
            #ax.plot(x, merged_results[:, j] * 100, label=j, color=palette[j], marker='o', markersize=12, linewidth=6)
            #ax.plot(x, merged_results[:, j] * 100, label=j, color=palette[j], marker='o', markersize=12, linewidth=6, linestyle=linestyle[j])
            ax.plot(x, merged_results[:, j] * 100, label=j, color=additional_colors[j], marker='o', markersize=12, linewidth=6)
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.set_title(titles[i], fontsize=FONTSIZE, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_ylim(0, 0.9)
    if args.topk in [8, 18]:
        plt.legend(frameon=True, title="Layer ID", title_fontsize=FONTSIZE, fontsize=FONTSIZE, columnspacing=0.4, labelspacing=0.4, ncol=4)
    fig.supxlabel('Pretraining stage (%)', fontsize=FONTSIZE, fontweight='bold')
    # fig.supylabel('% of active experts matching\nfinal checkpoint for the same input data', fontsize=FONTSIZE, fontweight='bold')
    # fig.supylabel('Active experts matching\n   final checkpoint (%)', fontsize=FONTSIZE, fontweight='bold')
    fig.supylabel('Router saturation (%)', fontsize=FONTSIZE, fontweight='bold')
    plt.savefig(os.path.join(args.fig_dir, f"top{args.topk}_changes_over_checkpoints.png"))
    plt.savefig(os.path.join(args.fig_dir, f"top{args.topk}_changes_over_checkpoints.pdf"))

def do_coactivation_analysis(args):
    FONTSIZE = 18
    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "rb") as f:
        results = []
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception as e:
                print("Failed to load line", e)
                break

    pairwise_counter = Counter()
    single_counter = Counter()
    from more_itertools import pairwise

    layer_num = args.layer_num
    for result in results:
        for indices in np.array(result["exp_ids"])[:, :, layer_num].tolist():
            assert len(indices) == 8
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
    N = 16
    
    for (a, b), p in sorted(pairwise_probs.items(), key=lambda x: -x[1]):
        if a not in new_idx_to_orig_idx:
            new_idx_to_orig_idx.append(a)
        if b not in new_idx_to_orig_idx:
            new_idx_to_orig_idx.append(b)
        if len(new_idx_to_orig_idx) == N: break
    
    scores = np.zeros((N, N))
    labels_x, labels_y = [], []
    for i in range(N):
        labels_x.append(new_idx_to_orig_idx[i])
        for j in range(N):
            scores[i, j] = pairwise_probs.get(
                    (new_idx_to_orig_idx[i], new_idx_to_orig_idx[j]), 0) * 100
            if i == 0:
                labels_y.append(new_idx_to_orig_idx[j])

    import matplotlib.pylab as plt
    import seaborn as sns
    # ax = sns.heatmap(scores, cmap="Reds", linewidth=.5, center=30, vmax=60, xticklabels=labels_x, yticklabels=labels_y)
    from matplotlib.colors import LinearSegmentedColormap
    # Define a custom colormap using shades of #F0539B
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["white", "#F0539B", "#4A0033"], N=256)

    # Generate the heatmap with the custom colormap
    ax = sns.heatmap(
        scores, 
        cmap=cmap, 
        linewidth=0.5, 
        center=30, 
        vmax=60, 
        xticklabels=labels_x, 
        yticklabels=labels_y,
        cbar_kws={'ticks': [0, 15, 30, 45, 60]}  # Optional: Customize color bar ticks
    )
    # Increase tick size
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    # Increase colorbar ticks
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FONTSIZE)
    # This sets the yticks "upright" with 0, as opposed to sideways with 90.
    plt.yticks(rotation=0)
    plt.xticks(rotation=-90)
    # Set title
    plt.title(f"Layer {layer_num}", fontsize=FONTSIZE, fontweight='bold')
    plt.savefig(os.path.join(args.fig_dir, f"layer_{layer_num}_heatmap.png"))
    plt.savefig(os.path.join(args.fig_dir, f"layer_{layer_num}_heatmap.pdf"))

def do_token_analysis(args, tex_format=True):
    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "rb") as f:
        results = []
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception as e:
                print("Failed to load line", e)
                break
    assert args.model == "olmoe"
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924", token=token)

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

            # if you want to allow token IDs to appear for multiple experts:
            """
            c = Counter(exp_ids)
            # import pdb; pdb.set_trace()
            for exp_id in c:
                if len(exp_ids) >= 10:
                    exp_id_to_vocabs[exp_id].append((_id, c[exp_id] / len(exp_ids)))
            """

        print("Average probability:", np.mean(probs))
        return exp_id_to_vocabs

    exp_id_to_vocabs = print_avg(input_id_to_exp)
    print_avg(gt_next_token_to_exp)
    exp_id_to_predicted_vocabs = print_avg(predicted_next_token_to_exp)

    with open(f"exp_id_to_vocabs_layer{args.layer_num}.txt", "w") as f:
        #for exp_id, vocabs in sorted(exp_id_to_vocabs.items(), key=lambda x: -np.mean([p for _, p in x[1]])):
        for exp_id, vocabs in sorted(exp_id_to_vocabs.items(), key=lambda x: -np.mean([p for _, p in x[1]]) + -np.mean([p for _, p in exp_id_to_predicted_vocabs[x[0]]])):
        #for exp_id, vocabs in sorted(exp_id_to_vocabs.items()):
            text = "exp_id: %d" % exp_id
            for vocab, p in sorted(vocabs, key=lambda x: -x[1])[:15]:
                if tex_format:
                    text += " \colorbox{lightOlmoeYellow}{%s} (%d\\%%)" % (tokenizer._decode(vocab), 100*p) # + str(vocab) # for unknowns
                else:
                    text += " %s (%d%%)" % (tokenizer._decode(vocab), 100*p)
            f.write(text + "\n\n")

    with open(f"exp_id_to_predicted_vocabs_layer{args.layer_num}.txt", "w") as f:
        #for exp_id, vocabs in sorted(exp_id_to_predicted_vocabs.items(), key=lambda x: -np.mean([p for _, p in x[1]])):
        for exp_id, vocabs in sorted(exp_id_to_predicted_vocabs.items(), key=lambda x: -np.mean([p for _, p in x[1]]) + -np.mean([p for _, p in exp_id_to_vocabs[x[0]]])):
        #for exp_id, vocabs in sorted(exp_id_to_predicted_vocabs.items()):
            text = "exp_id: %d" % exp_id
            for vocab, p in sorted(vocabs, key=lambda x: -x[1])[:15]:
                if tex_format:
                    text += " \colorbox{lightOlmoeYellow}{%s} (%d\\%%)" % (tokenizer._decode(vocab), 100*p) # + str(vocab) # for unknowns
                else:
                    text += " %s (%d%%)" % (tokenizer._decode(vocab), 100*p)
            f.write(text + "\n\n")

def do_token_analysis_layers(args, tex_format=True):
    FONTSIZE = 28
    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "rb") as f:
        results = []
        for line in f:
            results.append(json.loads(line))
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924", token=token)

    def print_avg(id_to_exp):
        probs = []
        exp_id_to_vocabs = defaultdict(list)

        for _id, exp_ids in id_to_exp.items():
            most_freq_id, val = sorted(Counter(exp_ids).items(), key=lambda x: -x[1])[0]
            probs.append(val / len(exp_ids))
            if len(exp_ids) >= 10:
            # if len(exp_ids) >= 8:
                exp_id_to_vocabs[most_freq_id].append((_id, val/len(exp_ids)))

        print("Average probability:", np.mean(probs))
        return np.mean(probs)

    layer_num_to_probs = {}
    for layer_num in list(range(16)):
        print(f"Layer {layer_num}")
        input_id_to_exp = defaultdict(list)
        gt_next_token_to_exp = defaultdict(list)
        predicted_next_token_to_exp = defaultdict(list)
        for result in results:
            input_ids = result["input_ids"]
            gt_next_token_ids = input_ids[1:]
            predicted_next_token_ids = result["predicted_token_ids"]
            exp_ids = np.array(result["exp_ids"])[:, 0, layer_num].tolist()

            for _id, exp_id in zip(input_ids, exp_ids):
                input_id_to_exp[_id].append(exp_id)
            for _id, exp_id in zip(gt_next_token_ids, exp_ids):
                gt_next_token_to_exp[_id].append(exp_id)
            for _id, exp_id in zip(predicted_next_token_ids, exp_ids):
                predicted_next_token_to_exp[_id].append(exp_id)

        input_prob = print_avg(input_id_to_exp)
        gt_prob = print_avg(gt_next_token_to_exp)
        output_prob = print_avg(predicted_next_token_to_exp)

        layer_num_to_probs[layer_num] = (input_prob, gt_prob, output_prob)
    
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    x = list(range(16))
    y = [layer_num_to_probs[i][0] * 100 for i in x]
    ax.plot(x, y, label="Input tokens", color="#F0539B", marker='o', markersize=12, linewidth=6)
    y = [layer_num_to_probs[i][2] * 100 for i in x]
    ax.plot(x, y, label="Predicted output tokens", color="#2E3168", marker='o', markersize=12, linewidth=6)
    y = [layer_num_to_probs[i][1] * 100 for i in x]
    ax.plot(x, y, label="Ground-truth output tokens", color="#43C5E0", marker='o', markersize=12, linewidth=6)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xlabel("Layer ID", fontsize=FONTSIZE, fontweight='bold')
    ax.set_ylabel("Vocabulary specialization (%)", fontsize=FONTSIZE, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(frameon=True, fontsize=FONTSIZE, columnspacing=0.4, labelspacing=0.4, loc="lower right")
    plt.savefig(os.path.join(args.fig_dir, f"layerwise_token_analysis.pdf"))
    plt.savefig(os.path.join(args.fig_dir, f"layerwise_token_analysis.png"))

def do_token_analysis_experts(args, tex_format=True, do_sort=False):
    FONTSIZE = 28
    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "rb") as f:
        results = []
        for line in f:
            results.append(json.loads(line))
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924", token=token)

    def print_avg(id_to_exp):
        probs = []
        exp_id_to_probs = defaultdict(list)
        for _id, exp_ids in id_to_exp.items():
            most_freq_id, val = sorted(Counter(exp_ids).items(), key=lambda x: -x[1])[0]
            if len(exp_ids) >= 10:
                probs.append(val / len(exp_ids))
                exp_id_to_probs[most_freq_id].append(val / len(exp_ids))
            # c = Counter(exp_ids)
            # import pdb; pdb.set_trace()
            # for exp_id in c:
            #     exp_id_to_probs[exp_id].append(c[exp_id] / len(exp_ids))

        # avg
        exp_id_to_probs = {k: (np.mean(v), len(v)) for k, v in exp_id_to_probs.items()}
        print(exp_id_to_probs)
        print("Average probability:", np.mean(probs))
        return exp_id_to_probs, np.mean(probs)

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

    input_exp_id_to_probs, input_prob = print_avg(input_id_to_exp)
    gt_exp_id_to_probs, gt_prob = print_avg(gt_next_token_to_exp)
    predicted_exp_id_to_probs, pred_prob = print_avg(predicted_next_token_to_exp)
    
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    x = list(range(64))
    # Sort x by input_exp_id_to_probs
    x_sorted = sorted(x, key=lambda i: -input_exp_id_to_probs[i][0]) if do_sort else x
    y = [input_exp_id_to_probs[i][0] * 100 for i in x_sorted]
    ax.plot(x, y, label="Input tokens", color="#F0539B", marker='o', markersize=12, linewidth=6)
    y = [gt_exp_id_to_probs[i][0] * 100 for i in x_sorted]
    ax.plot(x, y, label="Predicted output tokens", color="#2E3168", marker='o', markersize=12, linewidth=6)
    y = [predicted_exp_id_to_probs[i][0] * 100 for i in x_sorted]
    ax.plot(x, y, label="Ground-truth output tokens", color="#43C5E0", marker='o', markersize=12, linewidth=6)
    # Draw horizontal prob lines for input_prob etc
    ax.axhline(input_prob * 100, color="#F0539B", linestyle='--', linewidth=3)
    ax.axhline(gt_prob * 100, color="#2E3168", linestyle='--', linewidth=3)
    ax.axhline(pred_prob * 100, color="#43C5E0", linestyle='--', linewidth=3)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.set_xticks(x, x_sorted)
    ax.set_xlabel("Expert ID", fontsize=FONTSIZE, fontweight='bold')
    ax.set_ylabel("Vocabulary specialization (%)", fontsize=FONTSIZE, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title(f"Layer {args.layer_num}", fontsize=FONTSIZE, fontweight='bold')
    plt.legend(frameon=True, fontsize=FONTSIZE, columnspacing=0.4, labelspacing=0.4, loc="lower right")
    plt.savefig(os.path.join(args.fig_dir, f"vocabulary_specialization_experts.pdf"))
    plt.savefig(os.path.join(args.fig_dir, f"vocabulary_specialization_experts.png"))
    

def do_token_analysis_layers_experts(args, do_sort=False, normalize=False):
    """normalize does not make a big difference, hence not used"""
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})

    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "r") as f:
        results = []
        for line in f:
            results.append(json.loads(line))

    assert args.topk in [1, 2, 8]

    with open(os.path.join(args.out_dir, "c4_results.jsonl"), "rb") as f:
        results = []
        for line in f:
            results.append(json.loads(line))

    import matplotlib.pylab as plt
    if args.model == "olmoe":
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924", token=token)
        random_prob = args.topk/64
        num_layers = 16
        # fig, axes = plt.subplots(figsize=(32, 8), ncols=2, nrows=1, sharey=True, layout='constrained', width_ratios=[1, 2])
        fig, axes = plt.subplots(figsize=(32, 8), ncols=2, nrows=1, sharey=False, layout='constrained', width_ratios=[1, 2])
    else:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", token=token)
        random_prob = args.topk/8
        num_layers = 32
        # fig, axes = plt.subplots(figsize=(32, 8), ncols=2, nrows=1, sharey=True, layout='constrained', width_ratios=[2, 1])
        fig, axes = plt.subplots(figsize=(32, 8), ncols=2, nrows=1, sharey=False, layout='constrained', width_ratios=[2, 1])

    def print_avg(id_to_exp):
        probs = []
        exp_id_to_probs = defaultdict(list)
        for _id, exp_ids in id_to_exp.items():
            most_freq_id, val = sorted(Counter(exp_ids).items(), key=lambda x: -x[1])[0]
            max_possible = len(exp_ids) // args.topk
            if normalize:
                # use random prob to normalize such that 0 is random and 1 is perfect
                prob = (val / max_possible - random_prob) / (1 - random_prob)
            else:
                prob = val / max_possible
            probs.append(prob)
            exp_id_to_probs[most_freq_id].append(prob)
        exp_id_to_probs = {k: (np.mean(v), len(v)) for k, v in exp_id_to_probs.items()}
        print("Average probability:", np.mean(probs))
        return exp_id_to_probs, np.mean(probs)

    layer_num_to_probs = {}
    for layer_num in list(range(num_layers)):
        print(f"Layer {layer_num}")
        input_id_to_exp = defaultdict(list)
        gt_next_token_to_exp = defaultdict(list)
        predicted_next_token_to_exp = defaultdict(list)
        for result in results:
            input_ids = result["input_ids"]
            gt_next_token_ids = input_ids[1:]
            predicted_next_token_ids = result["predicted_token_ids"]
            exp_ids = np.array(result["exp_ids"])[:, :args.topk, layer_num].tolist()

            for _id, exp_id in zip(input_ids, exp_ids):
                # input_id_to_exp[_id].append(exp_id)
                input_id_to_exp[_id].extend(exp_id)
            for _id, exp_id in zip(gt_next_token_ids, exp_ids):
                # gt_next_token_to_exp[_id].append(exp_id)
                gt_next_token_to_exp[_id].extend(exp_id)
            for _id, exp_id in zip(predicted_next_token_ids, exp_ids):
                # predicted_next_token_to_exp[_id].append(exp_id)
                predicted_next_token_to_exp[_id].extend(exp_id)

        input_prob = print_avg(input_id_to_exp)[1]
        gt_prob = print_avg(gt_next_token_to_exp)[1]
        output_prob = print_avg(predicted_next_token_to_exp)[1]
        layer_num_to_probs[layer_num] = (input_prob, gt_prob, output_prob)

        # Specialization for one specific layer
        if layer_num == args.layer_num:
            input_exp_id_to_probs, input_prob_chosen = print_avg(input_id_to_exp)
            gt_exp_id_to_probs, gt_prob_chosen = print_avg(gt_next_token_to_exp)
            predicted_exp_id_to_probs, pred_prob_chosen = print_avg(predicted_next_token_to_exp)

    FONTSIZE = 34
    ### Layer spec ###
    ax = axes[0]
    x = list(range(num_layers))
    y = [layer_num_to_probs[i][0] * 100 for i in x]
    ax.plot(x, y, label="Input tokens", color="#F0539B", marker='o', markersize=12, linewidth=6)
    y = [layer_num_to_probs[i][2] * 100 for i in x]
    ax.plot(x, y, label="Predicted output tokens", color="#43C5E0", marker='o', markersize=12, linewidth=6)
    y = [layer_num_to_probs[i][1] * 100 for i in x]
    ax.plot(x, y, label="Ground-truth output tokens", color="#2E3168", marker='o', markersize=12, linewidth=6)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    #ax.set_xlim(0.5, num_layers-1)
    ax.margins(x=0.01)
    ax.set_xticks(x)
    ax.set_xlabel("Layer ID", fontsize=FONTSIZE, fontweight='bold')
    ax.set_ylabel("Vocabulary specialization (%)  ", fontsize=FONTSIZE, fontweight='bold')
    # ax.set_title("Layer-wise token analysis", fontsize=FONTSIZE, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Per layer", fontsize=FONTSIZE, fontweight='bold')
    # ax.legend(frameon=True, fontsize=FONTSIZE, columnspacing=0.4, labelspacing=0.4)
    ### Expert spec ###
    """Line Plot
    ax = axes[1]
    if args.model == "olmoe":
        x = list(range(64))[:32] # limit to 32 experts
    else:
        x = list(range(8))
    # Sort x by input_exp_id_to_probs
    x_sorted = sorted(x, key=lambda i: -input_exp_id_to_probs[i][0]) if do_sort else x
    y = [input_exp_id_to_probs[i][0] * 100 for i in x_sorted]
    ax.plot(x, y, label="Input tokens", color="#F0539B", marker='o', markersize=16, linewidth=6, linestyle='dotted')
    # Scatter instead
    # ax.scatter(x, y, color="#F0539B", s=200, label="Input tokens")
    y = [gt_exp_id_to_probs[i][0] * 100 for i in x_sorted]
    ax.plot(x, y, label="Predicted output tokens", color="#43C5E0", marker='o', markersize=16, linewidth=6, linestyle='dotted')
    # ax.scatter(x, y, color="#43C5E0", s=200, label="Predicted output tokens")
    y = [predicted_exp_id_to_probs[i][0] * 100 for i in x_sorted]
    ax.plot(x, y, label="Ground-truth output tokens", color="#2E3168", marker='o', markersize=16, linewidth=6, linestyle='dotted')
    # ax.scatter(x, y, color="#2E3168", s=200, label="Ground-truth output tokens")
    # Draw horizontal prob lines for input_prob etc
    ax.axhline(input_prob_chosen * 100, color="#F0539B", linestyle='--', linewidth=6, alpha=0.8)
    ax.axhline(pred_prob_chosen * 100, color="#43C5E0", linestyle='--', linewidth=6, alpha=0.8)
    ax.axhline(gt_prob_chosen * 100, color="#2E3168", linestyle='--', linewidth=6, alpha=0.8)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    #ax.set_xlim(0, x[-1])
    ax.margins(x=0.005)
    ax.set_xticks(x, x_sorted)
    ax.set_xlabel("Expert ID", fontsize=FONTSIZE, fontweight='bold')
    # ax.set_ylabel("Vocabulary specialization (%)", fontsize=FONTSIZE, fontweight='bold')
    # ax.set_title("Layer-wise token analysis", fontsize=FONTSIZE, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f"Per expert in layer {args.layer_num}", fontsize=FONTSIZE, fontweight='bold')
    plt.legend(frameon=True, fontsize=FONTSIZE, columnspacing=0.4, labelspacing=0.4, loc="lower right")
    """
    #"""Bar Plot
    ax = axes[1]
    if args.model == "olmoe":
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 37, 58]
    else:
        x = list(range(8))
    # Sort x by input_exp_id_to_probs
    x_sorted = list(range(len(x)))
    y = [input_exp_id_to_probs[i][0] * 100 for i in x]
    # ax.bar(x, y, color="#F0539B", label="Input tokens")
    # Space them out
    x_bar = np.array(x_sorted) - 0.275
    ax.bar(x_bar, y, color="#F0539B", label="Input token ID", width=0.25, alpha=0.8)
    y = [predicted_exp_id_to_probs[i][0] * 100 for i in x_sorted]
    # ax.bar(x, y, color="#43C5E0", label="Predicted output tokens")
    x_bar = np.array(x_sorted)
    ax.bar(x_bar, y, color="#43C5E0", label="Predicted output token ID", width=0.25, alpha=0.8)
    y = [gt_exp_id_to_probs[i][0] * 100 for i in x_sorted]
    # ax.bar(x, y, color="#2E3168", label="Ground-truth output tokens")
    x_bar = np.array(x_sorted) + 0.275
    ax.bar(x_bar, y, color="#2E3168", label="Ground-truth output token ID", width=0.25, alpha=0.8)
    # Draw horizontal prob lines for input_prob etc
    ax.axhline(input_prob_chosen * 100, color="#F0539B", linestyle='--', linewidth=6)
    ax.axhline(pred_prob_chosen * 100, color="#43C5E0", linestyle='--', linewidth=6)
    ax.axhline(gt_prob_chosen * 100, color="#2E3168", linestyle='--', linewidth=6)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.margins(x=0.005)
    ax.set_xticks(x_sorted, x)
    ax.set_xlabel("Expert ID", fontsize=FONTSIZE, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(f"Per expert in layer {args.layer_num}", fontsize=FONTSIZE, fontweight='bold')
    plt.legend(frameon=True, fontsize=FONTSIZE, columnspacing=0.4, labelspacing=0.4, loc="lower right")
    #"""
    plt.savefig(os.path.join(args.fig_dir, f"vocabulary_specialization_top{args.topk}_{args.model}.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(args.fig_dir, f"vocabulary_specialization_top{args.topk}_{args.model}.png"), bbox_inches='tight')


if __name__ == '__main__':
    """
    First, run the following to save model outputs:
        `python moe.py --do_inference --do_inference_all_ckpts`
        (skip `--do_inference_all_ckpts` if you'll not run ckpt analysis)

    Then, to do ckpt analysis (Router Saturation):
        `python moe.py --do_ckpt_analysis --topk 1
        python moe.py --do_ckpt_analysis --topk 8`

    To do coactivation analysis:
        `python moe.py --do_coactivation_analysis --layer_num 0
        python moe.py --do_coactivation_analysis --layer_num 7
        python moe.py --do_coactivation_analysis --layer_num 15`

    To do token analysis (Vocabulary specialization):
        `python moe.py --do_token_analysis`
        `python moe.py --do_token_analysis_layers_experts --topk 1`

    For all comments: use `--tokenized_path`, `--out_dir`, `--fig_dir` to specify where to save stuff
    To use Mixtral, use `--model mixtral` (also requires rerunning do_inference)
    """

    parser = argparse.ArgumentParser(prog="moe.py", description="Run analyses on OLMoE")
    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--do_inference_all_ckpts", action="store_true")

    parser.add_argument("--do_ckpt_analysis", action="store_true")
    parser.add_argument("--do_coactivation_analysis", action="store_true")
    parser.add_argument("--do_token_analysis", action="store_true")
    parser.add_argument("--do_token_analysis_layers", action="store_true")
    parser.add_argument("--do_token_analysis_experts", action="store_true")
    parser.add_argument("--do_token_analysis_layers_experts", action="store_true")    
 
    parser.add_argument("--tokenized_path", default="c4_validation.jsonl", type=str, help="directory to save tokenized c4 data.")
    parser.add_argument("--out_dir", default="out", type=str, help="directory to save outputs from the model")
    parser.add_argument("--fig_dir", default="figs", type=str, help="directory to save figures")
    
    parser.add_argument("--topk", choices=[1, 2, 8, 18], default=1, type=int)
    parser.add_argument("--layer_num", choices=list(range(16)), default=7, type=int)
    parser.add_argument("--model", default="olmoe", type=str, help="Which model; if not olmoe, then mixtral")

    args = parser.parse_args()
    
    if args.do_inference:
        if not os.path.exists(args.tokenized_path):
            tokenize_c4(args.tokenized_path, model=args.model)
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        do_inference(args, run_all_checkpoints=args.do_inference_all_ckpts)

    if args.do_ckpt_analysis or args.do_coactivation_analysis or args.do_token_analysis:
        if not os.path.exists(args.fig_dir):
            os.mkdir(args.fig_dir)

    if args.do_ckpt_analysis:
        do_ckpt_analysis(args)

    if args.do_coactivation_analysis:
        do_coactivation_analysis(args)

    if args.do_token_analysis:
        do_token_analysis(args)
    
    if args.do_token_analysis_layers:
        do_token_analysis_layers(args)

    if args.do_token_analysis_experts:
        do_token_analysis_experts(args)

    if args.do_token_analysis_layers_experts:
        do_token_analysis_layers_experts(args)