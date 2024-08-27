from transformers import OlmoeForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import time
import pickle as pkl
import json
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
token = "YOUR_TOKEN"
final_revision = "step1223842-tokens5100B"

start_time = time.time()

def load_analysis_data(tokenizer, domain, bs):
    np.random.seed(2024)
    tokens = []
    data_path = f"routing_output/text/{domain}_texts.txt"

    with open(data_path) as f:
        text = f.read()
        tokens = tokenizer(text, truncation=False)["input_ids"]
        while len(tokens) >= bs:
            yield tokens[:bs]
            tokens = tokens[bs:]


def load_model(revision=final_revision):
    DEVICE = "cuda"
    model = OlmoeForCausalLM.from_pretrained("OLMoE/OLMoE-1B-7B-0824").to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("OLMoE/OLMoE-1B-7B-0824")
    return model, tokenizer

def load_model_mistral():
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def print_expert_percentage(exp_counts):
    total = sum(exp_counts.values())
    for eid, ecount in exp_counts.most_common():
        print(f"Expert {eid}: {ecount/total*100:.2f}")


def run_analysis(domain, model_name=None):
    layer_counters = defaultdict(Counter)
    crosslayer_counters = defaultdict(Counter)

    eid2token_layer0 = defaultdict(Counter)
    eid2token_layer7 = defaultdict(Counter)
    eid2token_layer15 = defaultdict(Counter)

    # for each expert, what are some commen tokens that are assigned to it?
    # write a code to count that and print it out: {expert_id: Counter({token1: 32, token2: 23})...}
    for i, input_ids in tqdm(enumerate(load_analysis_data(tokenizer, domain=domain, bs=length))):
        input_ids = torch.LongTensor(input_ids).reshape(1, -1).to(DEVICE)
        out = model(input_ids=input_ids, output_router_logits=True)

        # input id shapes: 2048 seqlen
        input_ids = input_ids[0].detach().cpu().numpy().tolist()
        logits = out["logits"][0].detach().cpu().numpy()
        # 16 layer, (2048 tokens, 64 experts)
        router_logits = [l.detach().cpu().numpy() for l in out["router_logits"]]
        # 2048 tokens, 8 experts, 16 layers
        if model_name == "mistral":
            exp_ids = np.stack([np.argsort(-logits, -1)[:, :2].tolist() for logits in router_logits], -1)
        elif model_name == "olmoe":
            exp_ids = np.stack([np.argsort(-logits, -1)[:, :8].tolist() for logits in router_logits], -1)
        # 2048 tokens, 8 experts
        exp_ids_layer0 = exp_ids[:, :, 0]
        exp_ids_layer7 = exp_ids[:, :, 7]
        exp_ids_layer15 = exp_ids[:, :, 15]

        for id, token in enumerate(input_ids):
            experts = exp_ids_layer0[id, :]
            for e in experts:
                eid2token_layer0[e][token] += 1
            for e in exp_ids_layer7[id, :]:
                eid2token_layer7[e][token] += 1
            for e in exp_ids_layer15[id, :]:
                eid2token_layer15[e][token] += 1

        for layer in range(exp_ids.shape[2]):
            exp_counts = Counter(exp_ids[:, :, layer].flatten())
            layer_counters[layer].update(exp_counts)

        for layer_i in range(exp_ids.shape[2] - 1):
            for layer_j in range(exp_ids.shape[2]):
                exps_counts = Counter(zip(exp_ids[:, :, layer_i].flatten(), exp_ids[:, :, layer_j].flatten()))
                crosslayer_counters[(layer_i, layer_j)].update(exps_counts)

        if i > 100:
            break

    return layer_counters, crosslayer_counters, eid2token_layer0, eid2token_layer7, eid2token_layer15

name2finaldata = {"github_oss_with_stack": "github", "arxiv": "arxiv", "c4": "c4", "b3g": "book", "wikipedia": "wikipedia"}

if __name__=='__main__':
    model_name = "olmoe"
    print(model_name)
    if model_name == "mistral":
        model, tokenizer = load_model_mistral()
    elif model_name == "olmoe":
        model, tokenizer = load_model(revision=final_revision)

    length = 2048
    # save the expert counts for each domain into a json file
    for domain in tqdm(["github_oss_with_stack", "arxiv", "c4", "b3g", "wikipedia"]):
        print(f"Domain: {domain}")
        layer_counters, crosslayer_counters, eid2token_layer0, eid2token_layer7, eid2token_layer15 = run_analysis(domain, model_name)
        Path(f"routing_output/{model_name}/expert_counts").mkdir(parents=True, exist_ok=True)
        Path(f"routing_output/{model_name}/expert_counts_crosslayer").mkdir(parents=True, exist_ok=True)
        Path(f"routing_output/{model_name}/eid2token").mkdir(parents=True, exist_ok=True)
        with open(f"routing_output/{model_name}/expert_counts/{name2finaldata[domain]}.pkl", "wb") as f:
            pkl.dump([layer_counters[0], layer_counters[7], layer_counters[15]], f)
        with open(f"routing_output/{model_name}/expert_counts_crosslayer/{name2finaldata[domain]}.pkl", "wb") as f:
            pkl.dump([crosslayer_counters[(0, 7)], crosslayer_counters[(7, 15)]], f)
        with open(f"routing_output/{model_name}/eid2token/{name2finaldata[domain]}.pkl", "wb") as f:
            pkl.dump([eid2token_layer0, eid2token_layer7, eid2token_layer15], f)