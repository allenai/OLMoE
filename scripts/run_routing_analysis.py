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
token = None

start_time = time.time()


# Adapted from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
# Fixed aggregating over all layers
def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2
) -> float:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=1)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-2)) / len(gate_logits)
    return overall_loss * num_experts


def load_analysis_data(tokenizer, domain, bs):
    np.random.seed(2024)
    tokens = []

    if domain == "tulu":
        data_path = f"routing_output/text/tulu-v3.1.jsonl"
        with open(data_path) as f:
            for line in f:
                text = json.loads(line)["text"]
                tokens = tokenizer(text, truncation=False)["input_ids"]
                while len(tokens) >= bs:
                    yield tokens[:bs]
                    tokens = tokens[bs:]
                yield tokens
    else:
        data_path = f"routing_output/text/{domain}_texts.txt"
        with open(data_path) as f:
            text = f.read()
            tokens = tokenizer(text, truncation=False)["input_ids"]
            while len(tokens) >= bs:
                yield tokens[:bs]
                tokens = tokens[bs:]

def load_sft_model():
    DEVICE = "cuda"
    model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-SFT", token=token).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-SFT", token=token)
    return model, tokenizer


def load_dpo_model():
    DEVICE = "cuda"
    model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct", token=token).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct", token=token)
    return model, tokenizer


def load_model():
    DEVICE = "cuda"
    model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", token=token).to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924", token=token)
    return model, tokenizer

def load_model_mistral():
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', token=token)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
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

    total_token_count = 0

    aux_losses = []

    # for each expert, what are some commen tokens that are assigned to it?
    # write a code to count that and print it out: {expert_id: Counter({token1: 32, token2: 23})...}
    for i, input_ids in tqdm(enumerate(load_analysis_data(tokenizer, domain=domain, bs=length))):
        input_ids = torch.LongTensor(input_ids).reshape(1, -1).to(DEVICE)
        out = model(input_ids=input_ids, output_router_logits=True)

        aux_loss = load_balancing_loss_func(
            out["router_logits"],
            model.num_experts,
            model.num_experts_per_tok,
        )
        aux_losses.append(aux_loss.cpu().item())

        # input id shapes: 2048 seqlen
        input_ids = input_ids[0].detach().cpu().numpy().tolist()
        total_token_count += len(input_ids)

        # 16 layer, (2048 tokens, 64 experts)
        router_logits = [l.detach().cpu().numpy() for l in out["router_logits"]]
        # 2048 tokens, 8 experts, 16 layers
        if model_name == "mistral":
            exp_ids = np.stack([np.argsort(-logits, -1)[:, :2].tolist() for logits in router_logits], -1)
        elif model_name.startswith("olmoe"):
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

        if total_token_count > 204800:
            break

    print(f"Average aux loss: {np.mean(aux_losses)}")

    return layer_counters, crosslayer_counters, eid2token_layer0, eid2token_layer7, eid2token_layer15

name2finaldata = {"github_oss_with_stack": "github", "arxiv": "arxiv", "c4": "c4", "b3g": "book", "wikipedia": "wikipedia", "tulu": "tulu"}

if __name__=='__main__':
    model_name = "olmoe"
    print(model_name)
    if model_name == "mistral":
        model, tokenizer = load_model_mistral()
    elif model_name == "olmoe-sft":
        model, tokenizer = load_sft_model()
    elif model_name == "olmoe-dpo":
        model, tokenizer = load_dpo_model()
    elif model_name == "olmoe":
        model, tokenizer = load_model()

    length = 2048
    for domain in tqdm(["tulu", "github_oss_with_stack", "arxiv", "c4", "b3g", "wikipedia"]):
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