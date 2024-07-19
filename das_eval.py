import os
import argparse
import itertools
import json
from tqdm import tqdm
import torch
import transformers
import datetime
import numpy as np
import pandas as pd
import pyvene as pv
import pyreft
import yaml
import random
import wandb
from datasets import load_dataset
from intervention_trainer import make_complex_position_supervised_data_module, InterventionTrainerForCausalLM

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET_NAME = "CEBaB/CEBaB"

ASSISTANT_PREFIX = '<|assistant|>\n'

PROMPT_TEMPLATE = """Write a short restaurant review for the following restaurant:
Name: {restaurant_name}
Cuisine: {cuisine}
Price tier: {price_tier}
Dining style: {dining_style}
Region: {region}"""

def create_example(example):
    # convert string to dict (VERY UNSAFE, DO NOT USE ON UNTRUSTED DATA)
    metadata = eval(example['opentable_metadata'])
    prompt = PROMPT_TEMPLATE.format(**metadata)
    completion = example['description']
    messages = [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': completion}
    ]
    return {'messages': messages}

def create_toy_dataset(
    model,
    tokenizer,
    num_interventions: int,
    positions: str = "f1+l1",
    nonstop: bool = True,
    share_weights: bool = True,
    intervention_offset: int = 0
):
    prompt = "When {subject} and {object} went to the store, {subject} gave a drink to"
    names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Helen", "Ivy", "Jack"]
    # source subject, source object, base subject, base object
    dataset = list(itertools.product(names, repeat=4))
    dataset = [d for d in dataset if d[0] != d[1] and d[2] != d[3] and d[1] != d[3]]
    random.shuffle(dataset)
    data = []
    for d in dataset:
        source = prompt.format(subject=d[0], object=d[1])
        base = prompt.format(subject=d[2], object=d[3])
        base_output = f" {d[3]}" # base object
        source_output = f" {d[1]}" # source object
        cf = f" {d[1]}" # same object as source
        data.append({
            'base_input': base,
            'base_output': base_output,
            'cf_output': cf,
            'source_input': source,
            'source_output': source_output
        })
    
    base_inputs = [d['base_input'] for d in data]
    cf_outputs = [d['cf_output'] for d in data]
    source_inputs = [d['source_input'] for d in data]
    source_outputs = [d['source_output'] for d in data]

    return make_complex_position_supervised_data_module(
        tokenizer,
        model,
        base_inputs,
        cf_outputs,
        source_inputs,
        source_outputs,
        positions=positions,
        num_interventions=num_interventions,
        nonstop=nonstop,
        share_weights=share_weights,
        intervention_offset=intervention_offset
    )

def create_dataset(
    model,
    tokenizer,
    num_interventions: int,
    positions: str = "f1+l1",
    nonstop: bool = True,
    share_weights: bool = True,
    intervention_offset: int = 0,
    dataset_split: str = "train_inclusive",
    aspect: str = "service"
):
    aspect_key = f'{aspect}_aspect_majority'

    train_dataset = load_dataset(DATASET_NAME, split=dataset_split)
    train_dataset = train_dataset.map(create_example)
    edit_dataset = train_dataset.filter(lambda x: x['edit_type'] == aspect)

    df = pd.DataFrame(train_dataset)

    data = []
    for cf in tqdm(edit_dataset):
        source = df[df[aspect_key] == cf['edit_goal']].sample().iloc[0]
        original_id = cf['original_id'] + '000000' if cf['original_id'] != '0' else '0'
        base = df[df['id'] == original_id].iloc[0]

        source_tokens = tokenizer.apply_chat_template(source['messages'], tokenize=False)
        base_tokens = tokenizer.apply_chat_template(base['messages'], tokenize=False)
        cf_tokens = tokenizer.apply_chat_template(cf['messages'], tokenize=False)

        base_inputs, base_outputs = base_tokens.split(ASSISTANT_PREFIX)
        _, cf_outputs = cf_tokens.split(ASSISTANT_PREFIX) # same inputs as base
        source_inputs, source_outputs = source_tokens.split(ASSISTANT_PREFIX)

        data.append({
            'base_input': base_inputs + ASSISTANT_PREFIX,
            'base_output': base_outputs,
            'cf_output': cf_outputs,
            'source_input': source_inputs + ASSISTANT_PREFIX,
            'source_output': source_outputs
        })
    
    base_inputs = [d['base_input'] for d in data]
    cf_outputs = [d['cf_output'] for d in data]
    source_inputs = [d['source_input'] for d in data]
    source_outputs = [d['source_output'] for d in data]

    return make_complex_position_supervised_data_module(
        tokenizer,
        model,
        base_inputs,
        cf_outputs,
        source_inputs,
        source_outputs,
        positions=positions,
        num_interventions=num_interventions,
        nonstop=nonstop,
        share_weights=share_weights,
        intervention_offset=intervention_offset
    )

def evaluate(model, tokenizer, data_module, **generate_kwargs):
    model.eval()
    device = model.get_device()
    eval_data = []
    for d in tqdm(data_module['train_dataset'], desc="Evaluating"):
        with torch.no_grad():
            original_output, das_output = model.generate(
                {
                    "input_ids": torch.tensor(d["input_ids"]).unsqueeze(0).to(device),
                },
                sources=[{
                    "input_ids": torch.tensor(d["source_input_ids"]).unsqueeze(0).to(device),
                }],
                unit_locations={"sources->base": (
                    # copy from
                    torch.tensor([d['source_intervention_locations']]).permute(1, 0, 2).tolist(),
                    # paste to
                    torch.tensor([d["intervention_locations"]]).permute(1, 0, 2).tolist()
                )},
                subspaces=0,
                output_original_output=True,
                use_cache=False,
                **generate_kwargs
            )
            eval_data.append({
                "original_output": tokenizer.decode(original_output[0], skip_special_tokens=True),
                "das_output": tokenizer.decode(das_output[0], skip_special_tokens=True),
                "base": tokenizer.decode(d["input_ids"], skip_special_tokens=True),
                "source": tokenizer.decode(d["source_input_ids"], skip_special_tokens=True),
            })
    return eval_data

def print_trainable_parameters(model):
    if isinstance(model, pyreft.ReftModel):
        model.print_trainable_parameters()
        return
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def main(
    # pyvene args
    model_name_or_path: str = "llama2_7b", # from models.yaml
    das_path: str = "llama2_7b",
    positions: str = "f5+l5",
    share_weights: bool = False,
    intervention_offset: int = 0,
    # dataset args
    toy_dataset: bool = False,
    dataset_split: str = "train_inclusive",
    aspect: str = "service",
    # generation args
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    # logging args
    output_dir: str = "das_eval",
    # commandline args
    args: argparse.Namespace = None
):
    print(
        "Training intervention on CEBaB\n"
        "------------------------------\n"
        f"Pyvene args:\n"
        f"  model_name_or_path: {model_name_or_path}\n"
        f"  das_path: {das_path}\n"
        f"  positions: {positions}\n"
        f"  share_weights: {share_weights}\n"
        f"  intervention_offset: {intervention_offset}\n"
        f"Dataset args:\n"
        f"  toy_dataset: {toy_dataset}\n"
        f"  dataset_split: {dataset_split}\n"
        f"  aspect: {aspect}\n"
        f"Generation args:\n"
        f"  max_new_tokens: {max_new_tokens}\n"
        f"  do_sample: {do_sample}\n"
        f"  temperature: {temperature}\n"
        f"Logging args:\n"
        f"  output_dir: {output_dir}\n"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.unk_token

    print('Model class:', type(model))
    # add Phi3 to pyvene library
    pv.type_to_dimension_mapping[type(model)] = {
        "block_output": ("hidden_size",),
        "model.embed_tokens.output": ("hidden_size",),
        **{
            f"model.layers[{int(i)}].output": ("hidden_size",) for i in range(model.config.num_hidden_layers)
        }
    }

    train_model = pv.IntervenableModel.load(
        das_path,
        model
    )
    train_model.set_device(device)

    print('Loaded model. Number of interventions:', len(train_model.interventions))

    if toy_dataset:
        data_module = create_toy_dataset(
            model=model,
            tokenizer=tokenizer,
            num_interventions=len(train_model.interventions),
            positions=positions,
            nonstop=True,
            share_weights=share_weights,
            intervention_offset=intervention_offset
        )
    else:
        data_module = create_dataset(
            model=model,
            tokenizer=tokenizer,
            num_interventions=len(train_model.interventions),
            positions=positions,
            nonstop=True,
            share_weights=share_weights,
            dataset_split=dataset_split,
            aspect=aspect,
            intervention_offset=intervention_offset
        )

    eval_data = evaluate(
        train_model,
        tokenizer,
        data_module,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature
    )
    
    # create run ID from timestamp
    model_name = model_name_or_path.replace("/", "_")
    run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = f"{output_dir}/{model_name}/{run_id}"

    os.makedirs(output_dir, exist_ok=True)

    eval_df = pd.DataFrame(eval_data, columns=["original_output", "das_output", "base", "source"])
    eval_df.to_csv(f"{output_dir}/eval.csv", index=False)
    
    with open(f"{output_dir}/config.json", "w+") as f:
        json.dump(vars(args), f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a defensive intervention on AdvBench.")
    # das args
    parser.add_argument("--model_name_or_path", type=str, default="llama2_7b")
    parser.add_argument("--das_path", type=str, default="llama2_7b")
    parser.add_argument("--positions", type=str, default="f5+l5")
    parser.add_argument("--share_weights", action="store_true")
    parser.add_argument("--intervention_offset", type=int, default=0)
    # dataset args
    parser.add_argument("--toy_dataset", action="store_true")
    parser.add_argument("--dataset_split", type=str, default="train_inclusive")
    parser.add_argument("--aspect", type=str, default="service")
    # generation args
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    # logging args
    parser.add_argument("--output_dir", type=str, default="adv_reft")

    args = parser.parse_args()
    
    main(**vars(args), args=args) # unpack args to pass as kwargs