from functools import partial
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
from intervention_trainer import make_complex_position_supervised_data_module, DASTrainerForCausalLM
from utils import (
    SEED,
    ASPECT_KEYWORDS,
    set_seed,
    create_input,
    create_input_with_example
)
set_seed(SEED)

DATASET_NAME = "CEBaB/CEBaB"

def create_toy_dataset(
    model,
    tokenizer,
    num_interventions: int,
    positions: str = "f1+l1",
    nonstop: bool = True,
    share_weights: bool = True,
    intervention_offset: int = 0,
    evaluation: bool = False
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
        intervention_offset=intervention_offset,
        evaluation=evaluation
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
    aspect: str = "service",
    num_sources_per_cf: int = 10,
    categories: bool = ['Positive', 'Negative', 'unknown'],
    keyword_match: bool = False,
    keyword_match_first: bool = True,
    prompt_with_example: bool = False,
    assistant_prefix: str = "<|assistant|>",
    evaluation: bool = False
):
    train_dataset = load_dataset("CEBaB/CEBaB", split=dataset_split)
    # filter out examples where there is no majority aspect (otherwise can't use examples in prompt)
    if prompt_with_example:
        train_dataset = train_dataset.filter(
            lambda d: 
                d['food_aspect_majority'] not in ['no majority', ''] and \
                d['service_aspect_majority'] not in ['no majority', ''] and \
                d['noise_aspect_majority'] not in ['no majority', ''] and \
                d['ambiance_aspect_majority'] not in ['no majority', '']
        )

    aspect_key = f'{aspect}_aspect_majority'
    
    train_df = train_dataset.to_pandas()
    create_input_fn = partial(create_input_with_example, train_df) if prompt_with_example else create_input
    train_dataset = train_dataset.map(create_input_fn)
    # filter out examples with no messages
    train_dataset = train_dataset.filter(lambda x: x['messages'] is not None)
    edit_dataset = train_dataset.filter(lambda x: x['edit_type'] == aspect)

    df = train_dataset.to_pandas()

    # apply filters
    df = df[df[aspect_key].isin(categories)]
    if keyword_match:
        df = df[df['description'].apply(lambda x: any(k in x.lower() for k in ASPECT_KEYWORDS[aspect]))]

    data = []
    for cf in tqdm(edit_dataset):
        # skip instances where the aspect is not edited to the goal
        if cf[aspect_key] != cf['edit_goal']:
            continue
        # skip instances where the aspect is not mentioned in the description (by keyword)
        if keyword_match and not any(k in cf['description'].lower() for k in ASPECT_KEYWORDS[aspect]):
            continue

        original_id = cf['original_id'] + '000000' if cf['original_id'] != '0' else '0'
        base = df[df['id'] == original_id]
        # skip instances where we can't find the original description (might be due to keyword filtering)
        if base.shape[0] == 0:
            continue
        base = base.iloc[0]
        sources = df[df[aspect_key] == cf[aspect_key]]
        if sources.shape[0] > num_sources_per_cf:
            sources = sources.sample(num_sources_per_cf, replace=False)
        
        for _, source in sources.iterrows():
            source_tokens = tokenizer.apply_chat_template(source['messages'], tokenize=False)
            base_tokens = tokenizer.apply_chat_template(base['messages'], tokenize=False)
            cf_tokens = tokenizer.apply_chat_template(cf['messages'], tokenize=False)

            if keyword_match:
                # split input-output by last matching keyword (where the input contains the keyword)
                def split_by_keyword(tokens, keywords, first=False):
                    # ignore case when matching keywords (shouldn't affect indices)
                    keyword_indices = [
                        tokens.lower().find(k) + len(k) for k in keywords if k in tokens.lower()
                    ]
                    assert keyword_indices, f"No {aspect}-aspect keywords found in {tokens}"
                    split_index = min(keyword_indices) if first else max(keyword_indices)
                    return tokens[:split_index], tokens[split_index:]
                base_inputs, base_outputs = split_by_keyword(
                    base_tokens, ASPECT_KEYWORDS[aspect], first=keyword_match_first
                )
                _, cf_outputs = split_by_keyword(
                    cf_tokens, ASPECT_KEYWORDS[aspect], first=keyword_match_first
                )
                source_inputs, source_outputs = split_by_keyword(
                    source_tokens, ASPECT_KEYWORDS[aspect], first=keyword_match_first
                )
            else:
                # split input-output by assistant prefix
                base_inputs, base_outputs = base_tokens.split(assistant_prefix)
                base_inputs = base_inputs + assistant_prefix
                _, cf_outputs = cf_tokens.split(assistant_prefix) # same inputs as base
                source_inputs, source_outputs = source_tokens.split(assistant_prefix)
                source_inputs = source_inputs + assistant_prefix

            data.append({
                'base_input': base_inputs,
                'base_output': base_outputs,
                'cf_output': cf_outputs,
                'source_input': source_inputs,
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
        intervention_offset=intervention_offset,
        evaluation=evaluation
    )

def main(
    # pyvene args
    model_name_or_path: str = "llama2_7b", # from models.yaml
    use_flash_attention: bool = False,
    layers: str = "18;28",
    subspace_dim: int = 128,
    rank: int = 4,
    positions: str = "f5+l5",
    share_weights: bool = False,
    dropout: float = 0.,
    intervention_offset: int = 0,
    # dataset args
    toy_dataset: bool = False,
    train_split: str = "train_inclusive",
    eval_split: str = "validation",
    aspect: str = "service",
    num_sources_per_cf: int = 10,
    categories: bool = ['Positive', 'Negative', 'unknown'],
    keyword_match: bool = False,
    keyword_match_first: bool = True,
    prompt_with_example: bool = False,
    assistant_prefix: str = "<|assistant|>",
    # training args
    num_train_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 10,
    max_seq_length: int = 2048,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "linear",
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    # logging args
    logging_steps: int = 10,
    output_dir: str = "adv_reft",
    save_model: bool = False,
    use_wandb: bool = False,
    # evaluation args
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    # commandline args
    args: argparse.Namespace = None
):
    print(
        "Training intervention on CEBaB\n"
        "------------------------------\n"
        f"Pyvene args:\n"
        f"  model_name_or_path: {model_name_or_path}\n"
        f"  subspace_dim: {subspace_dim}\n"
        f"  layers: {layers}\n"
        f"  rank: {rank}\n"
        f"  positions: {positions}\n"
        f"  share_weights: {share_weights}\n"
        f"  dropout: {dropout}\n"
        f"Dataset args:\n"
        f"  toy_dataset: {toy_dataset}\n"
        f"  train_split: {train_split}\n"
        f"  eval_split: {eval_split}\n"
        f"  aspect: {aspect}\n"
        f"  num_sources_per_cf: {num_sources_per_cf}\n"
        f"  categories: {categories}\n"
        f"  keyword_match: {keyword_match}\n"
        f"  keyword_match_first: {keyword_match_first}\n"
        f"  prompt_with_example: {prompt_with_example}\n"
        f"  assistant_prefix: {assistant_prefix}\n"
        f"Training args:\n"
        f"  num_train_epochs: {num_train_epochs}\n"
        f"  learning_rate: {learning_rate}\n"
        f"  batch_size: {batch_size}\n"
        f"  max_seq_length: {max_seq_length}\n"
        f"  warmup_ratio: {warmup_ratio}\n"
        f"  lr_scheduler_type: {lr_scheduler_type}\n"
        f"  max_grad_norm: {max_grad_norm}\n"
        f"  gradient_accumulation_steps: {gradient_accumulation_steps}\n" 
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else None
    )
    # get tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.unk_token

    if "gpt2" in model_name_or_path:
        pv.type_to_dimension_mapping[type(model)] = {
            "block_output": ("hidden_size",),
            "transformer.wte.output": ("hidden_size",),
            **{
                f"transformer.h[{i}].output": ("hidden_size",) for i in range(model.config.num_hidden_layers)
            }
        }
    else:
        # add Phi3 to pyvene library
        pv.type_to_dimension_mapping[type(model)] = {
            "block_output": ("hidden_size",),
            "model.embed_tokens.output": ("hidden_size",),
            **{
                f"model.layers[{i}].output": ("hidden_size",) for i in range(model.config.num_hidden_layers)
            }
        }

    def get_representation(layer, rank):
        is_gpt2 = "gpt2" in model_name_or_path
        embed_component = "transformer.wte.output" if is_gpt2 else "model.embed_tokens.output"
        layer_component = f"transformer.h[{int(layer)}].output" if is_gpt2 else f"model.layers[{int(layer)}].output"
        component = embed_component if layer == "embedding" else layer_component
        return {
            "component": component,
            "low_rank_dimension": rank,
            "subspace_partition": [[0, subspace_dim], [subspace_dim, model.config.hidden_size]],
        }
    
    if layers == "all":
        layers = [l for l in range(model.config.num_hidden_layers)]
    else:
        layers = layers.split(";")
    representations = [get_representation(layer, rank) for layer in layers]
    
    # if not share_weights, double the layers (diff for first and last positions)
    if "+" in positions and not share_weights:
        layers += layers

    # get train model (reft or lora)
    reft_config = pv.IntervenableConfig(
        representations=representations,
        intervention_types=pv.RotatedSpaceIntervention
    )
    train_model = pyreft.get_reft_model(model, reft_config)
    train_model.set_device(device)
    print('Number of interventions:', len(reft_config.representations))

    train_model.print_trainable_parameters()

    if toy_dataset:
        train_data_module = create_toy_dataset(
            model=model,
            tokenizer=tokenizer,
            num_interventions=len(representations),
            positions=positions,
            nonstop=True,
            share_weights=share_weights,
            intervention_offset=intervention_offset
        )

        eval_data_module = create_toy_dataset(
            model=model,
            tokenizer=tokenizer,
            num_interventions=len(representations),
            positions=positions,
            nonstop=True,
            share_weights=share_weights,
            intervention_offset=intervention_offset,
            evaluation=True
        )
    else:
        train_data_module = create_dataset(
            model=model,
            tokenizer=tokenizer,
            num_interventions=len(representations),
            positions=positions,
            nonstop=True,
            share_weights=share_weights,
            dataset_split=train_split,
            aspect=aspect,
            intervention_offset=intervention_offset,
            num_sources_per_cf=num_sources_per_cf,
            categories=categories,
            keyword_match=keyword_match,
            keyword_match_first=keyword_match_first,
            prompt_with_example=prompt_with_example,
            assistant_prefix=assistant_prefix,
            evaluation=False
        )

        eval_data_module = create_dataset(
            model=model,
            tokenizer=tokenizer,
            num_interventions=len(representations),
            positions=positions,
            nonstop=True,
            share_weights=share_weights,
            dataset_split=eval_split,
            aspect=aspect,
            intervention_offset=intervention_offset,
            num_sources_per_cf=num_sources_per_cf,
            categories=categories,
            keyword_match=keyword_match,
            keyword_match_first=keyword_match_first,
            prompt_with_example=prompt_with_example,
            assistant_prefix=assistant_prefix,
            evaluation=True
        )

    report_to = "none"
    if use_wandb:
        wandb.init(project="das_cebab", config=vars(args))
        report_to = "wandb"

    # create run ID from timestamp
    model_name = model_name_or_path.replace("/", "_")
    run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = f"{output_dir}/{model_name}/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = transformers.TrainingArguments(
        num_train_epochs=num_train_epochs,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        remove_unused_columns=False,
        evaluation_strategy="no",
        report_to=report_to,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # try to avoid OOM errors
        dataloader_pin_memory=False
    )

    data_module = {
        **train_data_module,
        'eval_dataset': eval_data_module['train_dataset']
    }
    trainer = DASTrainerForCausalLM(
        model=train_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    trainer.train()

    # change padding side to left for evaluation
    tokenizer.padding_side = "left"
    eval_data = trainer.evaluate(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature
    )

    eval_df = pd.DataFrame(eval_data, columns=["original_output", "das_output", "base", "source"])
    eval_df.to_csv(f"{output_dir}/eval.csv", index=False)

    if save_model:
        train_model.save(f"{output_dir}/weights")
    
    with open(f"{output_dir}/config.json", "w+") as f:
        config = vars(args)
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a defensive intervention on AdvBench.")
    # das args
    parser.add_argument("--model_name_or_path", type=str, default="llama2_7b")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--layers", type=str, default="18;28")
    parser.add_argument("--subspace_dim", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--positions", type=str, default="f5+l5")
    parser.add_argument("--share_weights", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--intervention_offset", type=int, default=0)
    # dataset args
    parser.add_argument("--toy_dataset", action="store_true")
    parser.add_argument("--train_split", type=str, default="train_inclusive")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--aspect", type=str, default="service")
    parser.add_argument("--num_sources_per_cf", type=int, default=10)
    parser.add_argument("--categories", nargs="+", default=['Positive', 'Negative', 'unknown'])
    parser.add_argument("--keyword_match", action="store_true")
    parser.add_argument("--keyword_match_first", action="store_true")
    parser.add_argument("--prompt_with_example", action="store_true")
    parser.add_argument("--assistant_prefix", type=str, default="<|assistant|>")
    # training args
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # evaluation args
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    # logging args
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="adv_reft")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()
    
    main(**vars(args), args=args) # unpack args to pass as kwargs