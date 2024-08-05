import argparse
import datetime
from functools import partial
import os
from typing import Optional
import numpy as np
import pandas as pd
import torch
from tqdm import trange
import wandb
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from utils import (
    get_original,
    create_input_zero_shot,
    create_input_validation, 
    LABELS, 
    SEED, 
    set_seed
)
set_seed(SEED)

def main(
    # model arguments
    model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct",
    use_flash_attention: bool = False,
    # eval arguments
    dataset_split: Optional[str] = None,
    output_dir: str = "inst_tune",
    per_device_eval_batch_size: int = 8,
    aspect: str = "service",
    max_new_tokens: int = 256,
    num_return_sequences: int = 10,
    assistant_prefix: str = "<|assistant|>",
    assistant_suffix: str = "<|end|>",
    # classification arguments
    classifier_name_or_path: str = "train_classifier",
    **generate_kwargs
):
    #####################
    # Evaluation        #
    #####################
    # Load the model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_flash_attention else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        # padding_side="right"
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    # set up unique output directory
    run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(output_dir, run_id)

    if dataset_split is None:
        dataset_split = "train_inclusive"
    dataset = load_dataset("CEBaB/CEBaB", split=dataset_split)
    # focus on where aspect is explicitly mentioned
    dataset = dataset.filter(
        lambda d: d[f'{aspect}_aspect_majority'] in ['Positive', 'Negative']
    )
    df = pd.DataFrame(dataset)
    edited_df = df[(~df['is_original']) & (df['edit_type'] == aspect)]

    def generate(batch):
        examples = [
            create_input_validation(
                create_input_zero_shot,
                tokenizer,
                example,
            )
            for example in batch
        ]
        inputs = tokenizer(
            examples,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_return_sequences,
                **generate_kwargs
            )
        generations_batch = tokenizer.batch_decode(outputs)
        data_batch = [b for b in batch for _ in range(num_return_sequences)]
        prompts_batch = [e for e in examples for _ in range(num_return_sequences)]
        return data_batch, prompts_batch, generations_batch
    
    def perplexity(batch, apply_chat_template=True):
        if apply_chat_template:
            examples = [
                tokenizer.apply_chat_template(
                    create_input_zero_shot(example)['messages'], 
                    tokenize=False,
                )
                for example in batch
            ]
        else:
            examples = batch
        inputs = tokenizer(
            examples,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs['input_ids']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ces = []
            for i in range(inputs['input_ids'].shape[0]):
                ce = torch.nn.functional.cross_entropy(
                    shift_logits[i], shift_labels[i], reduction='mean'
                )
                ces.append(ce)
        perplexities = torch.exp(torch.stack(ces)).cpu().tolist()
        if apply_chat_template:
            # repeat perplexities for each generated sequence
            perplexities = [
                p for p in perplexities for _ in range(num_return_sequences)
            ]
        return perplexities

    data = []
    prompts = []
    generations = []
    perplexities_cebab = []
    perplexities_gen = []
    # iterate over edited datapoints
    for b in trange(0, len(edited_df), per_device_eval_batch_size, desc="Generating..."):
        batch_cf = df.iloc[b:b+per_device_eval_batch_size].to_dict(orient="records")
        batch_or = [
            get_original(df, example, return_description=False)
            for example in batch_cf
        ]
        filtered_batch = [
            (cf, org.to_dict()) for cf, org in zip(batch_cf, batch_or) if org is not None
        ]
        if len(filtered_batch) == 0:
            continue
        batch_cf, batch_or = zip(*filtered_batch)
        # generate completions
        data_cf, prompts_cf, generations_cf = generate(batch_cf)
        data_or, prompts_or, generations_or = generate(batch_or)
        # compute perplexities
        perplexities_cf = perplexity(batch_cf, apply_chat_template=True)
        perplexities_or = perplexity(batch_or, apply_chat_template=True)
        perplexities_cf_gen = perplexity(generations_cf, apply_chat_template=False)
        perplexities_or_gen = perplexity(generations_or, apply_chat_template=False)
        # save the data
        data += data_cf + data_or
        prompts += prompts_cf + prompts_or
        generations += generations_cf + generations_or
        perplexities_cebab += perplexities_cf + perplexities_or
        perplexities_gen += perplexities_cf_gen + perplexities_or_gen

    df = pd.DataFrame(data)
    df['generation'] = generations
    df['prompt'] = prompts
    df['perplexity_cebab'] = perplexities_cebab
    df['perplexity_generation'] = perplexities_gen
    df['text'] = df['generation'].map(
        lambda x: x.split(assistant_prefix)[1].split(assistant_suffix)[0].strip()
    )

    # save the generations
    os.makedirs(f"{output_dir}/eval", exist_ok=True)
    df.to_csv(f"{output_dir}/eval/generations.csv", index=False)

    #####################
    # Classification    #
    #####################

    # done with train/evaluation, free up memory
    del model, tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(classifier_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier_name_or_path,
        device_map=device
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    predictions = []
    for b in trange(0, df.shape[0], per_device_eval_batch_size):
        batch = df.iloc[b:b+per_device_eval_batch_size]
        examples = batch['text'].tolist()
        inputs = tokenizer(examples, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**inputs)
        predictions += outputs.logits.argmax(dim=1).tolist()

    df[f'{aspect}_label'] = [LABELS[p] for p in predictions]
    # overwrite the generations with the classifications
    df.to_csv(f"{output_dir}/eval/generations.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--dataset_split", type=str, default="train_inclusive")
    parser.add_argument("--output_dir", type=str, default="inst_tune")
    parser.add_argument("--aspect", type=str, default="service")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--assistant_prefix", type=str, default="<|assistant|>")
    parser.add_argument("--assistant_suffix", type=str, default="<|end|>")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--classifier_name_or_path", type=str, default="train_classifier")
    args = parser.parse_args()
    main(**vars(args))