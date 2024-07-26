import argparse
import datetime
import json
import random
import os
import torch
from tqdm import trange
import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

SEED = 42
random.seed(SEED)

PROMPT_TEMPLATE = """Write a short restaurant review for the following restaurant:
Name: {restaurant_name}
Cuisine: {cuisine}
Price tier: {price_tier}
Dining style: {dining_style}
Region: {region}"""

LABELS = ['Negative', 'Positive', 'unknown']

def create_example(tokenizer, example, prefill_generation=False):
    # convert string to dict (VERY UNSAFE, DO NOT USE ON UNTRUSTED DATA)
    metadata = eval(example['opentable_metadata'])
    prompt = PROMPT_TEMPLATE.format(**metadata)
    messages = [
        {'role': 'user', 'content': prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True,
    )
    if prefill_generation:
        inputs += example['prefix']
    return inputs

def get_original(df, example):
    original_id = example['original_id'] + '000000' if example['original_id'] != '0' else '0'
    base = df[df['id'] == original_id]
    assert base.shape[0] == 1
    base = base.iloc[0]
    return base['description']

def get_prefix(example):
    for i in range(min(len(example['original_description']), len(example['description']))):
        if example['original_description'][i] != example['description'][i]:
            break
    return example['original_description'][:i]

def main(
    model_name_or_path: str = "inst_tune",
    output_dir: str = "inst_gens",
    dataset_split: str = "train_inclusive",
    num_generations_per_example: int = 10,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    prefill_generation: bool = False,
    aspect: str = "service",
    use_flash_attention: bool = True,
    # post-processing arguments
    assistant_prefix: str = "<|assistant|>",
    assistant_suffix: str = "<|end|>",
    # classification arguments
    classifier_name_or_path: str = "train_classifier",
    **generate_kwargs
):
    #######################
    # Generate examples   #
    #######################
    print('Generating examples...')
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
        padding_side="left"
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    train_dataset = load_dataset("CEBaB/CEBaB", split=dataset_split)
    if prefill_generation:
        original_df = pd.DataFrame(train_dataset)
        df = original_df[original_df['edit_type'] == aspect].copy()
        df['original_description'] = df.apply(lambda x: get_original(original_df, x), axis=1)
        df['prefix'] = df.apply(get_prefix, axis=1)
    else:
        df = pd.DataFrame(train_dataset)
        df['restaurant_id'] = df['opentable_metadata'].map(lambda x: eval(x)['restaurant_id'])
        df = df.drop_duplicates(subset='restaurant_id').reset_index(drop=True)

    data = []
    generations = []
    for b in trange(0, len(df), batch_size, desc="Generating..."):
        batch = df.iloc[b:b+batch_size].to_dict(orient="records")
        examples = [create_example(tokenizer, example, prefill_generation=prefill_generation) for example in batch]
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
                num_return_sequences=num_generations_per_example,
                **generate_kwargs
            )
            generations += tokenizer.batch_decode(outputs)
            data += [b for b in batch for _ in range(num_generations_per_example)]
    
    df = pd.DataFrame(data)
    df['generation'] = generations

    #########################
    # Classify examples     #
    #########################
    print('Classifying examples...')
    # delete model and tokenizer to free up memory
    del model, tokenizer
    torch.cuda.empty_cache()

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

    df['text'] = df['generation'].map(
        lambda x: x.split(assistant_prefix)[1].split(assistant_suffix)[0].strip()
    )

    predictions = []
    for b in trange(0, df.shape[0], batch_size):
        batch = df.iloc[b:b+batch_size]
        examples = batch['text'].tolist()
        inputs = tokenizer(examples, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**inputs)
        predictions += outputs.logits.argmax(dim=1).tolist()

    df[f'{aspect}_labels'] = [LABELS[p] for p in predictions]

    run_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(output_dir, run_id)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "generations.csv"), index=False)

    with open(os.path.join(output_dir, "config.json"), "w+") as f:
        config = {
            "model_name_or_path": model_name_or_path,
            "output_dir": output_dir,
            "num_generations_per_example": num_generations_per_example,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "prefill_generation": prefill_generation,
            "aspect": aspect,
            "use_flash_attention": use_flash_attention,
            "padding_side": tokenizer.padding_side,
            **generate_kwargs
        }
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="inst_tune")
    parser.add_argument("--output_dir", type=str, default="inst_gens")
    parser.add_argument("--num_generations_per_example", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--prefill_generation", action="store_true")
    parser.add_argument("--aspect", type=str, default="service")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_flash_attention", action="store_true")
    # generation arguments
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    main(**vars(args))