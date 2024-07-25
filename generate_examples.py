import argparse
import json
import random
import os
import torch
from tqdm import trange
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

SEED = 42
random.seed(SEED)

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
    messages = [
        {'role': 'user', 'content': prompt},
    ]
    return {'messages': messages}

def main(
    model_name_or_path: str = "inst_tune",
    output_dir: str = "inst_gens",
    num_generations_per_example: int = 10,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    dataset_split: str = "train_observational",
    **generate_kwargs
):
    # Load the model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    train_dataset = load_dataset("CEBaB/CEBaB", split=dataset_split)
    df = pd.DataFrame(train_dataset)
    df['restaurant_id'] = df['opentable_metadata'].map(lambda x: eval(x)['restaurant_id'])
    df = df.drop_duplicates(subset='restaurant_id').reset_index(drop=True)

    data = []
    generations = []
    for b in trange(0, len(df), batch_size, desc="Generating..."):
        batch = df.iloc[b:b+batch_size].to_dict(orient="records")
        examples = [create_example(example) for example in batch]
        inputs = tokenizer.apply_chat_template(
            examples,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_generations_per_example,
                **generate_kwargs
            )
            generations += tokenizer.batch_decode(outputs, skip_special_tokens=True)
            data += [b for b in batch for _ in range(num_generations_per_example)]
    
    df = pd.DataFrame(data)
    df['generation'] = generations

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "generations.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="inst_tune")
    parser.add_argument("--output_dir", type=str, default="inst_gens")
    parser.add_argument("--num_generations_per_example", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_split", type=str, default="train_observational")
    # generation arguments
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    main(**vars(args))