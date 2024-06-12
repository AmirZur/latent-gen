import argparse
import json
import random
import os
import torch
from tqdm import tqdm
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
    completion = example['description']
    messages = [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': completion}
    ]
    return {'messages': messages}

def main(
    model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct",
    output_dir: str = "inst_gens",
    num_examples: int = 100,
    max_new_tokens: int = 256,
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
    train_dataset = load_dataset("CEBaB/CEBaB", split="train_inclusive")
    train_dataset = train_dataset.map(create_example)
    
    original_examples = train_dataset.filter(
        lambda x: x['is_original']
    ).shuffle(seed=SEED).select(range(num_examples))

    edited_examples = train_dataset.filter(
        lambda x: x['original_id'] in original_examples['original_id']
    )

    def generate_example(example):
        inputs = tokenizer.apply_chat_template(
            example['messages'][:1],
            return_tensors="pt",
            add_generation_prompt=True
        ).to(device)
        with torch.no_grad():
            generation = model.generate(
                input_ids=inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id,
                **generate_kwargs
            )
        return {
            'generation': tokenizer.decode(generation[0])
        }

    original_examples = original_examples.map(
        generate_example, desc="Generating model completions..."
    )

    def compute_example_perplexity(example):
        messages = example['messages']
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            perplexity = model(input_ids=inputs, labels=inputs).loss
            perplexity.append(torch.exp(perplexity).item())
        return {
            'description_perplexity': perplexity
        }

    edited_examples = edited_examples.map(
        compute_example_perplexity, desc="Computing description perplexities..."
    )

    def compute_output_perplexity(example):
        inputs = tokenizer(example['generation'], return_tensors="pt").to(device)
        with torch.no_grad():
            perplexity = model(**inputs, labels=inputs['input_ids']).loss
            perplexity.append(torch.exp(perplexity).item())
        return {
            'generation_perplexity': perplexity
        }

    original_examples = original_examples.map(
        compute_output_perplexity, desc="Computing generation perplexities..."
    )

    os.makedirs(output_dir, exist_ok=True)
    
    original_examples.save_to_disk(os.path.join(output_dir, "original_examples"))
    edited_examples.save_to_disk(os.path.join(output_dir, "edited_examples"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--output_dir", type=str, default="inst_gens")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    # generation arguments
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    main(**vars(args))