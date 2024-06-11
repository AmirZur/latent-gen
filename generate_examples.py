import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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
    dataset_split: str = "train_observational",
    output_dir: str = "inst_gens",
    num_examples: int = 100,
    max_new_tokens: int = 128
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
    train_dataset = train_dataset.map(create_example)

    examples = train_dataset.select(range(num_examples))
    outputs = []
    for example in tqdm(examples, desc="Generating examples..."):
        inputs = tokenizer.apply_chat_template(
            example['messages'][:1],
            return_tensors="pt",
            add_generation_prompt=True
        ).to(device)
        with torch.no_grad():
            generation = model.generate(
                input_ids=inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id
            )
        outputs.append(tokenizer.decode(generation[0], skip_special_tokens=True))
    
    orig_perplexities, gen_perplexities = [], []
    for example, output in tqdm(zip(examples, outputs), desc="Calculating perplexities..."):
        messages = example['messages']
        orig_inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            orig_perplexity = model(input_ids=orig_inputs).loss
            orig_perplexities.append(torch.exp(orig_perplexity).item())

        gen_inputs = tokenizer(output, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_perplexity = model(**gen_inputs).loss
            gen_perplexities.append(torch.exp(gen_perplexity).item())
    
    results = {
        'results': [
            {
                'prompt': examples[i]['messages'][0]['content'],
                'original_completion': examples[i]['messages'][1]['content'],
                'model_completion': outputs[i],
                'original_perplexity': orig_perplexities[i],
                'model_perplexity': gen_perplexities[i]
            }
            for i in range(num_examples)
        ]
    }
    with open(f'{output_dir}/outputs.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset_split", type=str, default="train_observational")
    parser.add_argument("--output_dir", type=str, default="inst_gens")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    main(**vars(args))