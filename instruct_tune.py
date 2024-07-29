import argparse
import datetime
from functools import partial
import os
import random
from typing import Optional
import numpy as np
import pandas as pd
import torch
from tqdm import trange
import wandb
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROMPT_TEMPLATE = """Write a short restaurant review for the following restaurant:
Name: {restaurant_name}
Cuisine: {cuisine}
Price tier: {price_tier}
Dining style: {dining_style}
Region: {region}"""

PROMPT_TEMPLATE_WITH_EXAMPLE = """Your task is to write short restaurant reviews. Follow the same sentiment to food, service, noise, and ambiance as in the example below.

Example restaurant:
Name: {example_restaurant_name}
Cuisine: {example_cuisine}
Price tier: {example_price_tier}
Dining style: {example_dining_style}
Region: {example_region}

Example review:
{example_review}

Restaurant to review:
Name: {restaurant_name}
Cuisine: {cuisine}
Price tier: {price_tier}
Dining style: {dining_style}
Region: {region}

Write a review:"""

LABELS = ['Negative', 'Positive', 'unknown']

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

def create_input(datapoint):
    # convert string to dict (VERY UNSAFE, DO NOT USE ON UNTRUSTED DATA)
    metadata = eval(datapoint['opentable_metadata'])
    prompt = PROMPT_TEMPLATE.format(**metadata)
    completion = datapoint['description']
    messages = [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': completion}
    ]
    return {'messages': messages}

def create_input_with_example(df, datapoint):
    metadata = eval(datapoint['opentable_metadata'])
    example = df[
        (df['food_aspect_majority'] == datapoint['food_aspect_majority']) & \
        (df['service_aspect_majority'] == datapoint['service_aspect_majority']) & \
        (df['noise_aspect_majority'] == datapoint['noise_aspect_majority']) & \
        (df['ambiance_aspect_majority'] == datapoint['ambiance_aspect_majority']) & \
        (df['id'] != datapoint['id'])
    ]
    # on the off chance where we don't have a matching example, skip it
    if example.shape[0] == 0:
        return {'messages': None}
    example = example.sample(random_state=SEED).iloc[0]
    example_metadata = eval(example['opentable_metadata'])
    prompt = PROMPT_TEMPLATE_WITH_EXAMPLE.format(
        restaurant_name=metadata['restaurant_name'],
        cuisine=metadata['cuisine'],
        price_tier=metadata['price_tier'],
        dining_style=metadata['dining_style'],
        region=metadata['region'],
        example_restaurant_name=example_metadata['restaurant_name'],
        example_cuisine=example_metadata['cuisine'],
        example_price_tier=example_metadata['price_tier'],
        example_dining_style=example_metadata['dining_style'],
        example_region=example_metadata['region'],
        example_review=example['description']
    )
    completion = datapoint['description']
    messages = [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': completion}
    ]
    return {'messages': messages}

def create_input_validation(
    create_input_fn, 
    tokenizer, 
    example,
    prefill_generation: bool = False
):
    # parse out only first message for validation (no completion)
    messages = create_input_fn(example)['messages'][:1]
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True,
    )
    if prefill_generation:
        inputs += example['prefix']
    return inputs

def create_dataset(
    prompt_with_example: bool = False,
    dataset_split: Optional[str] = None,
    remove_columns: bool = False
):
    # load the dataset (if split is optional, use prompt_with_example to determine which split to use)
    if dataset_split is None:
        dataset_split = "train_inclusive" if prompt_with_example else "train_observational"
    train_dataset = load_dataset("CEBaB/CEBaB", split=dataset_split)
    train_dataset = train_dataset.filter(
        lambda d: 
            d['food_aspect_majority'] not in ['no majority', ''] and \
            d['service_aspect_majority'] not in ['no majority', ''] and \
            d['noise_aspect_majority'] not in ['no majority', ''] and \
            d['ambiance_aspect_majority'] not in ['no majority', '']
    )
    train_df = train_dataset.to_pandas()
    create_input_fn = partial(create_input_with_example, train_df) if prompt_with_example else create_input
    # preprocess the data
    remove_columns = train_dataset.features if remove_columns else None
    train_dataset = train_dataset.map(create_input_fn, remove_columns=remove_columns)
    # filter out examples with no messages
    train_dataset = train_dataset.filter(lambda x: x['messages'] is not None)
    return train_dataset, create_input_fn

def main(
    # script arguments
    do_train: bool = True,
    do_eval: bool = False,
    do_classify: bool = False,
    # model arguments
    model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct",
    use_flash_attention: bool = False,
    # data arguments
    prompt_with_example: bool = False,
    # training arguments
    output_dir: str = "inst_tune",
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
    max_seq_length: int = 128,
    # logging arguments
    logging_dir: str = "logs",
    logging_steps: int = 100,
    save_model: bool = False,
    use_wandb: bool = False,
    # eval arguments
    eval_split: str = "validation",
    prefill_generation: bool = False,
    aspect: str = "service",
    max_new_tokens: int = 256,
    num_return_sequences: int = 10,
    assistant_prefix: str = "<|assistant|>",
    assistant_suffix: str = "<|end|>",
    # classification arguments
    classifier_name_or_path: str = "train_classifier",
    args: argparse.Namespace = None,
    **generate_kwargs
):
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

    # Train model
    if do_train:
        train_dataset, _ = create_dataset(prompt_with_example, remove_columns=True)

        report_to = []
        if use_wandb:
            wandb.init(project="cebab_instruct_tune", config=vars(args))
            report_to = "wandb"

        # Set up trainer
        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_dir=logging_dir,
            logging_steps=logging_steps,
            max_seq_length=max_seq_length,
            report_to=report_to,
            save_strategy="no"
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=sft_config,
            train_dataset=train_dataset
        )
        trainer.train()

        # Save the model
        if save_model:
            os.makedirs(f"{output_dir}/train", exist_ok=True)
            trainer.save_model(f"{output_dir}/train")

    # Evaluate the model
    if do_eval:
        # load eval dataset
        eval_dataset, create_input_fn = create_dataset(prompt_with_example, eval_split, remove_columns=False)
        if prefill_generation:
            original_df = pd.DataFrame(eval_dataset)
            df = original_df[original_df['edit_type'] == aspect].copy()
            df['original_description'] = df.apply(lambda x: get_original(original_df, x), axis=1)
            df['prefix'] = df.apply(get_prefix, axis=1)
        else:
            df = pd.DataFrame(eval_dataset)
            df['restaurant_id'] = df['opentable_metadata'].map(lambda x: eval(x)['restaurant_id'])
            df = df.drop_duplicates(subset='restaurant_id').reset_index(drop=True)

        data = []
        generations = []
        for b in trange(0, len(df), per_device_eval_batch_size, desc="Generating..."):
            batch = df.iloc[b:b+per_device_eval_batch_size].to_dict(orient="records")
            examples = [
                create_input_validation(
                    create_input_fn,
                    tokenizer,
                    example,
                    prefill_generation=prefill_generation
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
                generations += tokenizer.batch_decode(outputs)
                data += [b for b in batch for _ in range(num_return_sequences)]
        df = pd.DataFrame(data)
        df['generation'] = generations

        df['text'] = df['generation'].map(
            lambda x: x.split(assistant_prefix)[1].split(assistant_suffix)[0].strip()
        )

        # save the generations
        os.makedirs(f"{output_dir}/eval", exist_ok=True)
        df.to_csv(f"{output_dir}/eval/generations.csv", index=False)

    # done with train/evaluation, free up memory
    del model, tokenizer

    if do_classify:
        if not do_eval:
            df = pd.read_csv(f"{output_dir}/eval/generations.csv")
        
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

        df[f'{aspect}_labels'] = [LABELS[p] for p in predictions]
        # overwrite the generations with the classifications
        df.to_csv(f"{output_dir}/eval/generations.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_classify", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--prompt_with_example", action="store_true", 
                        help="Prompt with an example review (uses inclusive training data, otherwise uses observational training data)")
    parser.add_argument("--output_dir", type=str, default="inst_tune")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--prefill_generation", action="store_true")
    parser.add_argument("--aspect", type=str, default="service")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--assistant_prefix", type=str, default="<|assistant|>")
    parser.add_argument("--assistant_suffix", type=str, default="<|end|>")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--classifier_name_or_path", type=str, default="train_classifier")
    args = parser.parse_args()
    main(**vars(args), args=args)