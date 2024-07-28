import argparse
from functools import partial
import random
import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    ].sample(random_state=SEED).iloc[0]
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


def main(
    model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct",
    use_flash_attention: bool = False,
    prompt_with_example: bool = False,
    output_dir: str = "inst_tune",
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
    logging_dir: str = "logs",
    logging_steps: int = 100,
    max_seq_length: int = 128,
    save_model: bool = False,
    use_wandb: bool = False,
    args: argparse.Namespace = None
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

    # load the dataset
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
    train_dataset = train_dataset.map(create_input_fn, remove_columns=train_dataset.features)

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

    # Train the model
    trainer.train()

    # Save the model
    if save_model:
        trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    main(**vars(args), args=args)