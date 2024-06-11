import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
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
    output_dir: str = "inst_tune",
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
    logging_dir: str = "logs",
    logging_steps: int = 100,
    max_seq_length: int = 128,
):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    train_dataset = load_dataset("CEBaB/CEBaB", split=dataset_split)
    train_dataset = train_dataset.map(create_example)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        max_seq_length=max_seq_length,
        report_to=[]
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--dataset_split", type=str, default="train_observational")
    parser.add_argument("--output_dir", type=str, default="inst_tune")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=128)
    args = parser.parse_args()
    main(**vars(args))