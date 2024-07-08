import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

ASPECT = 'service'
ASPECT_KEY = f'{ASPECT}_aspect_majority'

def tokenize(d, tokenizer, max_length=128):
    tokens = tokenizer(
        d['description'], return_tensors='pt', padding='max_length', max_length=max_length, truncation=True
    )
    return {k: v[0] for k, v in tokens.items()}

def get_labels(d):
    if d[ASPECT_KEY] == 'Negative':
        return {'labels': 0}
    elif d[ASPECT_KEY] == 'Positive':
        return {'labels': 1}
    else:
        return {'labels': 2}
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return {
        'accuracy': (predictions == labels).mean()
    }

def main(
    model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct",
    dataset_split: str = "train_observational",
    output_dir: str = "classifier_train",
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
    logging_dir: str = "logs",
    logging_steps: int = 100,
    eval_steps: int = 100,
    max_seq_length: int = 128,
    save_model: bool = False
):
    train_dataset = load_dataset("CEBaB/CEBaB", split=dataset_split)
    eval_dataset = load_dataset("CEBaB/CEBaB", split="validation")

    train_dataset = train_dataset.filter(lambda d: d[ASPECT_KEY] != '')
    eval_dataset = eval_dataset.filter(lambda d: d[ASPECT_KEY] != '')

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = train_dataset.map(lambda d: tokenize(d, tokenizer, max_length=max_seq_length))
    eval_dataset = eval_dataset.map(lambda d: tokenize(d, tokenizer, max_length=max_seq_length))

    train_dataset = train_dataset.map(get_labels)
    eval_dataset = eval_dataset.map(get_labels)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=3,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        report_to='wandb',
        save_strategy="no",
        evaluation_strategy="steps",
        eval_steps=eval_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    if save_model:
        trainer.save_model(output_dir)

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
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()
    main(**vars(args))