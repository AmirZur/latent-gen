import argparse
from typing import Optional
from tqdm import trange
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
)

LABELS = ['Negative', 'Positive', 'unknown']

def main(
    model_name_or_path: str = "train_classifier",
    examples_path: str = "inst_gens/generations.csv",
    output_path: Optional[str] = None,
    batch_size: int = 8,
    aspect: str = "service",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        device_map=device
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    df = pd.read_csv(examples_path)

    predictions = []
    for b in trange(0, df.shape[0], batch_size):
        batch = df.iloc[b:b+batch_size]
        examples = batch['generation'].tolist()
        inputs = tokenizer(examples, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**inputs)
        predictions += outputs.logits.argmax(dim=1).tolist()

    df[f'{aspect}_labels'] = [LABELS[p] for p in predictions]

    if output_path is None:
        output_path = examples_path
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--examples_path", type=str, default="inst_gens/generations.csv")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--aspect", type=str, default="service")
    args = parser.parse_args()
    main(**vars(args))