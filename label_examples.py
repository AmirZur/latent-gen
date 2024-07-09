import argparse
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GEN_PREFIX = '<|assistant|> '
GEN_SUFFIX = '<|end|>'

def parse_example(example):
    generation = example['generation']
    generation = generation.split(GEN_PREFIX)[1].split(GEN_SUFFIX)[0]
    return {
        'generation': generation
    }

def main(
    model_name_or_path: str = "train_classifier",
    data_dir: str = "inst_gens/original_examples",
    output_dir: str = "inst_gens/labeled_examples"
):
    dataset = load_from_disk(data_dir)
    dataset = dataset.map(parse_example, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

    inputs = tokenizer(
        dataset['generation'], return_tensors='pt', padding=True, truncation=True
    )
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).tolist()
    dataset = dataset.add_column(predictions, 'predictions')
    dataset.save_to_disk(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="train_classifier")
    parser.add_argument("--data_dir", type=str, default="inst_gens/original_examples")
    parser.add_argument("--output_dir", type=str, default="inst_gens/labeled_examples")
    args = parser.parse_args()
    main(**vars(args))