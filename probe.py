import argparse
from typing import List
from tqdm import trange
from collections import Counter
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from sklearn.linear_model import Perceptron

LABELS = ['Negative', 'Positive', 'unknown']

def main(
    # model arguments
    model_name_or_path: str = "inst_tune",
    classifier_name_or_path: str = "classifier_train",
    use_flash_attention: bool = False,
    # data arguments
    train_path: str = "inst_gens/generations.csv",
    validation_path: str = "inst_gens/generations.csv",
    num_generations_per_example: int = 10,
    # preprocessing arguments
    remove_correct_prefixes : bool = False,
    count_threshold : int = -1,
    categories : List[str] = ['Negative', 'Positive', 'unknown'],
    prefix_length_threshold: int = -1,
    # probe arguments
    num_tokens_from_end: int = 5,
    output_path: str = "probe/scores.csv",
):
    #####################
    # Load data         #
    #####################
    print('Loading and preprocessing data...')
    # Load the dataset
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    def preprocess_data(df):
        data = []
        assert 'prefix' in df.columns, "Only running probe on full prefix for now."
        df.loc[df['prefix'].isna(), 'prefix'] = ''
        for des in df['description'].unique():
            des_df = df[df['description'] == des]
            if des_df.shape[0] != num_generations_per_example:
                continue
            assert (des_df.iloc[0]['prefix'] == des_df['prefix']).all(), des_df['prefix']
            most_common = Counter(des_df['service_labels']).most_common(1)[0]
            prefix_label = des_df.iloc[0]['prefix_labels'] if 'prefix_labels' in des_df.columns else None
            data.append({
                'description': des,
                'most_common': most_common[0],
                'count': most_common[1],
                'prefix': des_df.iloc[0]['prefix'],
                'generations': des_df['text'].tolist(),
                'labels': des_df['service_labels'].tolist(),
                'prefix_label': prefix_label
            })

        preprocessed_df = pd.DataFrame(data)
        # remove prefixes that are correct
        if remove_correct_prefixes:
            preprocessed_df = preprocessed_df[
                preprocessed_df['prefix_label'] != preprocessed_df['most_common']
            ]
        # select prefixes that have a count above a threshold
        preprocessed_df = preprocessed_df[preprocessed_df['count'] > count_threshold]
        # select prefixes that have a length above a threshold
        preprocessed_df = preprocessed_df[preprocessed_df['prefix'].apply(len) > prefix_length_threshold]
        # select prefixes that have a label in the categories
        preprocessed_df = preprocessed_df[preprocessed_df['most_common'].isin(categories)]

        return preprocessed_df.reset_index(drop=True)
    
    train_dataset = preprocess_data(train_df)
    validation_dataset = preprocess_data(validation_df)
    print(f"Loaded data. Train size: {len(train_dataset)}, Validation size: {len(validation_dataset)}")

    #####################
    # Get activations   #
    #####################
    print('Getting activations...')
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
    
    def get_activations(dataset, batch_size=8):
        all_activations = None
        for i in trange(0, len(dataset), batch_size):
            examples = dataset.iloc[i:i+batch_size]
            inputs = tokenizer(
                examples['prefix'].tolist(), return_tensors='pt', padding=True, truncation=True
            ).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            # get activation over last token
            activations = torch.stack(outputs.hidden_states).cpu().transpose(0, 1)[..., -num_tokens_from_end:, :]
            if all_activations is None:
                all_activations = activations
            else:
                all_activations = torch.cat([
                    all_activations,
                    activations
                ], dim=0)
        return all_activations.to(torch.float).numpy()

    train_activations = get_activations(train_dataset)
    validation_activations = get_activations(validation_dataset)
    train_labels = train_dataset['most_common'].to_numpy()
    validation_labels = validation_dataset['most_common'].to_numpy()

    #####################
    # Train probes      #
    #####################
    del model, tokenizer

    # train a separate probe for each layer & token
    score_data = []
    for layer in range(train_activations.shape[1]):
        for token in range(-1, -num_tokens_from_end-1, -1):
            train_activations_layer_token = train_activations[:, layer, token, :]
            validation_activations_layer_token = validation_activations[:, layer, token, :]

            probe = Perceptron()
            probe.fit(train_activations_layer_token, train_labels)
            train_score = probe.score(train_activations_layer_token, train_labels)
            validation_score = probe.score(validation_activations_layer_token, validation_labels)
            score_data.append({
                'layer': layer,
                'token': token,
                'train_score': train_score,
                'validation_score': validation_score
            })
    
    score_df = pd.DataFrame(score_data)
    score_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="inst_tune")
    parser.add_argument("--classifier_name_or_path", type=str, default="classifier_train")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--train_path", type=str, default="inst_gens/generations.csv")
    parser.add_argument("--validation_path", type=str, default="inst_gens/generations.csv")
    parser.add_argument("--num_generations_per_example", type=int, default=10)
    parser.add_argument("--remove_correct_prefixes", action="store_true")
    parser.add_argument("--count_threshold", type=int, default=5)
    parser.add_argument("--categories", nargs='+', default=['Negative', 'Positive', 'unknown'])
    parser.add_argument("--prefix_length_threshold", type=int, default=0)
    parser.add_argument("--num_tokens_from_end", type=int, default=5)
    parser.add_argument("--output_path", type=str, default="probe/scores.csv")
    args = parser.parse_args()
    main(**vars(args))