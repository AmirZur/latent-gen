import random
import numpy as np
import torch

INSTRUCTION_TEMPLATE = """Your task is to write short restaurant reviews. You must be consistent with the sentiment towards {aspects}."""

EXAMPLE_TEMPLATE = """Write a short restaurant review for the following restaurant:
Name: {restaurant_name}
Cuisine: {cuisine}
Price tier: {price_tier}
Dining style: {dining_style}
Region: {region}

Review:"""

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

PROMPT_TEMPLATE_ONE_SHOT = """Your task is to write short restaurant reviews. Follow the same sentiment to {aspects} as in the example below.

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

ASPECT_KEYWORDS = {
    'service': ['waiter', 'waitress', 'service', 'staff', 'server', 'host', 'hostess', 'reservation', 'wait']
}

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    example = example.sample().iloc[0]
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

def create_input_one_shot(df, datapoint):
    metadata = eval(datapoint['opentable_metadata'])
    example_df = df[df['id'] != datapoint['id']]
    # filter out examples that don't match the sentiment (if it's expressed)
    relevant_keys = []
    for key in ['food_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority', 'ambiance_aspect_majority']:
        if datapoint[key] in ['Positive', 'Negative']:
            example_df = example_df[example_df[key] == datapoint[key]]
            relevant_keys.append(key)
    # on the off chance where we don't have a matching example, skip it
    if example_df.shape[0] == 0:
        return {'messages': None}
    # sample a random example
    example = example_df.sample().iloc[0]
    example_metadata = eval(example['opentable_metadata'])

    # parse out relevant aspects
    aspects = ''
    for i, key in enumerate(relevant_keys):
        aspect = key.split('_')[0]
        if i == 0:
            aspects = aspect
        elif i == len(relevant_keys) - 1:
            aspects += f', and {aspect}'
        else:
            aspects += f', {aspect}'
        
    # instruction_prompt = INSTRUCTION_TEMPLATE.format(aspects=aspects)
    # example_prompt = EXAMPLE_TEMPLATE.format(**example_metadata)
    # prompt = EXAMPLE_TEMPLATE.format(**metadata)
    # example_completion = example['description']
    # completion = datapoint['description']
    # messages = [
    #     {'role': 'user', 'content': instruction_prompt},
    #     {'role': 'user', 'content': example_prompt},
    #     {'role': 'assistant', 'content': example_completion},
    #     {'role': 'user', 'content': prompt},
    #     {'role': 'assistant', 'content': completion}
    # ]

    prompt = PROMPT_TEMPLATE_ONE_SHOT.format(
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
        example_review=example['description'],
        aspects=aspects
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