from typing import Dict, Sequence
from tqdm import tqdm
import transformers
from pyreft import ReftTrainer
import pyvene as pv
import torch
import copy
import datasets
from datasets import Dataset
from dataclasses import dataclass
from torch.utils.data import DataLoader

IGNORE_INDEX = -100

@dataclass
class DASDataCollator(object):
    """Collate examples for Pyvene intervention training."""
    data_collator: transformers.DataCollator

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate examples for Pyvene intervention training.

        instances : dict with keys
            "input_ids" : torch.Tensor
                input ids of base + cf output
            "labels" : torch.Tensor
                labels of base + cf output (equal to input_ids but with prompt masked out)
            "intervention_locations" : torch.Tensor
                intervention locations for base
            "source_input_ids" : torch.Tensor
                input ids of source example + source output
            "source_intervention_locations" : torch.Tensor
                intervention locations for source example
        """
        base_instances = [{
            "input_ids": instance["input_ids"],
            "intervention_locations": instance["intervention_locations"],
            "labels": instance["labels"]
        } for instance in instances]

        source_instances = [{
            "input_ids": instance["source_input_ids"],
            "intervention_locations": instance["source_intervention_locations"],
            "labels": [-100] * len(instance['source_input_ids']) # dummy label, not used
        } for instance in instances]

        # collate as batch
        all_instances = base_instances + source_instances

        batch_inputs = self.data_collator(all_instances)
        all_input_ids = batch_inputs.pop("input_ids")
        all_attention_mask = batch_inputs.pop("attention_mask")
        all_labels = batch_inputs.pop("labels")
        all_intervention_locations = batch_inputs.pop("intervention_locations")
        
        # base inputs
        batch_inputs["input_ids"] = all_input_ids[:len(base_instances)]
        batch_inputs["attention_mask"] = all_attention_mask[:len(base_instances)]
        batch_inputs["labels"] = all_labels[:len(base_instances)]
        batch_inputs["intervention_locations"] = all_intervention_locations[:len(base_instances)]

        # source inputs
        batch_inputs["source_input_ids"] = all_input_ids[len(base_instances):]
        batch_inputs["source_attention_mask"] = all_attention_mask[len(base_instances):]
        batch_inputs["source_intervention_locations"] = all_intervention_locations[len(base_instances):]

        return batch_inputs

class DASTrainer(ReftTrainer):
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False
    ):
        """
        base (clean)
         - input_ids : torch.Tensor
            input ids of base + cf output
         - attention_mask : torch.Tensor
            attention mask of base + cf output
         - labels : torch.Tensor
            labels of base + cf output (equal to input_ids but with prompt masked out)

        source (patch)
         - source_input_ids : torch.Tensor
            input ids of source example + source output
         - source_attention_mask : torch.Tensor
            attention mask of source example + source output
        """
        if self.tokenizer.padding_side == "left":
            # shift intervention locations by left padding amount
            base_padding = inputs["input_ids"].shape[1] - inputs["attention_mask"].sum(dim=1)
            source_padding = inputs["source_input_ids"].shape[1] - inputs["source_attention_mask"].sum(dim=1)
            base_intervention_locations = inputs["intervention_locations"] + base_padding[:, None, None]
            source_intervention_locations = inputs["source_intervention_locations"] + source_padding[:, None, None]
        else:
            base_intervention_locations = inputs["intervention_locations"]
            source_intervention_locations = inputs["source_intervention_locations"]

        # run intervened forward pass
        _, cf_outputs = intervenable(
            # base
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            },
            # source
            [{
                "input_ids": inputs["source_input_ids"],
                "attention_mask": inputs["source_attention_mask"]
            }],
            unit_locations={"sources->base": (
                # copy from
                source_intervention_locations.permute(1, 0, 2).tolist(),
                # paste to
                base_intervention_locations.permute(1, 0, 2).tolist()
            )},
            labels=inputs["labels"],
            # for now, always use first subspace partition
            subspaces=0 # inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None
        )
        # return
        return (cf_outputs.loss, cf_outputs[:2]) if return_outputs else cf_outputs.loss
    
    def evaluate(self, **generate_kwargs):
        # ensure everything is in eval mode
        self.model.eval()
        device = self.model.get_device()
        original_outputs = []
        das_outputs = []
        bases = []
        sources = []

        dataloader = make_dataloader(
            self.eval_dataset, 
            self.args.eval_batch_size, 
            self.data_collator, 
            shuffle=False
        )
        eval_iterator = tqdm(dataloader, position=0, leave=True)
        device = self.model.get_device()

        for inputs in eval_iterator:
            # shift intervention locations by left padding amount
            base_padding = inputs["input_ids"].shape[1] - inputs["attention_mask"].sum(dim=1)
            source_padding = inputs["source_input_ids"].shape[1] - inputs["source_attention_mask"].sum(dim=1)
            base_intervention_locations = inputs["intervention_locations"] + base_padding[:, None, None]
            source_intervention_locations = inputs["source_intervention_locations"] + source_padding[:, None, None]
            with torch.no_grad():
                original_output, das_output = self.model.generate(
                    # base
                    {
                        "input_ids": inputs["input_ids"].to(device),
                        "attention_mask": inputs["attention_mask"].to(device)
                    },
                    # source
                    sources=[{
                        "input_ids": inputs["source_input_ids"].to(device),
                        "attention_mask": inputs["source_attention_mask"].to(device)
                    }],
                    unit_locations={"sources->base": (
                        # copy from
                        source_intervention_locations.permute(1, 0, 2).tolist(),
                        # paste to
                        base_intervention_locations.permute(1, 0, 2).tolist()
                    )},
                    subspaces=0,
                    output_original_output=True,
                    intervene_on_prompt=True,
                    **generate_kwargs
                )
                original_outputs += self.tokenizer.batch_decode(original_output, skip_special_tokens=True)
                das_outputs += self.tokenizer.batch_decode(das_output, skip_special_tokens=True)
                bases += self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
                sources += self.tokenizer.batch_decode(inputs["source_input_ids"], skip_special_tokens=True)
        return list(zip(original_outputs, das_outputs, bases, sources))

def make_dataloader(
        dataset: Dataset, batch_size: int, collate_fn: transformers.DataCollatorForSeq2Seq, shuffle: bool
    ) -> DataLoader:
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=collate_fn)

class DASTrainerForCausalLM(DASTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)

    def get_eval_dataloader(self) -> DataLoader:
        return make_dataloader(self.eval_dataset, self.args.eval_batch_size, self.data_collator, shuffle=False)

def parse_complex_positions(positions: str):
    # parse position
    first_n, last_n, decode_intv = 0, 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
        decode_intv = int(positions.split("+")[2].strip("d"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
        elif "d" in positions:
            decode_intv = int(positions.strip("d"))
    return first_n, last_n, decode_intv


def generate_numbers_by_interval(interval, max_number):
    """
    Generates numbers based on the given interval and max number.

    Parameters:
    interval (int): The interval between consecutive numbers.
    max_number (int): The maximal number up to which the integers should be generated.

    Returns:
    list: A list of integers that fall within the interval and are below the max number.
    """
    numbers = [num for num in range(interval, max_number, interval)]
    return numbers


def get_complex_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_prompt_position = kwargs["last_prompt_position"]
    last_input_position = kwargs["last_input_position"]
    if "positions" in kwargs:
        _first_n, _last_n, _decode_intv = \
            parse_complex_positions(kwargs["positions"])
    else:
        _first_n, _last_n, _decode_intv = \
            kwargs["first_n"], kwargs["last_n"], kwargs["decode_intv"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"

    first_n = min(last_prompt_position // 2, _first_n)
    last_n = min(last_prompt_position // 2, _last_n)
    decode_intv = _decode_intv

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_prompt_position
    if share_weights or (first_n == 0 or last_n == 0):
        # position_list = [i for i in range(first_n)] + \
        #     [i for i in range(last_prompt_position - last_n, last_prompt_position)] + \
        #     [pad_position for _ in range(pad_amount)]
        if last_n == 0:
            position_list = [first_n - 1]
        else:
            position_list = [last_prompt_position - last_n]
        intervention_locations = [position_list]*num_interventions
    else:
        left_pad_amount = (_first_n - first_n)
        right_pad_amount = (_last_n - last_n)
        left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
        right_intervention_locations = [i for i in range(last_prompt_position - last_n, last_prompt_position)] + \
            [pad_position for _ in range(right_pad_amount)]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [pad_position for _ in range(left_len-right_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(right_len-left_len)]
        intervention_locations = [left_intervention_locations]*(num_interventions//2) + \
            [right_intervention_locations]*(num_interventions//2)

    # add decoding step interventions
    if decode_intv != 0:
        with_decode_intervention_locations = []
        for locations in intervention_locations:
            _decode_locations = generate_numbers_by_interval(decode_intv, last_input_position)
            new_locations = locations + _decode_locations
            with_decode_intervention_locations.append(new_locations)
        return with_decode_intervention_locations
    
    return intervention_locations


def make_complex_position_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    model,
    base_inputs, 
    cf_outputs,
    source_inputs,
    source_outputs, 
    positions="f1+l1+d1", 
    num_interventions=1, 
    nonstop=False, 
    share_weights=False,
    intervention_offset=0,
    evaluation=False
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    first_n, last_n, decode_intv = parse_complex_positions(positions)
    
    all_base_input_ids, all_intervention_locations, all_output_ids = [], [], []
    all_source_input_ids, all_source_intervention_locations = [], []
    for i in range(len(base_inputs)):
        _base = base_inputs[i]
        _output = cf_outputs[i]
    
        base_prompt = _base
        base_input = base_prompt + _output
        if not nonstop:
            base_input += tokenizer.eos_token
    
        # tokenize
        base_prompt_ids = tokenizer(
            base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        # add intervention offset to set up training example as 
        # base (until intervention offset) -> cf output (after intervention offset)
        base_prompt_length = len(base_prompt_ids) + intervention_offset
        base_input_ids = tokenizer(
            base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        output_ids = copy.deepcopy(base_input_ids)
        output_ids[:base_prompt_length] = IGNORE_INDEX

        _source_input = source_inputs[i]
        _source_output = source_outputs[i]
        source_prompt = _source_input
        source_input = source_prompt + _source_output

        source_prompt_ids = tokenizer(
            source_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        # add intervention offset to source as well
        source_prompt_length = len(source_prompt_ids) + intervention_offset
        source_input_ids = tokenizer(
            source_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
        
        # NOTE: if using decoding intervals, use same length to guarantee same # of source & base intervention locations
        last_input_position = min(len(output_ids), len(source_input_ids))

        if evaluation:
            # chop off base input ids to only include the prompt
            # skip inputs where offset is longer than output
            if len(base_input_ids) <= base_prompt_length or \
                len(source_input_ids) <= source_prompt_length:
                continue
            base_input_ids = base_input_ids[:base_prompt_length]

        intervention_locations = get_complex_intervention_locations(
            last_prompt_position=base_prompt_length,
            last_input_position=last_input_position,
            first_n=first_n, 
            last_n=last_n,
            decode_intv=decode_intv,
            pad_mode="last",
            num_interventions=num_interventions,
            share_weights=share_weights,
        )
        
        source_intervention_locations = get_complex_intervention_locations(
            last_prompt_position=source_prompt_length,
            last_input_position=last_input_position,
            first_n=first_n, 
            last_n=last_n,
            decode_intv=decode_intv,
            pad_mode="last",
            num_interventions=num_interventions,
            share_weights=share_weights,
        )
        
        all_base_input_ids.append(base_input_ids)
        all_intervention_locations.append(intervention_locations)
        all_output_ids.append(output_ids)
        all_source_input_ids.append(source_input_ids)
        all_source_intervention_locations.append(source_intervention_locations)
        
    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_base_input_ids,
        "intervention_locations": all_intervention_locations,
        "labels": all_output_ids,
        "source_input_ids": all_source_input_ids,
        "source_intervention_locations": all_source_intervention_locations
    })
        
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        padding="longest"
    )
    data_collator = DASDataCollator(data_collator=data_collator_fn)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)