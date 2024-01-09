#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import datasets


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="bigscience/bloom-560m")
    tokenizer_name_or_path: Optional[str] = field(default="bigscience/bloom-560m")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dev_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    mask_input: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    train_with_peft: bool = field(default=False, metadata={"help": "Is training with peft"})
    


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, save_peft=False):
    """Collects the state dict and dump to disk."""
    if save_peft:
        trainer.model = trainer.model.cpu()
        trainer.model.save_pretrained(output_dir)
    else:
        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)
    

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    mask_input: bool
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    if mask_input:
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, mask_input: bool):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.list_data_dict = datasets.load_dataset('json',data_files=data_path)['train']
        self.tokenizer = tokenizer
        self.mask_input = mask_input
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        example=self.list_data_dict[i]
        source = example['query']
        if 'qwen' in self.tokenizer.name_or_path.lower():
            target = f"{example['response']}"
        else:
            target = f"{example['response']}{self.tokenizer.eos_token}"
        
        data_dict = preprocess([source], [target], self.tokenizer, self.mask_input)
        
        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]
        return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.train_data_path, mask_input=data_args.mask_input)
    dev_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.dev_data_path, mask_input=data_args.mask_input)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=dev_dataset, data_collator=data_collator)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if 'qwen' not in model_args.model_name_or_path.lower():
        training_args.predict_with_generate=True
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    
    print("load model")
    if training_args.train_with_peft:
        print("lora")
        from peft import get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
    
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )
    except:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True
        )
    if 'qwen' not in model_args.model_name_or_path.lower():
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )
    else:
        tokenizer.pad_token_id = tokenizer.eod_id
    
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Seq2SeqTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, save_peft=training_args.train_with_peft)
    

if __name__ == "__main__":
    train()
