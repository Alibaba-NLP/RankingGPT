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
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
import copy
import logging
from dataclasses import dataclass, field
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union,Sequence

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
import datasets
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, BloomForCausalLM, LlamaTokenizer
import random
import os


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    ref_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    temperature: Optional[float] = field(default=1.0)

    top: int = field(default=24)

    w_frozen: Optional[bool] = field(default=True)

@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_group_size: int = field(default=-1)
    len_query: int = field(default=64)
    len_doc: int = field(default=438)

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labels_gen: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        
        lm_logits = self.lm_head(hidden_states)
        
        with torch.no_grad():
            init_lm_logits = self.init_model(input_ids=input_ids,attention_mask=attention_mask)[0]

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            device = lm_logits.device
            labels = labels.to(device)
            labels_gen = labels_gen.to(device)
            indexs=(labels!=-100).long()
            label_no_ingore = torch.where(labels==-100,torch.ones(labels.shape).long().to(device),labels)

            preds = torch.nn.functional.log_softmax(lm_logits,dim=-1) #BLV
            logprobs = torch.gather(preds, -1, label_no_ingore.unsqueeze(dim=-1)).squeeze(dim=-1) # B L
            scores = (logprobs*indexs).sum(dim=-1)/indexs.sum(dim=-1) #B -> bsz*group


            scores = torch.exp(scores).view(-1,self.train_group_size)/self.temperature # bsz,group

            
            target_label=torch.zeros(scores.shape[0], dtype=torch.long).to(device)
            loss1 = self.cross_entropy(scores, target_label)
            
            # generation loss
            _,seq_length,vocab_size = lm_logits.shape
            pos_labels = labels_gen.view(-1,self.train_group_size,seq_length)[:,0] #BL
            pos_lm_logits = lm_logits.view(-1,self.train_group_size, seq_length, vocab_size)[:,0]

            loss2 = self.cross_entropy(
                pos_lm_logits.reshape(-1, vocab_size), pos_labels.reshape(-1)
            )
            
            # kl
            loss3 = self.kl_loss(input=preds.reshape([-1,vocab_size]), target=init_lm_logits.softmax(dim=-1).reshape([-1,vocab_size]))
            
            loss = loss1 + loss2 + loss3
            
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

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

class SupervisedDataset(Dataset):
    def __init__(self, data, train_group_size, tokenizer, len_query, len_doc):
        self.data = data
        self.train_group_size=train_group_size
        self.tokenizer = tokenizer
        self.len_query=len_query
        self.len_doc=len_doc

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        ex = self.data[idx]
        all_qd = []

        if len(ex['negative_passages'])<self.train_group_size-1:
            all_qd = random.choices(ex['negative_passages'], k=self.train_group_size-1)
        else:
            all_qd = random.sample(ex['negative_passages'], self.train_group_size-1)
        
        all_qd = [random.choice(ex['positive_passages'])] + all_qd
        
        def truncation(text,length):
            text=self.tokenizer.decode(self.tokenizer.encode(text,max_length=length, add_special_tokens=False))
            return text
        

        query = truncation(ex['query'], self.len_query).replace(self.tokenizer.pad_token,'PAD')
        all_doc = [truncation(qd['text'], self.len_doc).replace(self.tokenizer.pad_token,'PAD') for qd in all_qd]
        
        input_prompt = 'Document: {passage} Query:'
        
        sources = [input_prompt.format(passage = doc) for doc in all_doc]        
        targets=[query for _ in sources]

        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        labels_gen = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        assert len(input_ids)==len(labels)        

        return dict(input_ids=input_ids, labels=labels, labels_gen=labels_gen)

    def _tokenize_fn(self, strings: Sequence[str]) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            self.tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, labels_gen = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "labels_gen"))
        input_ids=[item for sublist in input_ids for item in sublist]
        labels=[item for sublist in labels for item in sublist] 
        labels_gen=[item for sublist in labels_gen for item in sublist] 

        for index in range(len(input_ids)):
            input_ids[index]=input_ids[index][:-1]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        labels_gen = torch.nn.utils.rnn.pad_sequence(labels_gen, batch_first=True, padding_value=IGNORE_INDEX)

        labels = labels[..., 1:].contiguous() #BL
        labels_gen = labels_gen[..., 1:].contiguous() #BL
        return dict(
            input_ids=input_ids,
            labels=labels,
            labels_gen=labels_gen,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.predict_with_generate=True
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.bsz = training_args.per_device_train_batch_size
    model.train_group_size = data_args.train_group_size
    model.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
    model.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    model.temperature = model_args.temperature

    model.init_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.ref_path,
        cache_dir=training_args.cache_dir
    ).eval()
    
    if model_args.w_frozen:
        # peft
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        for name, param in model.transformer.h[-1*model_args.top:].named_parameters():
            param.requires_grad = True
    
    from functools import partial
    model.forward = partial(forward, model)

    if 'llama' in model_args.tokenizer_name_or_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data = datasets.load_dataset('json',data_files=data_args.train_data_path)['train']
    

    train_dataset = SupervisedDataset(data=data, train_group_size=data_args.train_group_size,tokenizer=tokenizer,len_query=data_args.len_query,len_doc=data_args.len_doc)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    trainer = Seq2SeqTrainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset, data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    

if __name__ == "__main__":
    train()
