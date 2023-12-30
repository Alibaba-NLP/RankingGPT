import torch
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM,T5Tokenizer, T5ForConditionalGeneration
import torch
import argparse
import json
from tqdm import tqdm
import os
import copy

def get_model_tokenizer(model_path):
    if 'llama' in model_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)
    model.eval()
    return model,tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',
                    default="",
                    required=False)
parser.add_argument('--res_path',
                    default="",
                    required=False)
parser.add_argument('--rank_path',
                    default="",
                    required=False)
parser.add_argument('--data_name',
                    default='msmarco')

args = parser.parse_args()


model_path=args.model_path
data_name=args.data_name

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

bsz=8

prompt='Document: {doc} Query:'

model,tokenizer=get_model_tokenizer(model_path)
if 'qwen' in model_path.lower():
    tokenizer.pad_token_id = tokenizer.eod_id

def get_num_token(text):
    return len(tokenizer.encode(text))

prompt_len=get_num_token(prompt)
print(f"prompt_len: {prompt_len}")


def truncation(text,length):
    text=tokenizer.decode(tokenizer.encode(text,max_length=length, add_special_tokens=False))
    return text

def _tokenize_fn(strings):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
        )['input_ids']
        for text in strings
    ]
    input_ids = labels = [tokenized[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

all_examples=[]
all_sources=[]
all_qpids=[]
all_queries=[]

for line in tqdm(open(args.rank_path),desc='load data'):
    ex = json.loads(line)
    all_qpids.append((ex['qid'],ex['pid']))
    if data_name!='arguana':
        query = ex["query"].replace(DEFAULT_PAD_TOKEN,'PAD')
        query_len = get_num_token(query)
        passage_max_len = 512-prompt_len-query_len-10
        source = prompt.format(doc = truncation(ex['passage'], passage_max_len)).replace(DEFAULT_PAD_TOKEN,'PAD')
    else:
        source = prompt.format(doc = truncation(ex['passage'], 256)).replace(DEFAULT_PAD_TOKEN,'PAD')
        query = truncation(ex['query'], 256)
    all_examples.append(source+query)
    all_sources.append(source)
    all_queries.append(query)


with open(args.res_path,"w") as fw:
    for index in tqdm(range(0,len(all_examples),bsz)):
        examples=all_examples[index:index+bsz]
        sources=all_sources[index:index+bsz]
        qpids=all_qpids[index:index+bsz]
        queries=all_queries[index:index+bsz]
        qid, pid = qpids[0]
        
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        

        for index in range(len(input_ids)):
            input_ids[index]=input_ids[index][:-1]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
        
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).cuda()
        labels = labels[..., 1:].contiguous() #BL

        with torch.no_grad():
            lm_logits = model(input_ids=input_ids,attention_mask=input_ids.ne(tokenizer.pad_token_id))[0]
            preds = torch.nn.functional.log_softmax(lm_logits,dim=-1)
            label_no_ingore = torch.where(labels==-100,torch.ones(labels.shape).long().cuda(),labels)
            logprobs = torch.gather(preds, -1, label_no_ingore.unsqueeze(dim=-1)).squeeze(dim=-1) # B L
            indexs=(labels!=-100).long()
            scores=(logprobs*indexs).sum(dim=-1)/indexs.sum(dim=-1)
            scores=scores.cpu().tolist()

        for index,score in enumerate(scores):
            qid, pid=qpids[index]
            print(" ".join([qid,"Q0",pid,"-1",str(score),model_path]),file=fw)
        del lm_logits



results={}
for line in open(args.res_path):
    line = line.strip().split()
    qid = line[0]
    pid = line[2]

    score = float(line[4])
    if qid not in results:
        results[qid] = []
    results[qid].append((pid,score))

with open(args.res_path[:-4]+"_post.res","w") as fw:
    for qid in results:
        res = results[qid]
        sorted_res = sorted(res,key = lambda x:-x[1])
        for i,item in enumerate(sorted_res):
            print(" ".join([qid, "Q0", item[0], str(i), str(item[1]), 'llm']),file=fw)
