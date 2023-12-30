# Description
This is the official code for paper [RankingGPT: Empowering Large Language Models in Text Ranking with Progressive Enhancement](https://arxiv.org/abs/2311.16720).

# Requirements
```
transformers==4.28.1
datasets
pyserini
torch==1.13.1
```

# Data

- ./datasets/text_pairs.json: Weakly supervised text pairs

- ./datasets/msmarco.json: Supervised fine-tuning data

- ./rankdata/trec19: Top-1000 query-document pairs recalled by BM25


# Two-stage Training

## Pretrain
```
bash pretrain.sh bigscience/bloom-560m bloom-560m BloomBlock
```

## SFT
```
bash sft.sh ./outputs_pretrain_bloom-560m bloom-560m 16 BloomBlock
```

# Evaluation
```
bash eval.sh ./outputs_sft_bloom-560m trec19 bloom-560m
```

# Results
*Ranking results (NDCG@10) of the top-1000 candidate documents recalled by BM25.*
|         | DL19 | DL20 | BEIR | url |
|---------|------|------|------|-----------------|
| MonoBERT-340M | 72.3 | 70.3 | 50.5 |     [huggingface](https://huggingface.co/veneres/monobert-msmarco)          |
| MonoT5-220M  | 71.5 | 69.7 | 49.3 |     [huggingface](https://huggingface.co/castorini/monot5-base-msmarco)          |
| MonoT5-770M  | 73.2 | 71.2 | 53.1 |    [huggingface](https://huggingface.co/castorini/monot5-large-msmarco)          |
| MonoT5-3B  | 72.8 | 74.5 | 54.6 |     [huggingface](https://huggingface.co/castorini/monot5-3b-msmarco)          |
| RankT5-770M  | -    | -    | 53.7 |     [huggingface](https://huggingface.co/bergum/rank-T5-flan)           |
| RankLLaMA| 74.6 | 76.6 | 52.5 |  [huggingface](https://huggingface.co/castorini/rankllama-v1-7b-lora-passage) |
| RankingGPT-bloom-560m| 75.3 | 73.2 | 53.7 | [huggingface](https://huggingface.co/zyznull/RankingGPT-bloom-560m) [modelscope](https://modelscope.cn/models/damo/RankingGPT-bloom-560m)       |
| RankingGPT-bloom-1b1| 75.6 | 73.2 | 54.5 | [huggingface](https://huggingface.co/zyznull/RankingGPT-bloom-1b1)  [modelscope](https://modelscope.cn/models/damo/RankingGPT-bloom-1b1)        |
| RankingGPT-bloom-3b| 76.8 | 73.6 | 56.2 | [huggingface](https://huggingface.co/zyznull/RankingGPT-bloom-3b)  [modelscope](https://modelscope.cn/models/damo/RankingGPT-bloom-3b)        |
| RankingGPT-bloom-7b| 77.3 | 74.6 | 56.6 | [huggingface](https://huggingface.co/zyznull/RankingGPT-bloom-7b)  [modelscope](https://modelscope.cn/models/damo/RankingGPT-bloom-7b)        |
| RankingGPT-llama2-7b| 76.2 | 76.3 | 57.8 | [huggingface](https://huggingface.co/zyznull/RankingGPT-llama2-7b)  [modelscope](https://modelscope.cn/models/damo/RankingGPT-llama2-7b)        |
| RankingGPT-baichuan2-7b| 75.9 | 74.3 | 57.5 |  [huggingface](https://huggingface.co/zyznull/RankingGPT-baichuan2-7b) [modelscope](https://modelscope.cn/models/damo/RankingGPT-baichuan2-7b)        |
| RankingGPT-qwen-7b| 75.8 | 74.3 | 58.3 | [huggingface](https://huggingface.co/zyznull/RankingGPT-qwen-7b)  [modelscope](https://modelscope.cn/models/damo/RankingGPT-qwen-7b)        |