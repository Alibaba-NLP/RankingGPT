ep=1
lr=3e-5
bsz=1
group=48
gas=4
card=8
workers=64
save_steps=1000
data_name=msmarco.json
temperature=0.001
ref_path=$1
ref_name=$2
top=$3
layername=$4

out_dir="outputs_sft_${ref_name}"

echo ${out_dir}
mkdir -p ${out_dir}

torchrun --nproc_per_node=${card} --master_port=29405 sft.py \
    --model_name_or_path ${ref_path} \
    --tokenizer_name_or_path ${ref_path} \
    --train_data_path ./datasets/${data_name} \
    --model_max_length 512 \
    --output_dir ${out_dir} \
    --num_train_epochs ${ep} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${gas} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --train_group_size ${group} \
    --dataloader_num_workers ${workers} \
    --temperature ${temperature} \
    --len_query 32 \
    --len_doc 128 \
    --ref_path ${ref_path} \
    --only_query ${only_query} \
    --top ${top} \
    --bf16 True \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap ${layername}

echo ${out_dir}
