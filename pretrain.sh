model_name_or_path=$1
modelname=$2
layername=$3

data_name="text_pairs.json"
mask_input=true
ep=1
save_steps=1000
lr=5e-5
bsz=16
gas=4
card=4
worker=64

out_dir=outputs_pretrain_${modelname}
mkdir -p ${out_dir}
echo ${out_dir}

torchrun --nproc_per_node=${card} --master_port=28039 pretrain.py \
    --model_name_or_path ${model_name_or_path} \
    --tokenizer_name_or_path ${model_name_or_path} \
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
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --mask_input ${mask_input} \
    --dataloader_num_workers ${worker} \
    --bf16 True \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap ${layername}

echo ${out_dir}