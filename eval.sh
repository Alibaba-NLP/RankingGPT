model=$1
data_name=$2
rank_name=$3

output_dir="./rankdata/${data_name}/${rank_name}/"
mkdir -p ${output_dir}
echo $output_dir

recall_path="./rankdata/${data_name}/top1000.json"
qrel_path="./rankdata/${data_name}/qrels.txt"


echo ${rank_name}
echo ${recall_path}
echo ${qrel_path}


python eval.py \
    --model_path $model \
    --res_path "${output_dir}/$rank_name.res" \
    --rank_path $recall_path \
    --data_name $data_name

# ndcg
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 ${qrel_path} ${output_dir}/${rank_name}_post.res > ${output_dir}/${rank_name}_score.txt
cat ${output_dir}/${rank_name}_score.txt