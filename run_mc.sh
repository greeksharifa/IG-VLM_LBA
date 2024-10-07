llm_size=$1
dataset_name=$2
exp=$3
num_sub_qa_select=$4
device=$5
num_data=$6
echo "llm_size             : ${llm_size}"
echo "dataset_name         : ${dataset_name}"
echo "sub_qa_index and exp : ${exp}"
echo "num_sub_qa_select    : ${num_sub_qa_select}"
echo "device               : ${device}"
echo "num_data             : ${num_data}"

CUDA_VISIBLE_DEVICES=${device} python eval_llava_multiplechoice.py --path_qa_pair_csv ./data/multiple_choice_qa/${dataset_name}_sub_qas_val_xl_fewshot_vqaintrospect_unique.csv --path_video ./data/${dataset_name}/videos/%s.mp4 --path_result ./result_${dataset_name}_${llm_size}_select${num_sub_qa_select}_sub_qas_val_xxl_fvu/${exp} --llm_size ${llm_size}  --dataset_name ${dataset_name} --num_data ${num_data} --sub_qa_index ${exp} --num_sub_qa_select ${num_sub_qa_select}
# model_name = "llava-v1.6-vicuna-%s" % (llm_size)
# usage: bash run_mc.sh NExT_QA 4 10 base
# usage: bash run_mc.sh NExT_QA 4 10 1
