# xKV-4
# CUDA_VISIBLE_DEVICES=2 \
# python evaluate/eval_acc.py \
#     --datalen 8192 \
#     --batch_size 1 \
#     --dataset_name "ruler/niah_single_1" \
#     --model_name_or_path /data/shenth/models/SmolLM/1b \
#     --xKV \
#     --merge_k \
#     --merge_v \
#     --rank_k 512 \
#     --rank_v 512 \
#     --layer_group_size 4 \
#     --start_layer_idx 0 \
#     --end_layer_idx -1

CUDA_VISIBLE_DEVICES=3 \
TOKENIZERS_PARALLELISM=False \
python cka_model.py \
    --model_name_or_path /data/shenth/models/llama/2-7b-hf \
    --xKV \
    --customized_merge_config ./config/llama2-7.yaml