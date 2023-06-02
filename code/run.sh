time=$(date "+%Y-%m-%d_%H:%M:%S")
# nohup python main.py --config_file "./configs/main_experiments/bart_base_contrastive_fine-tuning.yaml" > contrast_${time}.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1,2 nohup python -m torch.distributed.launch main.py --config_file "./configs/pred_order_bart.yaml" > contrast_${time}.log 2>&1 &






nohup python -m torch.distributed.launch --nproc_per_node=3 main.py > contrast_${time}.log 2>&1 &
# nohup python main.py > contrast_${time}.log 2>&1 &

echo $! # output the pid of process. http://129.226.226.195/post/13717.html
echo $time


#-m torch.distributed.launch