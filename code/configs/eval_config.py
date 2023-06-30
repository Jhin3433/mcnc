# eval 
config = {
    "debug": "True",
    "annotation": 'test on testdataset',

    # gpu-related 
    "gpuid": "0",
    "use_gpu": True,
    "multi-gpu": False,
    "local_rank": -1,
    "gpu_num": 1,

    # dataset-related
    "data_dir": "../data/negg_data",
    "encode_max_length": 100,
    "decode_max_length": 50,
    "eval_decode_max_length": 50,
    "truncation": True,
    "random_span": False,
    "mask_num": 3,


    # model parameter
    "event_position_mask_num" : 3,  # 后续待删除
    "pretrain": False, # 后续待删除


    "custom": True, # new add 
    "stage_mode": "evaluate_model", #event-centric set to True, contrastive_fine-tuning set to False.
    # "checkpoint": "/sdc/wwc/mcnc-main/cache/checkpoints/05-04/2023-05-04_09:30:01_original loss without position loss, token position_emb + event position_emb, dataset recover all events, contrast_learning -> pre_train=False, margin=0.5, train_batch = 32/best_checkpoint.pt",
    # "checkpoint" : "/sdc/wwc/mcnc-main/cache/checkpoints/03-28/2023-03-28_22:45:00_original | pre_order=False, contrastive fine-tuning | pre_train=False, margin=0.5, train_batch = 32, use 03-28_14:17:41.checkpoint/best_checkpoint.pt",
    # "checkpoint": "/sdc/wwc/mcnc-main/cache/checkpoints/06-21/2023-06-21_16:12:32_kernalsize=3&5&merge, contrastive-fine-tuning, original method, use . as the seprator, use the calculation of alignment distance, remove the limitation of the seq_length/best_checkpoint.pt",
    "checkpoint": "/sdc/wwc/mcnc-main/cache/checkpoints/06-23/2023-06-23_20:08:47_only event-centric/best_checkpoint.pt",
    "resume": True, # event-centric set to True, contrastive_fine-tuning set to False.
    "dynamic_weight": False, # new add 
    "beta" : 1,
    "loss_fct": "ComplementEntropy", # CrossEntropyLoss MarginRankingLoss 
    "model_type": "bart_mask_random",
    "pretrained_model_path": "../init/init_model/bart-base",
    "pro_type" : "sqrt",
    "softmax": True,
    "margin": 0.5,
    "noise_lambda": 0,
    "denominator_correction_factor" : 0,
    "vocab_size": 50265,

    # training hyper parameter
    "seed": 2022,
    "per_gpu_train_batch_size": 64, 
    "max_train_steps": 0,
    "eval_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "train": False,
    "eval": True, # test
    "test": False,
    "fp16": False,
    "num_train_epochs" : 100,
    "patience": 5,
    "log_step": 10,

    "lr": 2.0e-6, #learning rate
    "lr_scheduler_type": "constant", # constant cosine
    "weight_decay": 1.0e-6,  # 1.0e-6 #1e-2
    "epsilon": 1.0e-8,
    
}