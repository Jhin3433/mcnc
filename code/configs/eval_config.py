# eval 
config = {
    "debug": "False",
    "annotation": 'all loss and logits, pre_train==False, modify max_length_to_50., lr:1.0-6',

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


    # model parameter
    "pred_order": True, # new add 
    "pretrain": False, #event-centric set to True, contrastive_fine-tuning set to False.
    # "checkpoint" : "/sdc/wwc/mcnc-main/cache/checkpoints/03-20/2023-03-20_17:02:33_all loss and logits, pre_train==True, modify max_length_to_50., lr:1.0-5, the input is nine shuffled events, the output is ordered events./best_checkpoint.pt", # event-centric set to True, contrastive_fine-tuning set to False.
    # only mask original checkpoint
    # "checkpoint" : "/sdc/wwc/mcnc-main/cache/checkpoints/02-16/2023-02-16_15:35:02_bart_base_contrastive_fine-tuning/best_checkpoint.pt", # event-centric set to True, contrastive_fine-tuning set to False.
    # only pre_order checkpoint
    "checkpoint" : "/sdc/wwc/mcnc-main/cache/checkpoints/03-10/2023-03-10_10:20:19_bart, contrastive_fine-tuning, modify max_length_to_50., lr:1.0-5, the input is nine shuffled events, the output is ordered events./best_checkpoint.pt", # event-centric set to True, contrastive_fine-tuning set to False.
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
    "seed": 970106,
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