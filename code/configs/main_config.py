# train
config = {
    "debug": "False",
    "annotation": 'only original | pre_order=False, contrastive fine-tuning | pre_train=False, margin=0.5, train_batch = 32',

    # gpu-related 
    "gpuid": "0,1",
    "use_gpu": True,
    "multi-gpu": True,
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
    "pred_order": False, # new add 
    "pretrain": False, # event-centric set to True, contrastive_fine-tuning set to False.
    # "checkpoint" : "/sdc/wwc/mcnc-main/cache/checkpoints/03-28/2023-03-28_14:17:41_original | pre_order=False, event-centric | pre_train=True, margin=0.4, train_batch = 128/best_checkpoint.pt", # event-centric set to True, contrastive_fine-tuning set to False.
    "checkpoint" : "/sdc/wwc/mcnc-main/cache/checkpoints/03-29/2023-03-29_13:21:07_only original | pre_order=False, event-centric | pre_train=True, margin=0.5, train_batch = 128/best_checkpoint.pt",
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
    "per_gpu_train_batch_size": 32, 
    "max_train_steps": 0,
    "eval_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "train": True,
    "eval": False, # test
    "test": True,
    "fp16": False,
    "num_train_epochs" : 100,
    "patience": 5,
    "log_step": 10,

    "lr": 1.0e-5, #learning rate 2e-6
    "lr_scheduler_type": "constant", # constant cosine
    "weight_decay": 1.0e-6,  # 1.0e-6 #1e-2
    "epsilon": 1.0e-8,
    
}






# /sdc/wwc/mcnc-main/cache/checkpoints/03-10/2023-03-10_10:20:19_bart, contrastive_fine-tuning, modify max_length_to_50., lr:1.0-5, the input is nine shuffled events, the output is ordered events./best_checkpoint.pt