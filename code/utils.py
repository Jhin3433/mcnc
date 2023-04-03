import logging
import os
from datetime import date
import time
import torch
import argparse
import numpy as np
import random
def init_args(mode = "train"):
    def str2bool(v): #https://blog.csdn.net/a15561415881/article/details/106088831
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    if mode == "train":
        from configs.main_config import config
    elif mode == "eval":
        from configs.eval_config import config

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str2bool, default=config["debug"], help="Debug or not.")
    parser.add_argument("--annotation", type=str, default=config["annotation"], help="annotation of running.")
  
    # gpu-related
    parser.add_argument("--gpuid", type=str, default=config["gpuid"], help="Running on which gpu.")
    parser.add_argument("--use_gpu", type=str2bool, default=config["use_gpu"], help="whether to use gpu or not.")
    parser.add_argument("--multi-gpu", type=str2bool, default=config["multi-gpu"], help="whether to use multi-gpu or not.")
    parser.add_argument("--local_rank", type=int, default=config["local_rank"], help="local rank.")
    parser.add_argument("--gpu_num", type=int, default=config["gpu_num"], help="gpu num.")
    # parser.add_argument("--nproc_per_node", type=int, default=2)

    # dataset-related
    parser.add_argument("--data_dir", type=str, default=config["data_dir"], help="datadir")
    parser.add_argument("--encode_max_length", type=int, default=config["encode_max_length"], help="encode_max_length")
    parser.add_argument("--decode_max_length", type=int, default=config["decode_max_length"], help="decode_max_length")
    parser.add_argument("--eval_decode_max_length", type=int, default=config["eval_decode_max_length"], help="eval_decode_max_length")
    parser.add_argument("--truncation", type=str2bool, default=config["truncation"], help="truncation or not.")
    parser.add_argument("--random_span", type=str2bool, default=config["random_span"], help="random_span or not.")
    parser.add_argument("--mask_num", type=int, default=config["mask_num"], help="mask_num.")

    # model parameter
    parser.add_argument("--pred_order", type=str2bool, default=config["pred_order"], help="pred_order or not.")
    parser.add_argument("--pretrain", type=str2bool, default=config["pretrain"], help="pretrain or not.")
    parser.add_argument("--checkpoint", type=str, default=config["checkpoint"], help="checkpoint")
    parser.add_argument("--resume", type=str2bool, default=config["resume"], help="resume or not.")
    parser.add_argument("--dynamic_weight", type=str2bool, default=config["dynamic_weight"], help="dynamic_weight or not.")
    parser.add_argument("--beta", type=float, default=config["beta"], help="beta")
    parser.add_argument("--loss_fct", type=str, default=config["loss_fct"], help="loss_fct")
    parser.add_argument("--model_type", type=str, default=config["model_type"], help="model_type")
    parser.add_argument("--pretrained_model_path", type=str, default=config["pretrained_model_path"], help="pretrained_model_path")
    parser.add_argument("--pro_type", type=str, default=config["pro_type"], help="pro_type")
    parser.add_argument("--softmax", type=str2bool, default=config["softmax"], help="softmax or not.")
    parser.add_argument("--margin", type=float, default=config["margin"], help="margin")
    parser.add_argument("--noise_lambda", type=float, default=config["noise_lambda"], help="noise_lambda")
    parser.add_argument("--denominator_correction_factor", type=float, default=config["denominator_correction_factor"], help="denominator_correction_factor")
    parser.add_argument("--vocab_size", type=float, default=config["vocab_size"], help="vocab_size")


    # training hyper parameter
    parser.add_argument("--seed", type=int, default=config["seed"], help="seed")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=config["per_gpu_train_batch_size"], help="per_gpu_train_batch_size")
    parser.add_argument("--max_train_steps", type=int, default=config["max_train_steps"], help="max_train_steps")
    parser.add_argument("--eval_batch_size", type=int, default=config["eval_batch_size"], help="eval_batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config["gradient_accumulation_steps"], help="gradient_accumulation_steps")
    parser.add_argument("--train", type=str2bool, default=config["train"], help="train or not.")
    parser.add_argument("--eval", type=str2bool, default=config["eval"], help="eval or not.")
    parser.add_argument("--test", type=str2bool, default=config["test"], help="test or not.")
    parser.add_argument("--fp16", type=str2bool, default=config["fp16"], help="fp16 or not.")
    parser.add_argument("--num_train_epochs", type=int, default=config["num_train_epochs"], help="num_train_epochs")
    parser.add_argument("--patience", type=int, default=config["patience"], help="patience")
    parser.add_argument("--log_step", type=int, default=config["log_step"], help="log_step")
    parser.add_argument("--lr", type=float, default=config["lr"], help="lr")
    parser.add_argument("--lr_scheduler_type", type=str, default=config["lr_scheduler_type"], help="lr_scheduler_type")
    parser.add_argument("--weight_decay", type=float, default=config["weight_decay"], help="weight_decay")
    parser.add_argument("--epsilon", type=float, default=config["epsilon"], help="epsilon")

    args = parser.parse_args()
    return args



def init_logger(args, start_date, running_time):

    # set log path

    if args.eval:
        log_path = '../cache/log/{}/{}-eval.log'.format(start_date, running_time + "_" + args.annotation)
    else:
        if args.debug:
            # modify the annotation for the prevention of log file name conflict.
            log_path = '../cache/log/{}/{}.log'.format(start_date, "debug" + "_" + args.annotation)
        else:
            log_path = '../cache/log/{}/{}.log'.format(start_date, running_time + "_" + args.annotation)
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    # set logger
    logger = None
    if args.local_rank in [-1,0]:
        if args.eval:
            logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',
                datefmt='%Y/%m/%d %H:%M:%S',
                level=logging.INFO)
        else:
            logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s | %(message)s',
                datefmt='%Y/%m/%d %H:%M:%S',
                level=logging.INFO,
                filename=log_path,
                filemode="w")
        logger = logging.getLogger()
        # logger.info("Process rank: {}, device: {}, distributed training: {}".format(
        #             args.local_rank, args.device, bool(args.local_rank != -1)))
        # logger.info("Training/evaluation parameters %s", args.to_str())
        logger.info("Training/evaluation parameters %s\n", "\n".join([arg + " : " + str(getattr(args, arg)) for arg in vars(args)]))
    
    return logger

def init_device(args):
    # set gpu and whehther perform distributed training or not.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    if args.multi_gpu and args.use_gpu:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12345'
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    else :
        local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_gpu == False:
        device = torch.device('cpu')

    return device, local_rank


def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True