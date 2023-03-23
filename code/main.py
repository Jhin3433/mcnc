import gc
import pickle
import torch
import random
import time
import logging
import math
import datetime
import numpy as np
import os
import argparse
from datetime import date
from models.base.cot import ComplementEntropy
from transformers import RobertaTokenizer,get_linear_schedule_with_warmup,RobertaForMultipleChoice,AdamW,get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from models.bart_1cls import bart_1cls
from multiprocessing import Pool
from torch.optim import Adam
from tools.common import seed_everything,Args,format_time
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_rank
from torch.utils.tensorboard import SummaryWriter  
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from apex import amp

all_pred_labels = []
all_ground_labels = []

def class_acc(preds, labels):    
    global all_pred_labels
    global all_ground_labels
    
    pred_labes = torch.max(preds, dim=1)[1]
    correct = torch.eq(pred_labes, labels.flatten()).float()         
    acc = correct.sum().item() / len(correct)

    all_pred_labels.extend(list(pred_labes.cpu().numpy()))
    all_ground_labels.extend(list(labels.cpu().numpy()))
    return acc

def train(args,train_dataloader,model,optimizer,lr_scheduler,writer,logger=None,global_step=0):
    t0 = time.time()
    avg_loss, avg_acc = [],[]
    print("Let's use {} GPUs.".format(torch.cuda.device_count()))
    print("Current Rank {} .".format(args.local_rank))
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    model.zero_grad()
    for step, return_batch in enumerate(train_dataloader):
        model.train()
        if len(return_batch) == 2: 
            (batch,batch_original) = return_batch
            batch = [t.long() for t in batch]
            batch = tuple(t.to(args.device) for t in batch)
            batch_original = [t.long() for t in batch_original]
            batch_original = tuple(t.to(args.device) for t in batch_original)    
            output = model(batch, batch_original)
        else: # remain the original code.
            batch_original = return_batch
            batch_original = [t.long() for t in batch_original]
            batch_original = tuple(t.to(args.device) for t in batch_original)
            output = model(batch_original)
        loss, logits, labels = output
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        loss = loss.item()
        # record avg_loss and avg_cc.
        avg_loss.append(loss)
        if labels is None :
            labels =  batch_original[-1]
        acc = class_acc(logits, labels) if logits!=None else 0 #the return value of logits event-centric training is none, so Train_Acc:0.0000 Train_Avg_acc:0.0000  
        avg_acc.append(acc)
        # update the model.
        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0*args.gradient_accumulation_steps)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            global_step += 1
        # write data to tensorboard.
        if global_step % 10==0 and args.local_rank in [-1,0] and args.debug is False:
                writer.add_scalar('loss', loss, global_step) # loss of current step/batch 
                writer.add_scalar('train_avg_acc', np.array(avg_acc).mean(), global_step)  #the same as Avg_acc in log file: average of acc of previous steps/batches. 
                writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], global_step)
        # logging data.
        if (step+1) % args.log_step == 0 and args.local_rank in [-1,0]:
            elapsed = format_time(time.time() - t0)
            logger.info('Batch {:>5,} of {:>5,}.Loss: {:} Train_Acc:{:} Train_Avg_acc:{:} Elapsed:{:}.' # acc in log file: acc of current step/batch, Avg_acc in log file: average of acc of previous steps/batches. 
            .format(step+1, len(train_dataloader),format(loss, '.4f'),format(acc, '.4f'),format(np.array(avg_acc).mean(), '.4f'),elapsed))
        # debug for evaluate, omit it later.
        # model.eval()
        # with torch.no_grad():
        #     loss, preds, labels = model(*tuple(batch))
        # logger.info(preds)
        
        
        # break # test evaluate()

    
    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc, global_step 


def evaluate(test_dataloader,model,args,logger):
    avg_acc = []
    model.eval()   
    with torch.no_grad():
        for step, return_batch in enumerate(test_dataloader):
            if len(return_batch) == 2:
                batch, batch_original = return_batch
                batch = [t.long() for t in batch]
                batch = tuple(t.to(args.device) for t in batch)
                batch_original = [t.long() for t in batch_original]
                batch_original = tuple(t.to(args.device) for t in batch_original)    
                loss, logits, labels = model(batch, batch_original)
            else:
                batch_original = return_batch
                batch_original = tuple(t.to(args.device) for t in batch_original)
                batch_original = [t.long() for t in batch_original]
                loss, logits, labels = model(batch_original)


            if labels is None :
                labels =  batch_original[-1] 

            acc = class_acc(logits, labels)
            avg_acc.append(acc)


                ## tage 1: pred the digital position.
                # _, _, batch_acc = model(*tuple(batch))
                # if batch_acc is not None: # prevent the output of model being None. set for pre_order task.
                #     avg_acc.append(batch_acc)
                ## tage 1: pred the position
    avg_acc = np.array(avg_acc).mean()
    return avg_acc

def str2bool(v): #https://blog.csdn.net/a15561415881/article/details/106088831
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':



    
    config = {
        "debug": "False",
        "annotation": 'all loss and logits, pre_train==False, modify max_length_to_50., lr:1.0-6',

        # gpu-related 
        "gpuid": "0,1,2,3",
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


        # model parameter
        "pred_order": True, # new add 
        "pretrain": False, #event-centric set to True, contrastive_fine-tuning set to False.
        "checkpoint" : "/sdc/wwc/mcnc-main/cache/checkpoints/03-20/2023-03-20_17:02:33_all loss and logits, pre_train==True, modify max_length_to_50., lr:1.0-5, the input is nine shuffled events, the output is ordered events./best_checkpoint.pt", # event-centric set to True, contrastive_fine-tuning set to False.
        # "checkpoint" : "", # event-centric set to True, contrastive_fine-tuning set to False.
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


        # training hyper parameter
        "seed": 970106,
        "per_gpu_train_batch_size": 16, 
        "max_train_steps": 0,
        "eval_batch_size": 64,
        "gradient_accumulation_steps": 2,
        "train": True,
        "eval": False, # test
        "test": True,
        "fp16": False,
        "num_train_epochs" : 100,
        "patience": 5,
        "log_step": 10,

        "lr": 2.0e-6, #learning rate
        "lr_scheduler_type": "constant", # constant cosine
        "weight_decay": 1.0e-6,  # 1.0e-6 #1e-2
        "epsilon": 1.0e-8,
        
    }


    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=str2bool, default=config["debug"], help="Debug or not.")
    parser.add_argument("--annotation", type=str, default=config["annotation"], help="annotation of running.")
  



    parser.add_argument("--gpuid", type=str, default=config["gpuid"], help="Running on which gpu.")
    parser.add_argument("--use_gpu", type=str2bool, default=config["use_gpu"], help="whether to use gpu or not.")
    parser.add_argument("--multi-gpu", type=str2bool, default=config["multi-gpu"], help="whether to use multi-gpu or not.")
    parser.add_argument("--local_rank", type=int, default=config["local_rank"], help="local rank.")
    parser.add_argument("--gpu_num", type=int, default=config["gpu_num"], help="gpu num.")
    # parser.add_argument("--nproc_per_node", type=int, default=2)


    parser.add_argument("--data_dir", type=str, default=config["data_dir"], help="datadir")
    parser.add_argument("--encode_max_length", type=int, default=config["encode_max_length"], help="encode_max_length")
    parser.add_argument("--decode_max_length", type=int, default=config["decode_max_length"], help="decode_max_length")
    parser.add_argument("--eval_decode_max_length", type=int, default=config["eval_decode_max_length"], help="eval_decode_max_length")
    parser.add_argument("--truncation", type=str2bool, default=config["truncation"], help="truncation or not.")
    parser.add_argument("--random_span", type=str2bool, default=config["random_span"], help="random_span or not.")

    
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid


    # set log path
    start_date = date.today().strftime('%m-%d')
    running_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
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
    
    # pre-ready
    torch.cuda.empty_cache()
    seed_everything(args.seed)
    if args.pred_order:
        from tools.bart_dataset_custom import bart_dataset
        from models.bart_custom import bart_mask_random
        MODEL_CLASSES = {
            'bart_1cls': bart_1cls,
            'bart_mask_random' : bart_mask_random,
        }

    else:
        from tools.bart_dataset_random import bart_dataset
        from models.bart_mask_random import bart_mask_random
        MODEL_CLASSES = {
            'bart_1cls': bart_1cls,
            'bart_mask_random' : bart_mask_random,
        }

    # set gpu and whehther perform distributed training or not.
    if args.multi_gpu and args.use_gpu:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12345'
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

    else :
        args.local_rank = -1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_gpu == False:
        device = torch.device('cpu')
    args.device = device

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
        logger.info("Process rank: {}, device: {}, distributed training: {}".format(
                    args.local_rank,device, bool(args.local_rank != -1)))
        # logger.info("Training/evaluation parameters %s", args.to_str())
        logger.info("Training/evaluation parameters %s", "\n".join([arg + " : " + str(getattr(args, arg)) for arg in vars(args)]))

    # whether to eval 
    if args.eval:
        model = MODEL_CLASSES[args.model_type](args)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'],strict=False)
        model.to(args.device)
        dev_batch = pickle.load(open(os.path.join(args.data_dir,'test/corpus_index_test.txt'),'rb'))
        dev_data = bart_dataset(dev_batch,args,'eval')
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
        test_acc = evaluate(dev_dataloader,model,args,logger=logger)
        logger.info("test_acc is {}".format(test_acc))


    if args.train:
        # load model
        model = MODEL_CLASSES[args.model_type](args)
        model.to(args.device)
        if args.resume:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(dict([(n, p) for n, p in checkpoint['model'].items()]), strict=False)
        if args.noise_lambda != 0:
            for name ,para in model.named_parameters():
                model.state_dict()[name][:] += (torch.rand(para.size())-0.5)*args.noise_lambda*torch.std(para)



        # set optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #optimizer_to(optimizer,args.device)  
        optimizer = Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)
        # complement_optimizer = Adam(optimizer_grouped_parameters,eps = args.epsilon,betas=(0.9,0.98),lr=args.lr)
        # len(data) = 1440295 140331

        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


        # set hyper parameters related to traing process.
        if args.data_dir.split('/')[-1] == 'negg_data':
            train_num = 140331
        else:
            train_num = 1440295
        num_update_steps_per_epoch = math.ceil(((train_num/(args.gpu_num * args.per_gpu_train_batch_size))) / args.gradient_accumulation_steps)
        if args.max_train_steps == 0:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        args.num_warmup_steps = args.max_train_steps * 0.05
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )    
    

        # set tensorboard.
        if args.local_rank in [-1,0]: #and args.debug is False:
            tensorboard_path = '../cache/tensorboard/{}/{}'.format(start_date, running_time + "_" + args.annotation)
            if not os.path.exists(os.path.dirname(tensorboard_path)):
                os.makedirs(os.path.dirname(tensorboard_path))
            writer = SummaryWriter(tensorboard_path)
        else:
            writer = None

        # pre ready for training.
        global_step = 0
        best_performance = 0
        best_checkpoint_path = None
        if args.debug:
            # train_raw_data = pickle.load(open('','rb'))
            train_raw_data = pickle.load(open(os.path.join(args.data_dir,'train/corpus_index_train0.txt'),'rb'))

        else:
            train_raw_data = pickle.load(open(os.path.join(args.data_dir,'train/corpus_index_train0.txt'),'rb'))
        patience = args.patience
        fail_time = 0
        start_epoch = 0
        for epoch in range(int(args.num_train_epochs)):
            if fail_time>=patience:
                break
            if epoch < start_epoch:
                continue
            if args.local_rank in [-1,0]:
                logger.info('local_rank={},epoch={}'.format(args.local_rank, epoch))
            train_dataset = bart_dataset(train_raw_data,args,'train')
            train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,num_workers=8)
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            train_loss, train_acc, global_step = train(args, train_dataloader,model,optimizer,lr_scheduler,writer,logger,global_step)
            if args.local_rank in [-1,0]:
                logger.info('epoch={},avg_train_acc={},avg_loss={}'.format(epoch, train_acc, train_loss))
            torch.cuda.empty_cache()
            gc.collect()


            #  Evaluation on ``eval'' dataset after traning an epoch.
            if args.local_rank in [-1,0]: #and not args.pred_order:
                dev_batch = pickle.load(open(os.path.join(args.data_dir,'dev/corpus_index_dev.txt'),'rb'))
                dev_data = bart_dataset(dev_batch,args,'eval')
                dev_sampler = SequentialSampler(dev_data)
                dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
                dev_acc = evaluate(dev_dataloader,model,args,logger=logger)
                writer.add_scalar('dev_acc', dev_acc, epoch)
                logger.info("epoch={},dev_acc={}".format(epoch,dev_acc))
                if dev_acc > best_performance:
                    checkpoints_path = '../cache/checkpoints/{}/{}/'.format(start_date, running_time + "_" + args.annotation)
                    if not os.path.exists(checkpoints_path):
                        os.makedirs(checkpoints_path)
                    best_checkpoint_path = os.path.join(checkpoints_path,'best_checkpoint.pt')
                    #optimizer_to(optimizer,torch.device('cpu'))
                    model.to(torch.device('cpu'))
                    #'optimizer':optimizer.state_dict(),
                    torch.save({'model':model.state_dict(),'epoch':epoch},best_checkpoint_path)
                    #optimizer_to(optimizer,args.device)
                    model.to(args.device)
                    best_performance = dev_acc
                    fail_time = 0
                    logger.info("best_performance={}, epoch={}".format(dev_acc, epoch))
                else:
                    fail_time+=1
    # Select the bestcheckpoint on eval dataset to evaluate on test dataset.
    if args.test and args.local_rank in [-1,0]:
        # logger.info("test best_checkpoint_path={}".format(best_checkpoint_path))
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        dev_batch = pickle.load(open(os.path.join(args.data_dir,'test/corpus_index_test.txt'),'rb'))
        dev_data = bart_dataset(dev_batch,args,'eval')
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size,num_workers=8)
        test_acc = evaluate(dev_dataloader,model,args,logger=logger)
        logger.info("best epoch={}, test_acc={}".format(checkpoint['epoch'], test_acc))



    torch.distributed.barrier()


if __name__=='__main__':
    main()

