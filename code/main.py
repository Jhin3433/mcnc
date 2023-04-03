import gc
import pickle
import torch
import time
import math
import numpy as np
import os
from datetime import date
from transformers import RobertaTokenizer,get_linear_schedule_with_warmup,RobertaForMultipleChoice,AdamW,get_scheduler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from models.bart_1cls import bart_1cls
from multiprocessing import Pool
from torch.optim import Adam
from tools.common import Args,format_time
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_rank
from torch.utils.tensorboard import SummaryWriter  
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from apex import amp

from utils import seed_everything
from utils import init_args
from utils import init_device
from utils import init_logger

all_pred_labels = []
all_pred_logits = []
all_ground_labels = []



def class_acc(preds, labels):    

    pred_labes = torch.max(preds, dim=1)[1]

    # pred_labes = torch.max(preds, dim=1)[1]
    correct = torch.eq(pred_labes, labels.flatten()).float()         
    acc = correct.sum().item() / len(correct)


    # results annalysis 
    # global all_pred_labels
    # global all_ground_labels
    # global all_pred_logits
    # all_pred_logits.extend(list(preds.cpu().numpy()))
    # all_pred_labels.extend(list(pred_labes.cpu().numpy()))
    # all_ground_labels.extend(list(labels.cpu().numpy()))
    return acc

def train(args,train_dataloader,model,optimizer,lr_scheduler,writer,logger=None,global_step=0):
    t0 = time.time()
    avg_loss, avg_acc = [],[]
    # print("Let's use {} GPUs.".format(torch.cuda.device_count()))
    # print("Current Rank {} .".format(args.local_rank))
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
        # for step, return_batch in enumerate(test_dataloader):
        for return_batch in tqdm(test_dataloader,total=len(test_dataloader)):
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


if __name__ == '__main__':
    start_date = date.today().strftime('%m-%d')
    running_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # pre-ready
    args = init_args(mode = "train")
    logger = init_logger(args, start_date, running_time)
    args.device, args.local_rank = init_device(args)
    if args.local_rank in [-1,0]:
        logger.info("Process rank: {}, device: {}, distributed training: {}".format(
                    args.local_rank, args.device, bool(args.local_rank != -1)))

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

