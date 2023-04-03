from sre_constants import RANGE
from torch import nn
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartLearnedPositionalEmbedding
from transformers.modeling_outputs import Seq2SeqLMOutput
from models.base.cot import ComplementEntropy
from torch.nn import CrossEntropyLoss
import torch
from transformers import BartTokenizer
import logging
logger = logging.getLogger()




class bart_mask_random(nn.Module):

    def __init__(self, args):
        super(bart_mask_random, self).__init__()

        
        self.mlm = BartForConditionalGeneration.from_pretrained(args.pretrained_model_path)
        self.mlm_original = BartForConditionalGeneration.from_pretrained(args.pretrained_model_path)

        self.tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_path)

        # add <sep> tokens
        special_tokens = ["<sep>"] # keep consistent with "bart_dataset_custom.py"
        self.tokenizer.add_tokens(special_tokens)
        self.mlm.resize_token_embeddings(len(self.tokenizer))
        self.mlm_original.resize_token_embeddings(len(self.tokenizer))


        # set encoders of two mlm as the same.
        self.mlm_original.model.encoder = self.mlm.model.encoder


        self.args = args
        self.config = self.mlm.config

    # # tage 1: pred the position
    # def forward(
    #     self,encode_inputs,encode_masks,decode_inputs,decode_masks,targets
    # ):
    #     if self.training:
    #         outputs = self.mlm(
    #             input_ids=encode_inputs,
    #             attention_mask=encode_masks,
    #             decoder_input_ids = decode_inputs,
    #             decoder_attention_mask = decode_masks
    #         )
    #         logits = outputs.logits # [batch, 50, 50265]
    #         shift_logits = logits[..., :-1, :].contiguous() #[batch, 49, 50265]
    #         shift_labels = decode_inputs[..., 1:].contiguous() # [batch, 49]

    #         shift_labels[decode_inputs[:, 1:] == self.tokenizer.pad_token_id] = -100
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #         return loss, None, None
    #     else:
    #         self.args.beams = 1
    #         self.args.temperature = 1.0
    #         self.args.p = 0
    #         self.args.k = 0

    #         targets = torch.tensor(targets, dtype=torch.int).cuda()
    #         outputs = self.mlm.generate(
    #             encode_inputs,
    #             do_sample=self.args.beams == 0,
    #             max_length=self.args.decode_max_length,
    #             min_length=13, # 8 + 5 events
    #             temperature=self.args.temperature,
    #             top_p=self.args.p if self.args.p > 0 else None,
    #             top_k=self.args.k if self.args.k > 0 else None,
    #             num_beams=self.args.beams if self.args.beams > 0 else None,
    #             early_stopping=True,
    #             no_repeat_ngram_size=2,
    #             eos_token_id=self.tokenizer.eos_token_id,
    #             decoder_start_token_id=self.tokenizer.bos_token_id,
    #             num_return_sequences=1 #max(1, args.beams)
    #         )

    #         # verity whehter the generation is done as expectation. Intutively, we should get legal output in eval phase.
    #         legal_flag = True
    #         legal_str = [str(i) for i in range(9)] + [str(20)]
    #         preds = [x.strip("").split(" ") for x in [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]]
    #         pred_flags = [] # To summarize which sample is predicted to be right.
    #         for index, pred in enumerate(preds): # whether the len of preds is equal to 13 and no illegal str.
    #             if len(pred) != 13:
    #                 logger.info("Illegal output in dataloader, the preds are {}, the golden targets are {}".format(pred, targets[index]))
    #                 continue # if the len of current pred is not equal to 13, contiune 
    #             for index_2, x in enumerate(pred):
    #                 if x not in legal_str or x == "":
    #                     legal_flag = False
    #                     break
    #                 else:
    #                     pred[index_2] = int(x)

    #             if legal_flag is False:
    #                 logger.info("Illegal output in dataloader, the preds are {}, the golden targets are {}".format(pred, targets[index]))
    #                 legal_flag = True
    #                 continue
    #             else:
    #                 pred = torch.tensor(pred, dtype=torch.int).cuda()
    #                 ground_pos = pred[targets[index]==8]
    #                 cand_pos = pred[targets[index]==20].view(-1, 4)
    #                 pred_flag = torch.sum(ground_pos.unsqueeze(-1).repeat(1,4).view(-1,4) < cand_pos, dim=-1) == 4
    #                 pred_flags.append(pred_flag)
            
    #         batch_acc = torch.sum(torch.tensor(pred_flags)).item() / len(pred_flags) if len(pred_flags) != 0 else 0
    #         return None, None, batch_acc
    #         # if legal_flag:
    #         #     for pred in preds:
    #         #         for index, x in enumerate(pred):
    #         #             pred[index] = int(x)
    #         #     preds = torch.tensor(preds, dtype=torch.int).cuda()
    #         #     targets = torch.tensor(targets, dtype=torch.int).cuda()
    #         #     ground_pos = preds[targets==8]
    #         #     cand_pos = preds[targets==20].view(-1,4)
    #         #     batch_acc = torch.sum(torch.sum(ground_pos.unsqueeze(-1).repeat(1,4).view(-1,4) < cand_pos, dim=-1) == 4) / outputs.shape[0]
    #         #     return None, None, batch_acc.item() 
    #         # else:
    #         #     return None, (preds, targets), None



    def event_centric_stage(self, inputs, model):
        encode_inputs,encode_masks,decode_inputs,decode_masks,labels,targets = tuple(inputs)
        batch_size,decode_len = decode_inputs.size()
        encode_len = encode_inputs.size()[-1]
        labels[labels == self.config.pad_token_id] = -100
        outputs = model(
            input_ids=encode_inputs,
            attention_mask=encode_masks,
            decoder_input_ids = decode_inputs,
            decoder_attention_mask = decode_masks
        )
        logits = outputs.logits # [batch, 50, 50265]
        shift_logits = logits[..., :-1, :].contiguous() #[batch, 49, 50265]
        shift_labels = labels[..., 1:].contiguous() # [batch, 49]
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, None, None
    


    def fine_tuning_stage(self, inputs, model):
        encode_inputs,encode_masks,decode_inputs,decode_masks,labels,targets = tuple(inputs)
        batch_size,num_choices,decode_len = decode_inputs.size()
        encode_len = encode_inputs.size()[-1]

        encode_inputs = encode_inputs.reshape(-1,encode_len)
        encode_masks = encode_masks.reshape(-1,encode_len)
        decode_inputs = decode_inputs.reshape(-1,decode_len)
        decode_masks = decode_masks.reshape(-1,decode_len)
        labels = labels.reshape(-1,decode_len)
        labels[labels == self.config.pad_token_id] = -100
        outputs = model(
            input_ids=encode_inputs,
            attention_mask=encode_masks,
            decoder_input_ids = decode_inputs,
            decoder_attention_mask = decode_masks,
            output_attentions=True
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        logits = logits.reshape(batch_size,num_choices,decode_len-1) 
        # logits = logits/self.args.temperature
        

        # sum the logits of tokens for one choice.
        if self.args.pro_type=='sqrt':
            with torch.no_grad():
                nonzero = torch.count_nonzero(logits,dim=2)+self.args.denominator_correction_factor #the num of none zero is equal to num of token_mask != -100
            logits = -(torch.sum(logits,dim=2)/nonzero)
        elif self.args.pro_type=='mul':
            with torch.no_grad():
                nonzero = torch.count_nonzero(logits,dim=2)
            logits = -(torch.sum(logits,dim=2)+torch.log(nonzero.float()))
        else:
            logits = -torch.sum(logits,dim=2)
            

        # calculate loss based on choice score
        if self.args.loss_fct == 'CrossEntropyLoss':
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits,targets)
        elif self.args.loss_fct == 'MarginRankingLoss':
            # if self.args.temperature != 0:
            #     logits = logits/self.args.temperature
            if self.args.softmax:
                logits = torch.softmax(logits,dim=1)
            scores_false = []
            scores_true = []
            for score,target in zip(logits,targets):
                score_false =  torch.stack([score[i] for i in range(num_choices) if i != target ])
                score_true = torch.stack([score[target] for i in range(num_choices-1)])
                scores_false.append(score_false)
                scores_true.append(score_true)
            scores_false = torch.stack(scores_false).view(-1)
            scores_true = torch.stack(scores_true).view(-1)
            loss_fct = nn.MarginRankingLoss(margin=self.args.margin)
            loss = loss_fct(scores_true,scores_false,torch.ones_like(scores_true))
        elif self.args.loss_fct == 'ComplementEntropy':
            if self.args.dynamic_weight:
                p = torch.softmax(logits, dim=-1) #[batch, 5]
                log_p = torch.log(p)
                with torch.no_grad():
                    sample_weight = (1 - (-torch.sum(p * log_p, dim=-1) / torch.log(torch.tensor([5.0],device=self.args.device)))) * 10
                logger.info(sample_weight)
                loss1 = (sample_weight * -torch.gather(log_p, dim=-1, index=targets.unsqueeze(1))).mean()
            else:
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(logits,targets)
            loss_fct = ComplementEntropy()
            loss2 = loss_fct(logits,targets)
            loss = loss1+self.args.beta*loss2
            # loss = loss1
        return loss,logits,targets




    def evaluate_stage(self, inputs, flag = "original"):
        encode_inputs,encode_masks,decode_inputs,decode_masks,labels,targets = tuple(inputs)
        if flag == "original":
            model = self.mlm_original
        else:
            model = self.mlm


        batch_size,num_choices,decode_len = decode_inputs.size()
        encode_len = encode_inputs.size()[-1]

        encode_inputs = encode_inputs.reshape(-1,encode_len) #[batch, 5, encode_len] -> [batch*5, encode_len] 
        encode_masks = encode_masks.reshape(-1,encode_len)
        decode_inputs = decode_inputs.reshape(-1,decode_len) #[batch, 5, decode_len] -> [batch*5, decode_len] 
        decode_masks = decode_masks.reshape(-1,decode_len)
        labels = labels.reshape(-1,decode_len) #[batch, 5, decode_len]

        labels[labels == self.config.pad_token_id] = -100
        outputs = model(
            input_ids=encode_inputs,
            attention_mask=encode_masks,
            decoder_input_ids = decode_inputs,
            decoder_attention_mask = decode_masks
        )

        logits = outputs.logits #[batch*5, decode_len, vocab_size]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        logits = logits.reshape(batch_size,num_choices,decode_len-1)


        if self.args.pro_type=='sqrt':
            logits = -(torch.sum(logits,dim=2)/(torch.count_nonzero(logits,dim=2))+self.args.denominator_correction_factor)
        elif self.args.pro_type=='mul':
            logits = -(torch.sum(logits,dim=2)+torch.log(torch.count_nonzero(logits,dim=2).float()))
        else:
            logits = -torch.sum(logits,dim=2)
        
        return None,logits,None

    def forward(
        self, pre_order_inputs, original_inputs
    ):
        
        if self.args.pretrain and self.training: #event-centric training
            pre_order_loss, _, _ = self.event_centric_stage(pre_order_inputs, self.mlm)
            original_loss, _, _ = self.event_centric_stage(original_inputs, self.mlm_original)
            all_loss = pre_order_loss + original_loss
            # all_loss = original_loss 
            return all_loss, None, None
    
        elif self.args.pretrain==False and self.training: #contrasitve fine-tuning
            pre_order_loss, pre_order_logits, pre_order_targets = self.fine_tuning_stage(pre_order_inputs, self.mlm)
            original_loss, original_logits, original_targets = self.fine_tuning_stage(original_inputs, self.mlm_original)

            assert torch.equal(pre_order_targets, original_targets)

            # all_loss = pre_order_loss + original_loss
            all_loss = original_loss

            # all_logits = pre_order_logits + original_logits
            all_logits = original_logits
            return all_loss, all_logits, pre_order_targets

        else:
            
            _, original_logits, _ = self.evaluate_stage(original_inputs, flag="original")
            _, pre_order_logits, _ = self.evaluate_stage(pre_order_inputs, flag="pre_order")

            # all_logits = pre_order_logits + original_logits
            # all_logits = original_logits
            all_logits = pre_order_logits
            return None, all_logits, None


