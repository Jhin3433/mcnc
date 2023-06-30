from sre_constants import RANGE
from torch import nn
# from models.my_modeling_bart import BartForConditionalGeneration,BartLearnedPositionalEmbedding
from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartLearnedPositionalEmbedding
from transformers.modeling_outputs import Seq2SeqLMOutput
from models.base.cot import ComplementEntropy 
from torch.nn import CrossEntropyLoss
import torch
from transformers import BartTokenizer
import logging
import random
logger = logging.getLogger()
from models.OPA_supervised_binomial import BlurContrastiveModelPair




class bart_mask_random(nn.Module):

    def __init__(self, args):
        super(bart_mask_random, self).__init__()

        self.mlm = BartForConditionalGeneration.from_pretrained(args.pretrained_model_path)
        self.tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_path)


        self.args = args
        self.config = self.mlm.config


        self.alighment_model = BlurContrastiveModelPair(input_dim=768)


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


        all_loss = loss #+ position_loss
        return all_loss, logits.reshape(batch_size, -1), None
    


    def fine_tuning_stage(self, inputs, model):
        encode_inputs,encode_masks,decode_inputs,decode_masks,labels,encode_event_spans,targets = tuple(inputs)
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
            output_attentions=True,
            encode_event_spans=encode_event_spans
        )
        logits = outputs.logits # [batch * choice_num, decode_len - 1, vocab_size]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) #[batch * choince_num * decode_len - 1]
        logits = logits.reshape(batch_size,num_choices,decode_len-1) 



        # sum the logits of tokens for one choice.
        if self.args.pro_type=='sqrt':
            with torch.no_grad():
                nonzero = torch.count_nonzero(logits,dim=2)+self.args.denominator_correction_factor #the num of none zero is equal to num of token_mask != -100
            logits = -(torch.sum(logits,dim=2)/nonzero)

        if self.args.loss_fct == 'ComplementEntropy':
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(logits,targets) 
            loss_fct = ComplementEntropy()
            loss2 = loss_fct(logits,targets) 
            loss = loss1 + self.args.beta * loss2

        return loss,logits,targets




    def evaluate_stage(self, inputs):
        encode_inputs,encode_masks,decode_inputs,decode_masks,labels,mask_indexes,targets = tuple(inputs)

        batch_size,num_choices,decode_len = decode_inputs.size()
        encode_len = encode_inputs.size()[-1]

        encode_inputs = encode_inputs.reshape(-1,encode_len) #[batch, 5, encode_len] -> [batch*5, encode_len] 
        encode_masks = encode_masks.reshape(-1,encode_len)
        decode_inputs = decode_inputs.reshape(-1,decode_len) #[batch, 5, decode_len] -> [batch*5, decode_len] 
        decode_masks = decode_masks.reshape(-1,decode_len)
        labels = labels.reshape(-1,decode_len) #[batch, 5, decode_len]

        labels[labels == self.config.pad_token_id] = -100


        outputs = self.mlm(
            input_ids=encode_inputs,
            attention_mask=encode_masks,
            decoder_input_ids = decode_inputs,
            decoder_attention_mask = decode_masks
        )

        logits, _ = outputs.logits #[batch*5, decode_len, vocab_size]
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
        
        return None,logits,targets

    def forward(
        self, pre_order_inputs
    ):
        
        if self.args.stage_mode == "event-centric" and self.training: #event-centric training
            pre_order_loss, pre_order_logits, _ = self.event_centric_stage(pre_order_inputs, self.mlm)
            return pre_order_loss, None, None
    
        elif self.args.stage_mode == "fine-tuning" and self.training: #contrasitve fine-tuning
            pre_order_loss, pre_order_logits, pre_order_targets = self.fine_tuning_stage(pre_order_inputs, self.mlm)

            return pre_order_loss, pre_order_logits, pre_order_targets

        else:
            assert self.args.stage_mode == "evaluate_model"
            _, pre_order_logits, pre_order_targets = self.evaluate_stage(pre_order_inputs)

            return None, pre_order_logits, pre_order_targets


