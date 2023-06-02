from sre_constants import RANGE
from torch import nn
from models.my_modeling_bart import BartForConditionalGeneration,BartLearnedPositionalEmbedding
# from transformers.models.bart.modeling_bart import BartForConditionalGeneration,BartLearnedPositionalEmbedding
from transformers.modeling_outputs import Seq2SeqLMOutput
from models.base.cot import ComplementEntropy 
from torch.nn import CrossEntropyLoss
import torch
from transformers import BartTokenizer
import logging
import random
logger = logging.getLogger()




class bart_mask_random(nn.Module):

    def __init__(self, args):
        super(bart_mask_random, self).__init__()

        self.mlm = BartForConditionalGeneration.from_pretrained(args.pretrained_model_path)
        self.tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_path)

        # add <sep> tokens
        special_tokens = ["<sep>"] # keep consistent with "bart_dataset_custom.py"
        self.tokenizer.add_tokens(special_tokens)
        self.mlm.resize_token_embeddings(len(self.tokenizer))

        self.args = args
        self.config = self.mlm.config
 
    def text_span(self, input_ids,mask_indexes):
        special_id_in_input_ids = [0, 1, 2, 50265] # <s> <pad> </s> <sep> 
        pre_defined_max_position = 9
        text_spans = []
        span_index = None
        position_value = None

        for i in range(input_ids.shape[0]):
            # 随机 position mask
            # pos_mask_num = random.randint(1,self.args.event_position_mask_num)
            # position_mask_indexs = random.sample(range(0, 9),pos_mask_num)
            # position mask和event mask一致
            # position_mask_indexs = mask_indexes[i]
            # 无position mask
            position_mask_indexs = []
            current_position = 0
            start = 1
            text_spans.append([])
            for index in range(input_ids[i].shape[0]):
                if input_ids[i][index] == 0:
                    continue
                if input_ids[i][index] == 50265:
                    text_spans[-1].extend([start, index - 1])
                    

                    # pre-ready for fillment of event_position 
                    x_axis = [i for _ in range(index - start)]
                    y_axis = [y for y in range(start, index)]
                    position_v = [current_position + 5 if current_position not in position_mask_indexs else 4 for _ in range(index - start)] # position bias = 5
                    if span_index == None and position_value == None:
                        span_index = [
                            torch.LongTensor(x_axis),
                            torch.LongTensor(y_axis)
                        ]
                        position_value = torch.tensor(position_v, dtype=torch.int8)
                    else:
                        span_index[0] = torch.cat((span_index[0], torch.LongTensor(x_axis)))
                        span_index[1] = torch.cat((span_index[1], torch.LongTensor(y_axis)))
                        position_value = torch.cat((position_value, torch.LongTensor(position_v)))
                    start = index + 1
                    current_position = current_position + 1
                elif input_ids[i][index] == 2:
                    text_spans[-1].extend([start, index - 1])

                    x_axis = [i for _ in range(index - start)]
                    y_axis = [y for y in range(start, index)]
                    position_v = [current_position + 5 if current_position not in position_mask_indexs else 4 for _ in range(index - start)] # position bias = 5
                    if span_index == None and position_value == None:
                        span_index = (
                            torch.LongTensor(x_axis),
                            torch.LongTensor(y_axis)
                        )
                        position_value = torch.tensor(position_v, dtype=torch.int8)
                    else:
                        span_index[0] = torch.cat((span_index[0], torch.LongTensor(x_axis)))
                        span_index[1] = torch.cat((span_index[1], torch.LongTensor(y_axis)))
                        position_value = torch.cat((position_value, torch.LongTensor(position_v)))
                    current_position = current_position + 1
                    break         
            
        text_spans = torch.tensor(text_spans, device=self.mlm.device)

        return text_spans, (span_index, position_value)

    def event_centric_stage(self, inputs, model):
        encode_inputs,encode_masks,decode_inputs,decode_masks,labels,mask_indexes,targets = tuple(inputs)
        batch_size,decode_len = decode_inputs.size()
        encode_len = encode_inputs.size()[-1]
        labels[labels == self.config.pad_token_id] = -100

        text_spans, construction_event_position_matrix = None, None
        # text_spans, construction_event_position_matrix = self.text_span(encode_inputs,mask_indexes)
        outputs = model(
            input_ids=encode_inputs,
            attention_mask=encode_masks,
            decoder_input_ids = decode_inputs,
            decoder_attention_mask = decode_masks,
            text_spans=text_spans,
            construction_event_position_matrix=construction_event_position_matrix
        )
        logits, position_logits = outputs.logits # [batch, 50, 50265]
        shift_logits = logits[..., :-1, :].contiguous() #[batch, 49, 50265]
        shift_labels = labels[..., 1:].contiguous() # [batch, 49]
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))



        # # 遍历方法赋值 decode_event_positions
        # decode_event_positions = [[0 for _ in range(decode_len)] for b in range(batch_size)]
        # # _, len_mask = mask_indexes.shape[1]
        # for bt in range(batch_size):
        #     current_mask_index = 0
        #     for index in range(decode_len):
        #         id = decode_inputs[bt][index]
        #         if id == 0:
        #             decode_event_positions[bt][index] = 0    # <s> 对应的event postion id
        #         elif id == 1:
        #             decode_event_positions[bt][index] = 1    # <pad>
        #         elif id == 2:
        #             decode_event_positions[bt][index] = 2    # </s>
        #         elif id == 50265:
        #             decode_event_positions[bt][index] = 3    # <sep>
        #             current_mask_index += 1
        #         else:
        #             decode_event_positions[bt][index] = mask_indexes[bt][current_mask_index].item() + 5
        # decode_event_positions = torch.tensor(decode_event_positions, device=model.device) 

        # shift_position_logits = position_logits[..., :-1, :].contiguous() #[batch, 49, 50265]
        # shift_position_labels = decode_event_positions[..., 1:].contiguous() # [batch, 49]
        # loss_fct_2 = CrossEntropyLoss()
        # position_loss = loss_fct_2(shift_position_logits.view(-1, shift_position_logits.size(-1)), shift_position_labels.view(-1))

        all_loss = loss #+ position_loss
        return all_loss, logits.reshape(batch_size, -1), None
    


    def fine_tuning_stage(self, inputs, model):
        encode_inputs,encode_masks,decode_inputs,decode_masks,labels,mask_indexes,targets = tuple(inputs)
        batch_size,num_choices,decode_len = decode_inputs.size()
        encode_len = encode_inputs.size()[-1]

        mask_indexes = mask_indexes.reshape(-1, mask_indexes.shape[-1])
        encode_inputs = encode_inputs.reshape(-1,encode_len)
        encode_masks = encode_masks.reshape(-1,encode_len)
        decode_inputs = decode_inputs.reshape(-1,decode_len)
        decode_masks = decode_masks.reshape(-1,decode_len)
        labels = labels.reshape(-1,decode_len)
        labels[labels == self.config.pad_token_id] = -100

        text_spans, construction_event_position_matrix = None, None
        # text_spans, construction_event_position_matrix = self.text_span(encode_inputs, mask_indexes)
        outputs = model(
            input_ids=encode_inputs,
            attention_mask=encode_masks,
            decoder_input_ids = decode_inputs,
            decoder_attention_mask = decode_masks,
            output_attentions=True,
            text_spans=text_spans,
            construction_event_position_matrix=construction_event_position_matrix
        )
        logits, position_logits = outputs.logits # [batch * choice_num, decode_len - 1, vocab_size]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='none')
        logits = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) #[batch * choince_num * decode_len - 1]
        logits = logits.reshape(batch_size,num_choices,decode_len-1) 

        # # position logits
        # decode_event_positions = [[0 for _ in range(decode_len)] for b in range(batch_size * num_choices)]
        # for bt in range(batch_size):
        #     current_mask_index = 0
        #     for index in range(decode_len):
        #         id = decode_inputs[bt][index]
        #         if id == 0:
        #             decode_event_positions[bt][index] = 0    # <s> 对应的event postion id
        #         elif id == 1:
        #             decode_event_positions[bt][index] = 1    # <pad>
        #         elif id == 2:
        #             decode_event_positions[bt][index] = 2    # </s>
        #         elif id == 50265:
        #             decode_event_positions[bt][index] = 3    # <sep>
        #             current_mask_index += 1
        #         else:
        #             decode_event_positions[bt][index] = mask_indexes[bt][current_mask_index].item() + 5
        # decode_event_positions = torch.tensor(decode_event_positions, device=model.device)           
        # shift_position_logits = position_logits[..., :-1, :].contiguous() #[batch, 49, 50265]
        # shift_position_labels = decode_event_positions[..., 1:].contiguous() # [batch, 49]
        # loss_fct_2 = CrossEntropyLoss(reduction='none')
        # position_logits = loss_fct_2(shift_position_logits.view(-1, shift_position_logits.size(-1)), shift_position_labels.view(-1))
        # position_logits = position_logits.reshape(batch_size,num_choices,decode_len-1) 



        # sum the logits of tokens for one choice.
        if self.args.pro_type=='sqrt':
            with torch.no_grad():
                nonzero = torch.count_nonzero(logits,dim=2)+self.args.denominator_correction_factor #the num of none zero is equal to num of token_mask != -100
                nozero_position = torch.count_nonzero(position_logits,dim=2)+self.args.denominator_correction_factor
            logits = -(torch.sum(logits,dim=2)/nonzero)
            # position_logits = -(torch.sum(position_logits,dim=2)/nozero_position)  # position_logits
        # calculate loss based on choice score
        if self.args.loss_fct == 'ComplementEntropy':
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(logits,targets) #+ loss_fct(position_logits,targets) # add position loss

            loss_fct = ComplementEntropy()
            loss2 = loss_fct(logits,targets) #+ loss_fct(position_logits,targets) # add position loss
            loss = loss1 + self.args.beta * loss2

        return loss,logits,targets




    def evaluate_stage(self, inputs):
        encode_inputs,encode_masks,decode_inputs,decode_masks,labels,mask_indexes,targets = tuple(inputs)

        batch_size,num_choices,decode_len = decode_inputs.size()
        encode_len = encode_inputs.size()[-1]

        mask_indexes = mask_indexes.reshape(-1, mask_indexes.shape[-1])
        encode_inputs = encode_inputs.reshape(-1,encode_len) #[batch, 5, encode_len] -> [batch*5, encode_len] 
        encode_masks = encode_masks.reshape(-1,encode_len)
        decode_inputs = decode_inputs.reshape(-1,decode_len) #[batch, 5, decode_len] -> [batch*5, decode_len] 
        decode_masks = decode_masks.reshape(-1,decode_len)
        labels = labels.reshape(-1,decode_len) #[batch, 5, decode_len]

        labels[labels == self.config.pad_token_id] = -100

        text_spans, construction_event_position_matrix = None, None
        # text_spans, construction_event_position_matrix = self.text_span(encode_inputs, mask_indexes)
        outputs = self.mlm(
            input_ids=encode_inputs,
            attention_mask=encode_masks,
            decoder_input_ids = decode_inputs,
            decoder_attention_mask = decode_masks,
            text_spans=text_spans,
            construction_event_position_matrix=construction_event_position_matrix
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
        
        if self.args.pretrain and self.training: #event-centric training
            pre_order_loss, pre_order_logits, _ = self.event_centric_stage(pre_order_inputs, self.mlm)
            return pre_order_loss, None, None
    
        elif self.args.pretrain==False and self.training: #contrasitve fine-tuning
            pre_order_loss, pre_order_logits, pre_order_targets = self.fine_tuning_stage(pre_order_inputs, self.mlm)

            return pre_order_loss, pre_order_logits, pre_order_targets

        else:
            
            _, pre_order_logits, pre_order_targets = self.evaluate_stage(pre_order_inputs)

            return None, pre_order_logits, pre_order_targets


