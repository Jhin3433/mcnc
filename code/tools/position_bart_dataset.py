from base64 import encode
from torch.utils.data import Dataset
from transformers import BartTokenizer
import random
import copy
import torch
import logging
logger = logging.getLogger()

def negg_event2str(event):
    def mystr(s):
        return '' if s is None else str(' '+s)
    return mystr(event[3]) + mystr(event[0].replace('+',' ')) + mystr(event[4]) + mystr(event[5])

class bart_dataset(Dataset):
    def __init__(self, raw_data, args, state):
        special_tokens = ["<sep>"] # keep consistent with "bart_custom.py" <sep>50265 <mask>50264
        if args.debug:
            self.raw_data = raw_data
        else:
            self.raw_data = raw_data
        self.args = args
        self.tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_path) 
        self.tokenizer.add_tokens(special_tokens) 
        self.event2str = negg_event2str
        
        self.state = state

    def __len__(self):
        return len(self.raw_data)
    


    def __getitem__(self, index):
        if self.state == 'train' and self.args.pretrain: 

            if len(self.raw_data[index]) == 4: 
                context, sent, answers, target = self.raw_data[index]
            else:
                context,answers, target = self.raw_data[index]
            raw_event_list = []
            for event in context:
                event_repr = self.event2str(event)
                raw_event_list.append(event_repr[1:])
            raw_event_list.append(self.event2str(answers[target])[1:])
    

    


            # --------------------------------------------- format of orginial event-centric ---------------------------------------------
            mask_num = random.randint(1,self.args.mask_num)
            mask_indexs = random.sample(range(0, 9),mask_num)
            list.sort(mask_indexs) # the indexes of masked events [1, 3, 4]
            while len(mask_indexs) != self.args.mask_num:
                mask_indexs.append(-100)

            # encode_input = ''
            # for i in range(9):
            #     if i in mask_indexs:
            #         encode_input += '<mask> . '
            #     else:
            #         encode_input +=  raw_event_list[i] + ' . '
            # decode_input = '. '
            # for i in mask_indexs:
            #     decode_input += raw_event_list[i] + ' . '
            # decode_input = decode_input[:-1]


            encode_input = ''
            for i in range(9):
                if i in mask_indexs and i != 8:
                    encode_input += '<mask> <sep> '
                elif i in mask_indexs and i == 8:
                    encode_input += '<mask>'
                elif i not in mask_indexs and i != 8 :
                    encode_input +=  raw_event_list[i] + ' <sep> '
                else:
                    encode_input +=  raw_event_list[i]

            decode_input = ''
            # # decode恢复所有的
            # for i in range(9):
            #     decode_input +=  raw_event_list[i] + ' <sep> '
            # decode恢复mask掉的   
            for i in mask_indexs:
                if i != -100:
                    decode_input += raw_event_list[i] + ' <sep> '
            decode_input = decode_input[:-1]


            
            encode_input_tokenized = self.tokenizer(encode_input,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.args.encode_max_length)
            

            decode_input_tokenized = self.tokenizer(decode_input,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.args.decode_max_length)
            encode_inputs = encode_input_tokenized['input_ids']
            encode_masks = encode_input_tokenized['attention_mask']
            decode_inputs = decode_input_tokenized['input_ids']
            decode_masks = decode_input_tokenized['attention_mask']
            labels = copy.deepcopy(decode_input_tokenized['input_ids'])
            example_original = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,mask_indexs,target]
            example_original = [torch.tensor(t,dtype=torch.int32) for t in example_original]
         
            # --------------------------------------------- format of orginial event-centric ---------------------------------------------
            
            


            
            
            return example_original







        else: #contrastive fine-tuning

            # --------------------------------------------- format of pre_order ---------------------------------------------
            if len(self.raw_data[index]) == 4:
                context,sent,answers,target = self.raw_data[index]
            else:
                context,answers,target = self.raw_data[index]

            # --------------------------------------------- format of orginial contrastive learning ---------------------------------------------
            encode_inputs = []
            encode_masks = []
            decode_inputs = []
            decode_masks = []
            labels  = []
            
            raw_event_list = []
            for event in context:
                event_repr = self.event2str(event)
                raw_event_list.append(event_repr[1:])

            encode_input = ''
            for i in range(9):
                if i==8:
                    encode_input += '<mask>'
                else:
                    encode_input +=  raw_event_list[i] + ' <sep> '
            encode_input_tokenized = self.tokenizer(encode_input,
                        add_special_tokens=True,
                        return_token_type_ids=False,
                        padding="max_length",
                        truncation=True,
                        max_length=self.args.encode_max_length)
            for i in range(5):
                encode_inputs.append(encode_input_tokenized['input_ids'])
                encode_masks.append(encode_input_tokenized['attention_mask'])


            for answer in answers:
                decode_input = self.event2str(answer)[1:]
                decode_input_tokenized = self.tokenizer(decode_input,
                                        add_special_tokens=True,
                                        return_token_type_ids=False,
                                        padding="max_length",
                                        truncation=True,
                                        max_length=self.args.eval_decode_max_length)
                decode_inputs.append(decode_input_tokenized['input_ids'])
                decode_masks.append(decode_input_tokenized['attention_mask'])
            labels = copy.deepcopy(decode_inputs)
            mask_indexs = [[8],[8],[8],[8],[8]]
            example_original = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,mask_indexs, target]
            example_original = [torch.tensor(t,dtype=torch.int32) for t in example_original]



            return example_original