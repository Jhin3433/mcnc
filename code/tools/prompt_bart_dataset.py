from base64 import encode
from torch.utils.data import Dataset
from transformers import BartTokenizerFast
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
        # special_tokens = ["<sep>"] # keep consistent with "bart_custom.py" <sep>50265 <mask>50264
        if args.debug:
            self.raw_data = raw_data
        else:
            self.raw_data = raw_data
        self.args = args
        self.tokenizer = BartTokenizerFast.from_pretrained(args.pretrained_model_path, return_offsets_mapping=True) 
        # self.tokenizer.add_tokens(special_tokens) 
        self.event2str = negg_event2str
        
        self.state = state


        self.prompt = {"event-centric":["Given the incomplete but ordered event sequence: ", "The masked events are:"],
                       "recover-order":["Given the complete but disordered event sequence: ", "The ordered events are:"],
                       "contrastive-fine-tuning":["Given the incomplete but ordered event sequence: ", "The masked events are:"],
                       
                       }


    def __len__(self):
        return len(self.raw_data)
    
    def event_centric_task(self, index):
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

        encode_input = ''
        for i in range(9):
            if i in mask_indexs:
                encode_input += '<mask>, '
            else:
                encode_input +=  raw_event_list[i] + ', '
        encode_input = encode_input.strip(", ") #去掉结尾的空格和逗号

        encode_input = self.prompt[self.args.stage_mode][0] + encode_input + ". " + self.prompt[self.args.stage_mode][1] #加入prompt


        decode_input = ''
        for i in mask_indexs:
            if i != -100:
                decode_input += raw_event_list[i] + ', '
        decode_input = decode_input.strip(", ") #去掉结尾的空格和逗号


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

        example_original = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,None,target]
        example_original = [torch.tensor(t,dtype=torch.int32) for t in example_original if t != None]
        
        # --------------------------------------------- format of orginial event-centric ---------------------------------------------
        return example_original
    
    
    def mask_and_pred_final_event_task(self, index):

        return
    

    def shuffle_and_order_task(self, index):
        # --------------------------------------------- format of pre_order with shuffle events ---------------------------------------------
        if len(self.raw_data[index]) == 4:
            context,sent,answers,target = self.raw_data[index]
        else:
            context,answers,target = self.raw_data[index]
        encode_inputs = []
        encode_masks = []
        decode_inputs = []
        decode_masks = []
        labels  = []
        

        raw_event_list = []
        for event in context:
            event_repr = self.event2str(event)
            raw_event_list.append(event_repr[1:])
        # raw_event_list.append(self.event2str(answers[target])[1:])
        # raw_tokens_list = [self.tokenizer.tokenize(event) for event in raw_event_list]


        #Only shuffle context events
        for i in range(5):
            events_shuffle_position = [i for i in range(len(raw_event_list))]
            random.shuffle(events_shuffle_position)
            events_raw_position = [i for i in range(len(raw_event_list))]

            encode_input = ''
            for j in events_shuffle_position:
                encode_input +=  raw_event_list[j] + ', '

            encode_input = encode_input + self.event2str(answers[i])[1:] + ","
            encode_input = encode_input.strip(",")

            encode_input = self.prompt[self.args.stage_mode][0] + encode_input + ". " + self.prompt[self.args.stage_mode][1] #加入prompt

            decode_input = ''
            for k in events_raw_position:
                decode_input +=  raw_event_list[k] + ", "
            decode_input = decode_input + self.event2str(answers[i])[1:] + ","
            decode_input = decode_input.strip(",")

            encode_inputs.append(encode_input)
            decode_inputs.append(decode_input)

        encode_input_tokenized = self.tokenizer(encode_inputs,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    padding="max_length",
                    truncation=True,
                    max_length=self.args.encode_max_length)




        decode_input_tokenized = self.tokenizer(decode_inputs,
                                add_special_tokens=True,
                                return_token_type_ids=False,
                                padding="max_length",
                                truncation=True,
                                max_length=self.args.decode_max_length)




        labels = copy.deepcopy(decode_input_tokenized['input_ids'])
        example = [encode_input_tokenized['input_ids'],encode_input_tokenized['attention_mask'],decode_input_tokenized['input_ids'],decode_input_tokenized['attention_mask'],labels,target]
        example = [torch.tensor(t,dtype=torch.int32) for t in example]
        # --------------------------------------------- format of pre_order with shuffle events ---------------------------------------------


        return example
    def __getitem__(self, index):
        if self.state == 'train' and self.args.stage_mode == "event-centric": 
            example = self.event_centric_task(index)
            return example


        elif self.state == 'eval' and self.args.stage_mode == "event-centric":
            example = self.mask_and_pred_final_event_task(index)
        
        elif self.args.stage_mode == "contrastive-fine-tuning":

            example = self.mask_and_pred_final_event_task(index)

        elif self.args.stage_mode == "order_task": #contrastive fine-tuning
            example = self.shuffle_and_order_task(self, index)
        

        return example