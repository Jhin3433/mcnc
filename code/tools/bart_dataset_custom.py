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
        special_tokens = ["<sep>"] # keep consistent with "bart_custom.py"
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
    
    def text_span(self, input_ids):
        text_spans = []
        start = 1
        for index, input_id in enumerate(input_ids):
            if input_id == 0:
                continue
            if input_id == 2:
                text_spans.append([start, index - 1])
                break
            if input_id == 4:
                text_spans.append([start, index - 1])
                start = index + 1
        return text_spans   

    def __getitem__(self, index):
        if self.state == 'train' and self.args.pretrain: 
            # --------------------------------------------- format of pre_order ---------------------------------------------
            # read nine events (eight event + right candidate event.)
            if len(self.raw_data[index]) == 4: 
                context, sent, answers, target = self.raw_data[index]
            else:
                context,answers, target = self.raw_data[index]
            raw_event_list = []
            for event in context:
                event_repr = self.event2str(event)
                raw_event_list.append(event_repr[1:])
            raw_event_list.append(self.event2str(answers[target])[1:])

            # examples = [[] for i in range(6)] # add the num of shuffle sequences.
            # for i in range(10): # add the num of shuffle sequences.
            events_shuffle_position = [i for i in range(len(raw_event_list))]
            random.shuffle(events_shuffle_position)
            
            # logger.info("events_shuffle_position is {}".format(events_shuffle_position))
            events_raw_position = [i for i in range(len(raw_event_list))]
            encode_input = ''
            for index, i in enumerate(events_shuffle_position):
                if index == len(events_shuffle_position) - 1 :
                    encode_input +=  raw_event_list[i]
                else:
                    encode_input +=  raw_event_list[i] + ' <sep> '

            decode_input = ''
            for index, i in enumerate(events_raw_position):
                if index == len(events_raw_position) - 1:
                    decode_input +=  raw_event_list[i]
                else:
                    decode_input +=  raw_event_list[i] + ' <sep> '


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
            labels = copy.deepcopy(decode_inputs)


            example = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,target]
            example = [torch.tensor(t,dtype=torch.int32) for t in example]
            # --------------------------------------------- format of pre_order ---------------------------------------------

    

    


            # --------------------------------------------- format of orginial event-centric ---------------------------------------------
            mask_num = random.randint(1,self.args.mask_num)
            mask_indexs = random.sample(range(0, 9),mask_num)
            list.sort(mask_indexs) # the indexes of masked events [1, 3, 4]


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
            # decode恢复所有的
            for i in range(9):
                decode_input +=  raw_event_list[i] + ' <sep> '
            # # decode恢复mask掉的   
            # for i in mask_indexs:
            #     decode_input += raw_event_list[i] + ' <sep> '
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
            example_original = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,target]
            example_original = [torch.tensor(t,dtype=torch.int32) for t in example_original]
         
            # --------------------------------------------- format of orginial event-centric ---------------------------------------------
            
            # --------------------------------------------- format of normal order ---------------------------------------------
            events_raw_position = [i for i in range(len(raw_event_list))]
            encode_input_2 = ''
            for index, i in enumerate(events_raw_position):
                if index == len(events_raw_position) - 1 :
                    encode_input_2 +=  raw_event_list[i]
                else:
                    encode_input_2 +=  raw_event_list[i] + ' <sep> '

            decode_input_2 = ''
            for index, i in enumerate(events_raw_position):
                if index == len(events_raw_position) - 1:
                    decode_input_2 +=  raw_event_list[i]
                else:
                    decode_input_2 +=  raw_event_list[i] + ' <sep> '
            encode_input_2_tokenized = self.tokenizer(encode_input_2,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.args.encode_max_length)
            decode_input_2_tokenized = self.tokenizer(decode_input_2,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.args.decode_max_length)
            encode_inputs_2 = encode_input_2_tokenized['input_ids']
            encode_masks_2 = encode_input_2_tokenized['attention_mask'] 
            decode_inputs_2 = decode_input_2_tokenized['input_ids']
            decode_masks_2 = decode_input_2_tokenized['attention_mask']
            labels_2 = copy.deepcopy(decode_inputs_2)
            example_normal = [encode_inputs_2,encode_masks_2,decode_inputs_2,decode_masks_2,labels_2,target]
            example_normal = [torch.tensor(t,dtype=torch.int32) for t in example_normal]
            # --------------------------------------------- format of normal order --------------------------------------------- 
            
            

            
            
            return example, example_original, example_normal







        else: #contrastive fine-tuning

            # --------------------------------------------- format of pre_order ---------------------------------------------
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

            for i in range(5):
                temp_raw_event_list = raw_event_list[:]
                temp_raw_event_list.append(self.event2str(answers[i])[1:])
                
                events_shuffle_position = [i for i in range(len(temp_raw_event_list))]
                random.shuffle(events_shuffle_position)
                events_raw_position = [i for i in range(len(temp_raw_event_list))]

                encode_input = ''
                for index, j in enumerate(events_shuffle_position):
                    if index == len(events_shuffle_position) - 1 :
                        encode_input +=  temp_raw_event_list[j]
                    else:
                        encode_input +=  temp_raw_event_list[j] + ' <sep> '

                
                decode_input = ''
                for index, k in enumerate(events_raw_position):
                    if index == len(events_raw_position) - 1:
                        decode_input +=  temp_raw_event_list[k]
                    else:
                        decode_input +=  temp_raw_event_list[k] + ' <sep> '


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
            # --------------------------------------------- format of pre_order ---------------------------------------------


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
            example_original = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,target]
            example_original = [torch.tensor(t,dtype=torch.int32) for t in example_original]



            # --------------------------------------------- format of normal order ---------------------------------------------
            events_raw_position = [i for i in range(len(raw_event_list))]
            encode_input_2 = ''
            for index, i in enumerate(events_raw_position):
                if index == len(events_raw_position) - 1 :
                    encode_input_2 +=  raw_event_list[i]
                else:
                    encode_input_2 +=  raw_event_list[i] + ' <sep> '

            decode_input_2 = ''
            for index, i in enumerate(events_raw_position):
                if index == len(events_raw_position) - 1:
                    decode_input_2 +=  raw_event_list[i]
                else:
                    decode_input_2 +=  raw_event_list[i] + ' <sep> '
            encode_input_2_tokenized = self.tokenizer(encode_input_2,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.args.encode_max_length)
            decode_input_2_tokenized = self.tokenizer(decode_input_2,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.args.decode_max_length)
            encode_inputs_2 = encode_input_2_tokenized['input_ids']
            encode_masks_2 = encode_input_2_tokenized['attention_mask'] 
            decode_inputs_2 = decode_input_2_tokenized['input_ids']
            decode_masks_2 = decode_input_2_tokenized['attention_mask']
            labels_2 = copy.deepcopy(decode_inputs_2)
            example_normal = [encode_inputs_2,encode_masks_2,decode_inputs_2,decode_masks_2,labels_2,target]
            example_normal = [torch.tensor(t,dtype=torch.int32) for t in example_normal]
            # --------------------------------------------- format of normal order --------------------------------------------- 


            return example, example_original, example_normal