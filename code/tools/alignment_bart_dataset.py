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

def event_index_span(input_ids_list, mode=None, max_length = 100):
    """
        max_length : token序列的最大长度
    """
    def process(input_ids):
        mask_list = [[[0 for _ in range(768)] for _ in range(max_length)] for _ in range(9)] # 九个event，对应9个mask_list，每一个mask_list产生一个event的embedding


        temp_index = [x for x, y in enumerate(input_ids) if y == 479] # 逗号分隔y==6 中间的事件, y==2是<s>中止,y=50265是<sep>分割，y=479是.分割

        # event_span = []
        assert len(temp_index) == 9

        for i_1, i in enumerate(temp_index):
            if i_1 == 0:
                # event_span.append([1,i]) #不包含i，第i个位置是分隔符
                mask_list[i_1][1:i] = [[1 for _ in range(768)] for _ in range(i-1)]
                next_start = i + 1
            else:
                # event_span.append([next_start,i])
                mask_list[i_1][next_start:i] = [[1 for _ in range(768)] for _ in range(i-next_start)]
                next_start = i + 1
                
        # assert len(event_span) == 9 # 九个事件
        return mask_list 

    if mode == "event-centric":
        mask_list = []
        for input_ids in input_ids_list:
            mask = process(input_ids)
            mask_list.append([mask])
        return mask_list
    else:
        mask_list = []
        for input_ids in input_ids_list[0:-1:5]:
            mask = process(input_ids)
            mask_list.append([mask for _ in range(5)])  
        return mask_list # 此时输入是8个已知event + <mask>event，复制choice_num遍   

 

def custom_collate_fn(batch, stage_mode, tokenizer):

        import copy
        if stage_mode != "event-centric":
            encode_inputs = []
            decode_inputs = []
            target = []
            for b in batch:
                for index in range(5):
                    encode_inputs.append(b[0])
                    decode_inputs.append(b[1][index])
                target.append(b[2])
        else:
            encode_inputs = []
            decode_inputs = []
            target = []
            for b in batch:
                encode_inputs.append(b[0])
                decode_inputs.append(b[1])
                target.append(b[2])
        encode_input_tokenized = tokenizer(encode_inputs, add_special_tokens=True, return_token_type_ids=False, padding = True)
        decode_input_tokenized = tokenizer(decode_inputs, add_special_tokens=True, return_token_type_ids=False, padding = True)
        encode_inputs = encode_input_tokenized['input_ids']
        encode_masks = encode_input_tokenized['attention_mask']
        decode_inputs = decode_input_tokenized['input_ids']
        decode_masks = decode_input_tokenized['attention_mask']
        labels = copy.deepcopy(decode_input_tokenized['input_ids'])


        event_spans = event_index_span(encode_inputs, mode=stage_mode, max_length=len(encode_inputs[0]))
        
        example = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,event_spans,target]
        example = [torch.tensor(t,dtype=torch.int32) for t in example]


        return example
class bart_dataset(Dataset):
    def __init__(self, raw_data, args, state):
        # special_tokens = ["<sep>"] # keep consistent with "bart_custom.py" <sep>50265 <mask>50264
        if args.debug:
            self.raw_data = raw_data
        else:
            self.raw_data = raw_data
        self.args = args
        # self.tokenizer = BartTokenizerFast.from_pretrained(args.pretrained_model_path, return_offsets_mapping=True) 
        # self.tokenizer.add_tokens(special_tokens) 
        self.event2str = negg_event2str
        
        self.state = state
 
        self.sep_token = "."

    def __len__(self):
        return len(self.raw_data)
    

    def event_centric_task(self, index):
        if self.state == 'train' and self.args.stage_mode == "event-centric": 

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
            #         encode_input += '<mask> <sep> '
            #     else:
            #         encode_input +=  raw_event_list[i] + ' <sep> '
            # encode_input = encode_input.strip(" <sep> ") #去掉结尾的空格和分隔符
            encode_input = ''
            for i in range(9):
                if i in mask_indexs:
                    encode_input += '<mask> {} '.format(self.sep_token)
                else:
                    encode_input +=  raw_event_list[i].replace(".", "") + ' {} '.format(self.sep_token)
            encode_input = encode_input[:-1] #去掉结尾的空格
  
            # decode_input = ''
            # for i in mask_indexs:
            #     if i != -100:
            #         decode_input += raw_event_list[i] + ' <sep> '
            # decode_input = decode_input.strip(" <sep> ") #去掉结尾的空格和分隔符
            decode_input = '. '
            for i in mask_indexs:
                if i != -100:
                    decode_input += raw_event_list[i].replace(".", "") + ' {} '.format(self.sep_token)
            decode_input = decode_input[:-1] #去掉结尾的空格

                     
            return (encode_input, decode_input, target)
    
    def mask_and_pred_final_event_task_old(self, index):
        if len(self.raw_data[index]) == 4:
            context,sent,answers,target = self.raw_data[index]
        else:
            context,answers,target = self.raw_data[index]
        encode_inputs = []
        encode_masks = []
        decode_inputs = []
        decode_masks = []
        labels  = []
        
        # --------------- --------------- --------------- --------------- --------------- #
        raw_event_list = []
        for event in context:
            event_repr = self.event2str(event)
            raw_event_list.append(event_repr[1:])
        # raw_event_list.append(self.event2str(answers[target])[1:])
        # raw_tokens_list = [self.tokenizer.tokenize(event) for event in raw_event_list]

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
        # --------------- --------------- --------------- --------------- --------------- #


        
        for answer in answers:

            ## decode恢复mask掉的
            decode_input = self.event2str(answer)[1:]
            
            # decode_input = self.event2str(answer)[1:]
            decode_input_tokenized = self.tokenizer(decode_input,
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.args.eval_decode_max_length)
            decode_inputs.append(decode_input_tokenized['input_ids'])
            decode_masks.append(decode_input_tokenized['attention_mask'])
        labels = copy.deepcopy(decode_inputs)

        encode_event_spans = self.event_index_span(encode_input_tokenized['input_ids'], mode="contrastive-fine-tuning") #5个相同
        example = [encode_inputs,encode_masks,decode_inputs,decode_masks,labels,encode_event_spans,target]
        example = [torch.tensor(t,dtype=torch.int32) for t in example]
        return example

    def mask_and_pred_final_event_task(self, index):
        
        if len(self.raw_data[index]) == 4:
            context,sent,answers,target = self.raw_data[index]
        else:
            context,answers,target = self.raw_data[index]

        
        # --------------- --------------- --------------- --------------- --------------- #
        raw_event_list = []
        for event in context:
            event_repr = self.event2str(event)
            raw_event_list.append(event_repr[1:])
        # raw_event_list.append(self.event2str(answers[target])[1:])
        # raw_tokens_list = [self.tokenizer.tokenize(event) for event in raw_event_list]

        encode_input = ''
        for i in range(9):
            if i==8:
                encode_input += '<mask> {}'.format(self.sep_token)
            else:
                encode_input +=  raw_event_list[i].replace(".", "") + ' {} '.format(self.sep_token)
        decode_input_list = []
        for answer in answers:
            decode_input = "{} ".format(self.sep_token) + self.event2str(answer)[1:].replace(".", "") + " {}".format(self.sep_token)
            if decode_input == '. city categorize links .':
                decode_input = '. city city city .'
            ## decode恢复mask掉的
            decode_input_list.append(decode_input)
            
        return (encode_input, decode_input_list, target)


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

            decode_input = ''
            for k in events_raw_position:
                decode_input +=  raw_event_list[k] + ", "
            decode_input = decode_input + self.event2str(answers[i])[1:] + ","


            encode_inputs.append(encode_input.strip(","))
            decode_inputs.append(decode_input.strip(","))

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

        encode_event_spans = self.event_index_span(encode_input_tokenized['input_ids'])
        # decode_event_spans = self.event_index_span(decode_input_tokenized['input_ids'])


        labels = copy.deepcopy(decode_input_tokenized['input_ids'])
        example = [encode_input_tokenized['input_ids'],encode_input_tokenized['attention_mask'],decode_input_tokenized['input_ids'],decode_input_tokenized['attention_mask'],labels,encode_event_spans,target]
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

        elif self.args.stage_mode == "evaluate_model":
            example = self.mask_and_pred_final_event_task(index)

        elif self.args.stage_mode == "order_task": #contrastive fine-tuning
            example = self.shuffle_and_order_task(self, index)

        return example
        