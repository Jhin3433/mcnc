import pickle
import torch
import numpy as np
file = open("03_28_22_45_00.pickle", "rb") # 
a = pickle.load(file) #pred logits 
b = pickle.load(file) # [1 if pred == label else 0 for pred,label in zip(b,c)].count(1)
c = pickle.load(file)


file2 = open("addDistance.pickle", "rb") # eval_config.py set pre_order == False
d = pickle.load(file2) #pred logits 
e = pickle.load(file2)
f = pickle.load(file2)


assert c == f


lmd_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5]
for lmd in lmd_list:
    final_logits = []
    final_labels = []
    for pre_order_logits, original_logits in zip(a,d):
        final_logits.append(pre_order_logits + lmd * original_logits)

    for i in range(len(final_logits)):
        final_labels.append(torch.max(torch.tensor(final_logits[i]).unsqueeze(dim=0), dim=1)[1].item())

    print("lmd:{}, acc:{}".format(lmd, torch.sum(torch.tensor(final_labels)==torch.tensor(c)).item()/ len(final_labels)))





print(torch.sum(torch.tensor(a)==torch.tensor(c)).item()/ len(a))



new = []
for x, y, z in zip(b,e,c):
    if x == z:
        new.append(x)
    elif y == z:
        new.append(y)
    else:
        # print(x, y)
        new.append(y)
assert len(new) == len(b)
print(torch.sum(torch.tensor(new)==torch.tensor(c)).item()/ len(a))
