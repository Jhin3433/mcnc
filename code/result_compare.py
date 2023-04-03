import pickle
import torch
import numpy as np
file = open("03_10_10_20_19_pre_order_results.pickle", "rb")
a = pickle.load(file) #pred logits in 03_10_10_20_19.pickle
b = pickle.load(file)
c = pickle.load(file)


file2 = open("02_16_15_35_02_original_results.pickle", "rb")
d = pickle.load(file2) #pred logits in 02_16_15_35_02.pickle
e = pickle.load(file2)
f = pickle.load(file2)


assert c == f



lmd_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5]

for lmd in lmd_list:
    final_logits = []
    final_labels = []
    for pre_order_logits, original_logits in zip(a,d):
        final_logits.append(lmd * pre_order_logits + original_logits)

    for i in range(len(final_logits)):
        final_labels.append(torch.max(torch.tensor(final_logits[i]).unsqueeze(dim=0), dim=1)[1].item())

    print("lmd:{}, acc:{}".format(lmd, torch.sum(torch.tensor(final_labels)==torch.tensor(c)).item()/ len(final_labels)))


# print(torch.sum(torch.tensor(a)==torch.tensor(c)).item()/ len(a))

# print(torch.sum(torch.tensor(a)==torch.tensor(d)).item()/ len(a))

# new = []
# for x, y, z in zip(a,c,b):
#     if x == z:
#         new.append(x)
#     elif y == z:
#         new.append(y)
#     else:
#         print(x, y)
#         new.append(y)
# assert len(new) == len(b)
# print(torch.sum(torch.tensor(new)==torch.tensor(b)).item()/ len(a))
