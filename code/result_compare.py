import pickle
import torch
file = open("03_10_10_20_19.pickle", "rb")
a = pickle.load(file) #pred in 03_10_10_20_19.pickle
b = pickle.load(file)

file2 = open("02_16_15_35_02.pickle", "rb")
c = pickle.load(file2) #pred in 02_16_15_35_02.pickle
d = pickle.load(file2)



print(torch.sum(torch.tensor(a)==torch.tensor(c)).item()/ len(a))

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
