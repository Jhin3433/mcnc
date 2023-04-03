# mcnc
## 1. model architecture
Share the same encoder but different decoder. Sum the respective logits by the two decoders as the final logits.
## 2. code structure
Refactor the code of main.py. Construct specific functions for each initialization.
In bart_dataset_random.py, add the code of recovering all the events instead of only <mask> event.



