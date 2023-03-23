# mcnc
## 1. model architecture
Share the same encoder but different decoder. Sum the respective logits by the two decoders as the final logits.
## 2. code structure
Add distributed training. Abandon the acquire method of parameters from config.yaml.
The parameters are stored in a dictionary. Argpaser read default values from the dictionary. 




