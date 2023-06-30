# Commit on June 2 
+ positon_bart.py, my_modeling_bart.py, position_bart_dataset.py 对position embedding进行了适配，效果不好，模型图见Fig1.jpg
+ 04_21_15_35_26.pickle 是positon_bart的结果,在event embedding中加入postion embedding
+ 03_28_22_45_00.pickle 是原始模型的结果，比02_16_15_35_02_original_results.pickle使用了分布式并加大了batch_size
+ 03_10_10_20_19_pre_order_results.pickle 是打乱顺序排序的结果


# Commit on June 30
+ 修改了main_align.py，将对齐距离计算代码使用矩阵运算的方式进行了重构，main_prompt.py相关代码后续可删除
+ 后续工作：可将encoder首位的embedding看作sequence embedding和candidate events’ embedding 进行计算；探索sequence augmentation方法；[encoder和decoder输出进行KL散度计算](https://arxiv.org/pdf/2306.06919.pdf)