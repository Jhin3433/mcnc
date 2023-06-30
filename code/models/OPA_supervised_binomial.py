import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

def tensor_shift(t, dim, shift):
    """
    t (tensor): tensor to be shifted. 
    dim (int): the dimension apply shift.
    shift (int): shift distance.
    """
    assert 0 <= shift <= t.size(dim), "shift distance should be smaller than or equal to the dim length."

    overflow = t.index_select(dim, torch.arange(t.size(dim)-shift, t.size(dim)).cuda())
    remain = t.index_select(dim, torch.arange(t.size(dim)-shift).cuda())

    return torch.cat((overflow, remain),dim=dim)

class SequenceInception(nn.Module):
    def __init__(self, in_dim,middle_dim,out_dim, normalized=1):
        super(SequenceInception, self).__init__()
        self.in_dim = in_dim
        self.middle_dim = middle_dim
        self.out_dim = out_dim
        self.normalized = normalized
        
        self.layer1 = nn.Linear(in_dim, middle_dim)
        self.relu =  nn.ReLU() #nn.Tanh()
        self.layer3 = nn.Linear(middle_dim, out_dim)
        self.layer2 = nn.Linear(middle_dim, middle_dim)

    def forward(self, input):
        batch_size = input.size(0)
        T = input.size(1)
        dim = input.size(2)
        input = input.view(batch_size*T,dim)
        out1 = self.layer1(input) # 为什么全0的emb经过线性层变成了有值
        out1 = self.relu(out1)
        out1 = self.layer2(out1)
        out1 = self.relu(out1)
        output = self.layer3(out1) # [batch_size, seq_len, dim]

        if self.normalized == 1:
            norm = output.norm(dim=1, p=2, keepdim=True) # [1440, 1]
            #if norm>0:
            output = output.div(norm.expand_as(output))
            output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
            output = torch.where(torch.isinf(output), torch.full_like(output, 0), output)

        output = output.view(batch_size,T,self.out_dim)
        return output

class PredictionInception(nn.Module):
    def __init__(self, in_dim,middle_dim,out_dim, normalized=1):
        super(PredictionInception, self).__init__()
        self.in_dim = in_dim
        self.middle_dim = middle_dim
        self.out_dim = out_dim
        self.normalized = normalized
        #inplace = True

        self.layer1 = nn.Linear(in_dim, middle_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(middle_dim, out_dim)
        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, input):
        batch_size = input.size(0)
        T = input.size(1)
        dim = input.size(2)
        input = input.view(-1,dim)
        out1 = self.layer1(input)
        out1 = self.relu(out1)
        output = self.layer2(out1)

        if self.normalized == 1:
            norm = output.norm(dim=1, p=2, keepdim=True)
            #if norm>0:
            output = output.div(norm.expand_as(output))
            output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
            output = torch.where(torch.isinf(output), torch.full_like(output, 0), output)

        output = output.view(batch_size,T,self.out_dim)
        return output


class OPA(nn.Module):
    def __init__(self, input_dim):
        super(OPA, self).__init__()
        #self.lam = lam
        self.input_dim = input_dim
        self.att_size = att_size = 30
        self.att_size2 = att_size2 = 100
        #self.att_size3 = att_size3 = 30
        self.scale = att_size ** -0.5
        self.scale2 = att_size2 ** -0.5


        self.middle_channel = 30
        self.conv1 = nn.Conv2d(2, self.middle_channel, 5, padding=2)
        self.conv2 = nn.Conv2d(self.middle_channel, self.middle_channel, 5, padding=2)
        self.conv3 = nn.Conv2d(self.middle_channel, 1, 3, padding=1)

    def forward(self, q_seq, q_len_seq, q_R, k_seq, k_len_seq, k_R):

        batch_size = q_seq.size(0)

        q_R = q_R.detach() 
        k_R = k_R.detach() 
        T_q = q_seq.size(1) # seq最长长度
        T_k = k_seq.size(1) # seq最长长度
        assert T_q == T_k
        
        # F.pairwise_distance(a,b,p=2)
        D = torch.cdist(q_seq.view(batch_size, T_q, self.input_dim), k_seq.view(batch_size, T_k, self.input_dim), 2) #[batch_size, T_q, T_k] # ground mertic，与对齐矩阵形状一致，每个元素值为欧氏距离           
        P = torch.cdist(q_R.view(batch_size, T_q, 1), k_R.view(batch_size, T_k, 1), 2) #[1, 20, 9] https://blog.csdn.net/mimiduck/article/details/128886148
        # 因为T_q T_k是最大长度，所以要对有效长度内的值进行normalization, 分别使用两个mask进行交叉相乘，获得有效元素矩阵
        row_mask = (torch.arange(T_q, device=q_seq.device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1) < q_len_seq).unsqueeze(-1).expand(batch_size,T_q,T_k)
        column_mask = (torch.arange(T_k, device=q_seq.device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1) < k_len_seq).unsqueeze(1).expand(batch_size,T_q,T_k)
        mask = row_mask * column_mask
        D = D * mask
        P = P * mask


        A = torch.cat([D.unsqueeze(1),P.unsqueeze(1)], 1) # [batch_size, 2, T_q, T_k]
        A = F.relu(self.conv1(A)) # [batch_size, 30, T_q, T_k]
        A = F.relu(self.conv2(A)) # [batch_size, 30, T_q, T_k]
        A = self.conv3(A).squeeze(1) + D # [batch_size, 1, T_q, T_k]   [batch_size, T_q, T_k]
        

        A[~mask] = 100
        A = torch.softmax(-A, dim=2) # 每row相加为1  

    
        At = A.view(batch_size,T_q*T_k,1) #[batch_size, T_q*T_k, 1]
        norm = At.norm(dim=1, p=1, keepdim=True) # 对T_q*T_k个数求1范数 https://zhuanlan.zhihu.com/p/260162240
        At = At.div(norm.expand_as(At)) 

        dis = torch.matmul(D.view(batch_size,1,T_q*T_k),At).squeeze() # 加入.squeeze()保证不是list
        return A, dis # [batch_size, T_q, T_k]  [batch_size]
class BlurGenerationPair(nn.Module):
    def __init__(self):
        super(BlurGenerationPair, self).__init__()


    def forward(self, seq, len_seq):
        import time
        start_time = time.time()

        batch_size = seq.size(0)
        T = seq.size(1)
        dim = seq.size(2)

        SeqtoBlur = torch.zeros(size=(batch_size,T,T), dtype=torch.float32, device=seq.device)
        avged_seq = torch.zeros_like(seq, device=seq.device)
        avged_len = torch.zeros_like(len_seq, device=seq.device)
    
        # 返回的R
        temp_position_index = torch.arange(1, T + 1, device=seq.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
        R = temp_position_index / len_seq
        # 用于去除pad位置
        # mask = torch.arange(T, device=seq.device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1) < len_seq
        # R = torch.masked_select(R, mask).reshape(batch_size, T)
        avged_R = torch.zeros_like(R, device=seq.device)
        
        # Condition 1: Tc > 4,seq[condition1_mask], avged_seq[condition1_mask]
        random_seed = torch.randint(1, 10, (batch_size, 1), device=seq.device)
        condition1_mask = ((len_seq > 4) & ( random_seed > 7)).squeeze() # torch.ones(seq.shape[0]) == 1 torch.ones(len_seq.shape[0]) == 1 or ((len_seq > 4) & ( random_seed > 7)).squeeze()
        if seq[condition1_mask].shape[0] != 0:
            temp_batch = seq[condition1_mask].shape[0] # 符合条件的batch数量
    
            midnum = torch.randn((temp_batch)).uniform_(0.55, 0.9)
            sidenum = (1.-midnum)/2.
            kernal_size = 3
            GaussKer = torch.zeros((temp_batch, kernal_size, 1), device=seq.device) #3是卷积长度
            GaussKer[:,0,:] = sidenum.unsqueeze(-1)
            GaussKer[:,2,:] = sidenum.unsqueeze(-1)
            GaussKer[:,1,:] = midnum.unsqueeze(-1)

            stride = random.randint(1,3)
            temp_T = int((T - kernal_size) / stride) + 1 # 卷积后序列的最大长度
            # seq[condition1_mask].unsqueeze(1) 是[batch,1,temp_len_seq,dim], GaussKer.unsqueeze(1)是[batch,1,kernal_size(3),kernal_size(1)], 这样卷积出来是对每个sample分别用了batch个卷积核，输出为[batch,batch,outTc,dim], 第二个batch是卷积核的个数
            # 参考https://zhuanlan.zhihu.com/p/251068800
            blured_seq = F.conv2d(seq[condition1_mask].unsqueeze(1),GaussKer.unsqueeze(1),stride=[stride,1]) # conv2d卷积后的seq
            blured_seq = blured_seq[torch.eye(blured_seq.shape[0])==1] #第一个sampe取第一个卷积结果，第二个sample取第二个卷积结果 参考：https://zhuanlan.zhihu.com/p/436020484
            avged_seq[condition1_mask] = F.pad(blured_seq,(0,0,0,seq.shape[1]-blured_seq.shape[1])) #(0,0,0,2)前两个0代表在最后一个维度上 前后两个方向都不填充，0,2代表在倒数第二个维度上前面不填充，在后面两个维度上填充2个单位  #https://blog.csdn.net/qq_34914551/article/details/102940377
            avged_len[condition1_mask] = ((len_seq[condition1_mask] - kernal_size) / stride).int() + 1 # 除去pad实际有效的长度

            # pad 参考https://blog.csdn.net/qq_34914551/article/details/102940377,#前两个0，0代表在最后一个维度上什么也不动，后面0，T-kernal_sized代表在倒数第二个维度上前面不补充而只在后面补充
            SeqtoBlur[condition1_mask] = F.pad( torch.cat( tuple(F.pad(GaussKer,(0,0,i*stride,T-kernal_size-i*stride)) for i in range(temp_T)) , dim = -1),  (0,seq.shape[1]-blured_seq.shape[1]))
            avged_R[condition1_mask] = temp_position_index[condition1_mask] / avged_len[condition1_mask] 
        


        condition2_mask = ((len_seq > 4) & (random_seed > 5) & (random_seed <= 7)).squeeze() # torch.ones(seq.shape[0]) == 1 
        if seq[condition2_mask].shape[0] != 0:
            temp_batch = seq[condition2_mask].shape[0] # 符合条件的batch数量
            midnum = torch.randn((temp_batch)).uniform_(0.33, 0.5)
            sidenum = (1.-midnum)/2.
            sidenum1 = torch.randn((temp_batch))
            for i in range(temp_batch):
                sidenum1[i].uniform_(0.16, sidenum[i]) 
            sidenum2 = sidenum - sidenum1
            kernal_size = 5
            GaussKer = torch.zeros((temp_batch, kernal_size, 1), device=seq.device) #4是卷积长度
            GaussKer[:,0,:] = sidenum2.unsqueeze(-1)
            GaussKer[:,1,:] = sidenum1.unsqueeze(-1)
            GaussKer[:,2,:] = midnum.unsqueeze(-1)
            GaussKer[:,3,:] = sidenum1.unsqueeze(-1)
            GaussKer[:,4,:] = sidenum2.unsqueeze(-1)
            stride = random.randint(1,3)
            temp_T = int((T - kernal_size) / stride) + 1 # 卷积后序列的最大长度
            blured_seq = F.conv2d(seq[condition2_mask].unsqueeze(1),GaussKer.unsqueeze(1),stride=[stride,1]) # conv2d卷积后的seq
            blured_seq = blured_seq[torch.eye(blured_seq.shape[0])==1] #第一个sampe取第一个卷积结果，第二个sample取第二个卷积结果 参考：https://zhuanlan.zhihu.com/p/436020484
            avged_seq[condition2_mask] = F.pad(blured_seq,(0,0,0,seq.shape[1]-blured_seq.shape[1])) #(0,0,0,2)前两个0代表在最后一个维度上 前后两个方向都不填充，0,2代表在倒数第二个维度上前面不填充，在后面两个维度上填充2个单位  #https://blog.csdn.net/qq_34914551/article/details/102940377
            avged_len[condition2_mask] = ((len_seq[condition2_mask] - kernal_size) / stride).int() + 1 # 除去pad实际有效的长度
            SeqtoBlur[condition2_mask] = F.pad( torch.cat( tuple(F.pad(GaussKer,(0,0,i*stride,T-kernal_size-i*stride)) for i in range(temp_T)) , dim = -1),  (0,seq.shape[1]-blured_seq.shape[1]))
            avged_R[condition2_mask] = temp_position_index[condition2_mask] / avged_len[condition2_mask] 

        condition3_mask = ((len_seq > 4) & ( random_seed <= 5)).squeeze()
        if seq[condition3_mask].shape[0] != 0:
            # len_seq_list = list(len_seq.cpu().numpy())
            # min_len = int(0.5*len_seq_list)
            # max_len = int(0.8*len_seq_list)
            # cur_len = [random.choice(range(min_len[i],max_len[i]+1)) for i in range(len_seq_list)]
            # interval = [random.sample(range(1,len_seq_list[i]),cur_len[i]-1) for i in range(len(cur_len))]
            # interval = [interval[i].sort().append(len_seq_list[i]) for i in range(len(interval))]
            # 对于每个序列分成相同的段，但是每个序列有效长度不一定是T，这个没有进行适配
            min_len = int(0.5 * T)
            max_len = int(0.8 * T)
            cur_len = random.choice(range(min_len, max_len+1)) #新长生的seq的part数
            end = random.sample(range(1,T), cur_len-1)
            end.sort()
            end.append(T)
            start = [0] + end[:-1]
            part_len = [e-s for e,s in zip(end,start)]

            # 这个mask用来确定生成的seq是由原来的seq哪一位置生成的
            mask = torch.arange(1, T+1, device=seq.device).unsqueeze(0).repeat(cur_len,1) 
            shift_pos = 0
            for i in range(1, cur_len):
                shift_pos = part_len[i-1] + shift_pos
                mask[i] = tensor_shift(mask[i], dim=0, shift=shift_pos)
            mask = mask <= torch.tensor(part_len, device=seq.device).unsqueeze(-1)

            # [4, 9]，四个节点，由原来的九个节点产生，每个节点是由哪几个原节点attention产生的
            localatt = torch.softmax(torch.where(mask, torch.tensor([[random.gauss(0,1) for _ in range(T)] for _ in range(cur_len)], device=seq.device), torch.tensor(-100, device=seq.device, dtype=torch.float32))   , dim=-1)
            
            # .view(max_templ,1),dim=0).cuda()
            temp_avged_seq = torch.sum(seq[condition3_mask].unsqueeze(1).expand(-1,cur_len,-1,-1) * localatt.unsqueeze(-1).expand(-1,-1,768).unsqueeze(0).expand(seq[condition3_mask].shape[0],-1,-1,-1), dim=-2)
            avged_seq[condition3_mask] = F.pad(temp_avged_seq, (0,0,0,T-cur_len))
            avged_len[condition3_mask] = torch.ones((seq[condition3_mask].shape[0], 1),device=seq.device,dtype=torch.int32) * cur_len
            SeqtoBlur[condition3_mask] = F.pad(localatt.transpose(0,1), (0,T-cur_len)).unsqueeze(0).repeat(seq[condition3_mask].shape[0],1,1)
            avged_R[condition3_mask] = temp_position_index[condition3_mask] / avged_len[condition3_mask] 


        # condition4_mask = (len_seq <= 2).squeeze()

        # 计算运行时间
        # end_time = time.time()
        # duration_time = end_time - start_time
        # print("execute time running %s seconds" % (duration_time))
        avged_R[avged_R > 1]=0
        R[R > 1]=0
        return SeqtoBlur.detach(), R.detach(), avged_seq.detach(), avged_len.detach(), avged_R.detach()





class BlurContrastiveModelPair(nn.Module):
    def __init__(self, input_dim, output_dim=-1, lam1=1., lam2=1.):
        super(BlurContrastiveModelPair, self).__init__()

        self.input_dim = input_dim
        if output_dim>0:
            self.output_dim = output_dim
        else:
            self.output_dim = input_dim
        self.middle_dim = 1024
        self.lsoftmax = nn.LogSoftmax(dim=1)

        self.alignment = OPA(self.output_dim)
        self.aug = BlurGenerationPair()
        self.encoder = SequenceInception(self.input_dim, self.middle_dim, self.output_dim)
        self.predictor = PredictionInception(self.input_dim, self.middle_dim, self.output_dim)
        #self.mse = nn.MSELoss(reduction='mean')
        self.lam1 = lam1
        self.lam2 = lam2
        self.mse = nn.MSELoss(reduction='mean')

        self.alpha = 40
        self.beta = 2
        self.hard_mining = True #None

    def forward(self, seq, len_seq, mode):
        # initialization
        margin=0.5
        m = 0.03

        batch_size = seq.size(0)
        if mode == "contrastive-fine-tuning":
            targets = torch.tensor([[1 if int(b_2/5) == int(b_1/5) else 0 for b_2 in range(batch_size)] for b_1 in range(batch_size)]).cuda() # [batch*choice_num, batch*choice_num] 只有属于同一个choice的作为正例
        else:
            targets = torch.eye(batch_size).cuda() #对角矩阵

        # Ldis = torch.zeros(batch_size, device=seq.device)
        # Ldis2 = torch.zeros(batch_size, device=seq.device)
        # DisMat = torch.zeros([batch_size,batch_size], dtype=torch.float, device=seq.device)

        # Sequence Augmentation
        SeqtoBlur, R, avged_seq, avged_len, avged_R = self.aug(seq, len_seq)

        seq = self.encoder(seq)
        avged_seq = self.encoder(avged_seq)
        Aa1b, disa1b = self.alignment(seq, len_seq, R, avged_seq, avged_len, avged_R)
        Aa2b, disa2b = self.alignment(avged_seq, avged_len, avged_R, seq, len_seq, R)


        DisMat = torch.diag_embed((disa1b + disa2b) / 2)
        Ldis = torch.mean(disa1b + disa2b) # 这里除以了batch_size,后面就不除了
        Ldis2 = self.mse(disa1b, disa2b) #0.1573
        Lalign2 = self.mse(Aa1b, Aa2b.transpose(1,2)) #sum:46 mean:0.0036
        Lalign = self.mse(Aa1b, SeqtoBlur) + self.mse(Aa2b, SeqtoBlur.transpose(1,2)) # sum:917.9  0.0708


        for b in range(1, batch_size):
            shift_seq = tensor_shift(seq, 0, b)
            shift_len_seq = tensor_shift(len_seq, 0, b)
            shift_R = tensor_shift(R, 0, b)

            As1b, dis1b = self.alignment(seq, len_seq, R, shift_seq, shift_len_seq, shift_R)
            As2b, dis2b = self.alignment(shift_seq, shift_len_seq, shift_R, seq, len_seq, R)


            DisMat.index_put_((torch.arange(batch_size,dtype=torch.int64,device=seq.device), tensor_shift(torch.arange(batch_size,dtype=torch.int64,device=seq.device), 0, b)), dis1b)
            DisMat.index_put_((tensor_shift(torch.arange(batch_size,dtype=torch.int64,device=seq.device), 0, b), torch.arange(batch_size,dtype=torch.int64,device=seq.device)), dis2b)

            Ldis2 = Ldis2 + self.mse(dis1b, dis2b)
            Lalign2 = Lalign2 + self.mse(As1b, As2b.transpose(1,2))


        # 根据targets决定哪些是pos_pair哪些是neg_pair
        pos_pair_ = DisMat * targets
        neg_pair_ = DisMat * (1 - targets)
        neg_pair_ = torch.where(neg_pair_==0, torch.tensor([100.0],device=seq.device), neg_pair_)

        max_pos_pair_ = torch.sort(pos_pair_, dim=-1)[0][:, -1]
        min_neg_pair = -(torch.sort(-neg_pair_, dim=-1)[0][:, -1])

        if self.hard_mining is not None and mode == "contrastive-fine-tuning":
            hard_pos_pair_mask = pos_pair_ + m > min_neg_pair
            hard_neg_pair_mask = neg_pair_ - m <  max_pos_pair_

            
            batch_index_mask = torch.sum(hard_neg_pair_mask,dim=-1) * torch.sum(hard_pos_pair_mask,dim=-1) != 0 # 每一个sample有多少个hard_neg_pair和多少个hard_pos_pair,如果没有其中之一，在该sample无效
            valid_index_num = torch.sum(batch_index_mask) #torch.count_nonzero(torch.sum(hard_neg_pair_mask,dim=-1) * torch.sum(hard_pos_pair_mask,dim=-1)) #如果某一个sample中没有neg_pair或者pos_pair，那么该sample无效
            

            pos_pair_num = torch.sum(hard_pos_pair_mask[batch_index_mask], dim=-1)
            neg_pair_num = torch.sum(hard_neg_pair_mask[batch_index_mask], dim=-1)

            pos_inner = torch.log(1 + torch.exp(self.beta*( (pos_pair_[batch_index_mask] - 0.5) ) ) ) #矩阵中为0的位置经过变化后有了值
            pos_loss = torch.sum( 2.0/self.beta * (torch.sum(pos_inner * hard_pos_pair_mask[batch_index_mask], dim=-1) / pos_pair_num) )
            neg_inner = torch.log(1 + torch.exp(-self.alpha*( (neg_pair_[batch_index_mask] - 0.5) ) ) )
            neg_loss = torch.sum( 2.0/self.alpha * torch.sum(neg_inner  * hard_neg_pair_mask[batch_index_mask], dim=-1) / neg_pair_num )  #内层的sum是用来将每一row的logits相加起来
            loss = pos_loss + neg_loss

            # Ldis /= batch_size
            # Lalign /= batch_size
            # Ldis2 /= batch_size
            # Lalign2 /= batch_size
            if valid_index_num != 0: 
                L = loss/valid_index_num + Ldis + Lalign #有c条数据没有计算loss
            else:
                L = Ldis/batch_size + Lalign/batch_size
            return L
        else: #"event-centric"
            pos_loss = torch.sum(    torch.sum(torch.log(1 + torch.exp(2*(pos_pair_ - 0.5))) * targets, dim=-1) / torch.sum(targets, dim=-1)    ) #先求某一行的和除以有效num，再对batch求和
            neg_loss = torch.sum(    torch.sum(torch.log(1 + torch.exp(-self.alpha*(neg_pair_ - 0.5) * (1-targets))), dim=-1) / torch.sum((1-targets), dim=-1)   )
            loss = pos_loss + neg_loss
            L = loss/batch_size + Ldis + Lalign
            return L

    def getlen(self, seq, len_seq):
        batch_size = seq.size(0)
        T = seq.size(1)
        dim = seq.size(2)
        R = torch.zeros(batch_size,T).cuda()
        for b in range(batch_size):
            Tc = len_seq[b].item()
            R[b,0:Tc] = torch.tensor([float(t+1)/float(Tc) for t in range(Tc)])
        return R


def KNN(X, k):
    X = X.float()
    mat_square = torch.mm(mat, mat.t())
    diag = torch.diagonal(mat_square)
    diag = diag.expand_as(mat_square)
    dist_mat = diag + diag.t() - 2*mat_square
    dist_col = dist_mat[-1, :-1]
    val, index = dist_col.topk(k, largest=False, sorted=True)
    return val, index

if __name__ == "__main__":
    seq = torch.tensor([[[1.,2.,3.,4.,5.,6],[7.,8.,9.,10.,11.,12.],[13.,14.,15.,16.,17.,18.]],[[11.,12.,13.,41.,52.,0.],[72.,81.,93.,8.,1.,0.],[3.,4.,6.,8.,12.,0.]]])
    seq = seq.transpose(1,2).contiguous()
    #print(seq)
    aug = BlurGeneration()
    len_seq = torch.tensor([[6],[5]])
    blured_seq, SeqtoBlur, avged_seq, R, blured_R, avged_R, blured_len, avged_len  = aug(seq,len_seq)
    print(blured_len)
    print(avged_len)
    
    bcm = BlurContrastiveModel(3, output_dim=3)
    As1, dis1, As2, dis2, Ab1, disb1, Ab2, disb2, Aa1, disa1, Aa2, disa2 = bcm(seq, len_seq)

    bcl = BlurContrastiveLoss()
    l = bcl(As1, dis1, As2, dis2, Ab1, disb1, Ab2, disb2, Aa1, disa1, Aa2, disa2)
    print(l)
