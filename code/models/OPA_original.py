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

        self.conv1 = nn.Conv2d(2, 30, 5, padding=2)
        self.conv2 = nn.Conv2d(30, 30, 5, padding=2)
        self.conv3 = nn.Conv2d(30, 1, 3, padding=1)

    def forward(self, seq_q, seq_k, len_q, len_k):

        batch_size = seq_q.size(0)
        len_q = len_q.detach() 
        len_k = len_k.detach() 
        T_q = seq_q.size(1) # 20
        T_k = seq_k.size(1) # 9

        

        D = torch.cdist(seq_q.view(batch_size, T_q, self.input_dim), seq_k.view(batch_size, T_k, self.input_dim), 2) # ground mertic，与对齐矩阵形状一致，每个元素值为欧氏距离           
        P = torch.cdist(len_q.view(batch_size,T_q,1), len_k.view(batch_size,T_k,1), 2) #[1, 20, 9] https://blog.csdn.net/mimiduck/article/details/128886148

        A = torch.cat([D, P], 0).unsqueeze(0) # [1, 2, 20, 9]
        A = F.relu(self.conv1(A)) # [1, 30, 20, 9]
        A = F.relu(self.conv2(A)) # [1, 30, 20, 9]
        A = self.conv3(A).squeeze(1) + D # [1, 20, 9] + [1 ,20, 9]
        A = torch.softmax(-A, dim=2) # 每row相加为1

        At = A.view(batch_size,T_q*T_k,1) #[1, 180, 1]
        norm = At.norm(dim=1, p=1, keepdim=True) # 对180个数求1范数 https://zhuanlan.zhihu.com/p/260162240
        At = At.div(norm.expand_as(At))

        dis = torch.matmul(D.view(batch_size,1,T_q*T_k),At).squeeze() # 加入.squeeze()保证不是list
        return A, dis #[1,T_q,T_k] [1]
class BlurGenerationPair(nn.Module):
    def __init__(self):
        super(BlurGenerationPair, self).__init__()

    def forward(self, seq, len_seq):
        batch_size = seq.size(0)
        T = seq.size(1)
        dim = seq.size(2)
        
        SeqtoBlur = torch.zeros( size=(batch_size,T,T),dtype=torch.float32 ).cuda()
        avged_seq = torch.zeros_like(seq).cuda()
        avged_len = torch.zeros_like(len_seq).cuda()
        R = torch.zeros(batch_size,T).cuda()
        avged_R = torch.zeros(batch_size,T).cuda()

        for b in range(batch_size):
            Tc = len_seq[b].item() # 没有pad的长度
            R[b,0:Tc] = torch.tensor([float(t+1)/float(Tc) for t in range(Tc)])

            if Tc>4:
                randseed = random.randint(1,10)
                if randseed>5:
                    tempseq = seq[b,0:Tc,:] # 将没有pad的部分取出来
                    tempseq = tempseq.view(1,1,Tc,dim) #[1,1,9,768]
                    if randseed>7:
                        GaussKer = torch.tensor([[[[0.1, 0.8, 0.1]]]])    # [1,1,1,3]                                            
                        midnum = random.uniform(0.55,0.9)
                        sidenum = (1.-midnum)/2.
                        GaussKer[0,0,0,0] = sidenum
                        GaussKer[0,0,0,2] = sidenum
                        GaussKer[0,0,0,1] = midnum
                        GaussKer = GaussKer.transpose(2,3).cuda() # [1, 1, 3, 1]
                        stride = random.randint(1,3)
                        outTc = int((Tc-3)/stride) + 1
                        blured_seq = F.conv2d(tempseq,GaussKer,stride=[stride,1]) # conv2d卷积后的seq
                        avged_seq[b,0:outTc,:] = blured_seq.squeeze(1)
                        startf = 0
                        for i in range(outTc):
                            endf = startf + 3
                            SeqtoBlur[b,startf:endf,i] = GaussKer.squeeze()
                            startf = startf + stride
                        #SeqtoBlur[b,1:Tc-1,0:outTc] = torch.eye(Tc-2)
                        avged_R[b,0:outTc] = torch.tensor([float(t+1)/float(outTc) for t in range(outTc)])
                        avged_len[b] = outTc

                    else:
                        GaussKer = torch.tensor([[[[0.1, 0.15, 0.5, 0.15, 0.1]]]])                                               
                        midnum = random.uniform(0.33,0.5)
                        sidenum = (1.-midnum)/2.
                        sidenum1 = random.uniform(0.16,sidenum)
                        sidenum2 = sidenum - sidenum1
                        GaussKer[0,0,0,0] = sidenum2
                        GaussKer[0,0,0,1] = sidenum1
                        GaussKer[0,0,0,2] = midnum
                        GaussKer[0,0,0,3] = sidenum1
                        GaussKer[0,0,0,4] = sidenum2
                        GaussKer = GaussKer.transpose(2,3).cuda()
                        stride = random.randint(1,3)
                        outTc = int((Tc-5)/stride) + 1
                        blured_seq = F.conv2d(tempseq,GaussKer,stride=[stride,1])
                        startf = 0
                        for i in range(outTc):
                            endf = startf + 5
                            SeqtoBlur[b,startf:endf,i] = GaussKer.squeeze()
                            startf = startf + stride
                        #SeqtoBlur[b,1:Tc-1,0:outTc] = torch.eye(Tc-2)
                        avged_R[b,0:outTc] = torch.tensor([float(t+1)/float(outTc) for t in range(outTc)])
                        avged_len[b] = outTc
                else:
                    min_len = int(0.5*Tc)
                    if min_len<1:
                        min_len = 1
                    max_len = int(0.8*Tc)
                    if max_len<1:
                        max_len = 1
                    if max_len>Tc:
                        max_len = Tc
                    cur_len = random.choice(range(min_len,max_len+1))
                    #print(cur_len)
                    interval = random.sample(range(1,Tc),cur_len-1)
                    interval.sort()
                    interval.append(Tc)
                    #print(interval)
                    startf = 0
                    for i in range(cur_len):
                        endf = interval[i]
                        templ = endf-startf
                        localatt = torch.softmax(torch.tensor([random.gauss(0,1) for _ in range(templ)]).view(templ,1),dim=0).cuda()
                        #print(localatt)               
                        tempfr = seq[b,startf:endf,:]
                        avged_seq[b,i,:] = (tempfr*localatt).sum(dim=0)
                        #print(localatt.shape, startf, endf)
                        SeqtoBlur[b,startf:endf,i] = localatt.view(-1)
                        startf = endf                  
                    avged_R[b,0:cur_len] = torch.tensor([float(t+1)/float(cur_len) for t in range(cur_len)])                   
                    avged_len[b] = cur_len
            else:
                if Tc>2:
                    tempseq = seq[b,0:Tc,:]
                    tempseq = tempseq.view(1,1,Tc,dim)
                    GaussKer = torch.tensor([[[[0.1, 0.8, 0.1]]]])                                               
                    midnum = random.uniform(0.55,0.9)
                    sidenum = (1.-midnum)/2.
                    GaussKer[0,0,0,0] = sidenum
                    GaussKer[0,0,0,2] = sidenum
                    GaussKer[0,0,0,1] = midnum
                    GaussKer = GaussKer.transpose(2,3).cuda()
                    blured_seq = F.conv2d(tempseq,GaussKer)
                    avged_seq[b,0:Tc-2,:] = blured_seq.squeeze(1)
                    SeqtoBlur[b,1:Tc-1,0:Tc-2] = torch.eye(Tc-2)
                    avged_R[b,0:Tc-2] = torch.tensor([float(t+1)/float(Tc-2) for t in range(Tc-2)])
                    avged_len[b] = Tc-2
                else:
                    avged_seq[b,:,:] = seq[b,:,:]
                    SeqtoBlur[b,0:Tc,0:Tc] = torch.eye(Tc)
                    avged_R[b,:] = R[b,:]
                    avged_len[b] = Tc

        # 对齐矩阵 SeqtoBlur.detach(), 增强出的seq avged_seq.detach(), avged_len.detach(), 每个位置占总长度的比率 R.detach(), avged_R.detach(), 

        return SeqtoBlur.detach(), avged_seq.detach(), R.detach(), avged_R.detach(), avged_len.detach()



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
        """
        seq : [batch(*choice_num), event_num, dim]
        len_seq: [batch(*choice_num), 1]
        """
        margin=0.5
        m = 0.03
        batch_size = seq.size(0)
        if mode == "contrastive-fine-tuning":
            targets = torch.tensor([[1 if int(b_2/5) == int(b_1/5) else 0 for b_2 in range(batch_size)] for b_1 in range(batch_size)]).cuda() #对角矩阵
        else:
            targets = torch.eye(batch_size).cuda() #对角矩阵


        SeqtoBlur, avged_seq, R, avged_R, avged_len = self.aug(seq, len_seq)

        DisMat = torch.zeros([batch_size,batch_size],dtype=torch.float).cuda()
        Lalign = 0
        Lalign2 = 0
        Ldis = 0
        Ldis2 = 0



        for b in range(batch_size):
            seq0 = seq[b,0:len_seq[b],:].unsqueeze(0)
            R0 = R[b,0:len_seq[b]].unsqueeze(0)
            avged_seq0 = avged_seq[b,0:avged_len[b],:].unsqueeze(0)
            avged_R0 = avged_R[b,0:avged_len[b]].unsqueeze(0)
            seq1 = self.encoder(seq0)
            avged_seq1 = self.encoder(avged_seq0)
            Aa1b, disa1b = self.alignment(seq1, avged_seq1, R0, avged_R0)
            Aa2b, disa2b = self.alignment(avged_seq1, seq1, avged_R0, R0)

            # disa1b = disa1b.view(-1)
            # disa2b = disa2b.view(-1)
            DisMat[b,b] = DisMat[b,b] + (disa1b + disa2b)/2.0
            Ldis = Ldis + disa1b + disa2b
            Ldis2 = Ldis2 + self.mse(disa1b,disa2b)
            Lalign2 = Lalign2 + self.mse(Aa1b,Aa2b.transpose(1,2))
            tempAlign = SeqtoBlur[b,0:len_seq[b],0:avged_len[b]].unsqueeze(0)
            #print(tempAlign.shape)
            Lalign = Lalign + self.mse(Aa1b,tempAlign) + self.mse(Aa2b,tempAlign.transpose(1,2))

            for b2 in range(b+1,batch_size):
                shift_seq0 = seq[b2,0:len_seq[b2],:].unsqueeze(0)
                shift_R0 = R[b2,0:len_seq[b2]].unsqueeze(0)

                shift_seq1 = self.encoder(shift_seq0)
                As1b, dis1b = self.alignment(seq1, shift_seq1, R0, shift_R0)
                As2b, dis2b = self.alignment(shift_seq1, seq1, shift_R0, R0)

                dis1b = dis1b.view(-1)
                dis2b = dis2b.view(-1)
                DisMat[b,b2] = DisMat[b,b2] + dis1b
                DisMat[b2,b] = DisMat[b2,b] + dis2b
                Ldis2 = Ldis2 + self.mse(dis1b,dis2b)
                Lalign2 = Lalign2 + self.mse(As1b,As2b.transpose(1,2))

        loss = 0
        c = 0
        for i in range(batch_size):
            pos_pair_ = torch.masked_select(DisMat[i], targets[i].squeeze()==1)
            neg_pair_ = torch.masked_select(DisMat[i], targets[i].squeeze()!=1)

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            if self.hard_mining is not None:
                
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ - m <  pos_pair_[-1])
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ + m > neg_pair_[0])
            
                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    c += 1
                    continue 
            
                pos_loss = 2.0/self.beta * torch.mean(torch.log(1 + torch.exp(self.beta*(pos_pair - 0.5))))
                neg_loss = 2.0/self.alpha * torch.mean(torch.log(1 + torch.exp(-self.alpha*(neg_pair - 0.5))))

            else:  
                pos_pair = pos_pair_
                neg_pair = neg_pair_ 

                pos_loss = torch.mean(torch.log(1 + torch.exp(2*(pos_pair - 0.5))))
                neg_loss = torch.mean(torch.log(1 + torch.exp(-self.alpha*(neg_pair - 0.5))))

            # 根据targets中，没有作为当前sample负例的neg_pair，那么无效数据，按理来说这句实际不会执行
            if len(neg_pair) < 1:
                c += 1
                continue

            loss = loss + pos_loss + neg_loss

        Ldis /= batch_size
        Lalign /= batch_size
        Ldis2 /= batch_size
        Lalign2 /= batch_size
        if batch_size - c != 0: 
            L = loss/(batch_size - c) + Ldis/batch_size + Lalign/batch_size #有c条数据没有计算loss
        else:
            L = Ldis/batch_size + Lalign/batch_size
        return L