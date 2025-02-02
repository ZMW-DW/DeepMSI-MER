import torch
from torch import nn
import config as CFG
import torch.nn.functional as F
from torch import nn

    
class contrastive_loss(nn.Module):
    def __init__(
            self,
            initial_temperature=CFG.temperature,
            projection_dim=CFG.projection_dim,
            mode='avg'
    ):
        super(contrastive_loss, self).__init__()
        self.temperature = initial_temperature
        self.mode = mode

    def forward(self, video_projection, text_projection, audio_projection, labels):
        batch_size = video_projection.shape[0]
        
        # 生成掩码
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  #相同标签的为True 
        pos_mask = labels_matrix.float() - torch.eye(batch_size, device=labels.device)  # 正样本：pos_mask 排除自己与自己的相似性
        neg_mask =  (~labels_matrix).float()    #负样本：
        
        # 如果正样本为0个的话，直接不需要后续了
        if torch.sum(pos_mask == 1).item() == 0:
            return torch.tensor(1e-8, device=labels.device, requires_grad=True)

        # 归一化
        video_projection = F.normalize(video_projection, p=2, dim=-1)
        text_projection  = F.normalize(text_projection, p=2, dim=-1)
        audio_projection = F.normalize(audio_projection, p=2, dim=-1)    
        
        # 计算视频与文本特征之间的相似度
        similarity_matrix_VT = torch.matmul(video_projection, text_projection.transpose(-1,-2)) / self.temperature
        similarity_matrix_VA = torch.matmul(video_projection, audio_projection.transpose(-1,-2)) / self.temperature

        # 指数化
        exp_similarity_VT = torch.exp(similarity_matrix_VT)
        exp_similarity_VA = torch.exp(similarity_matrix_VA)

        #正样本相似度 -- 分子
        positive_similarity_VT = exp_similarity_VT * pos_mask
        positive_similarity_VA = exp_similarity_VA * pos_mask

        if torch.sum(neg_mask == 1).item() == 0:
            loss_VT = positive_similarity_VT / (positive_similarity_VT + 1e-8)
            loss_VA = positive_similarity_VA / (positive_similarity_VA + 1e-8)
        else:
            negtive_similarity_VT  = exp_similarity_VT * neg_mask
            negtive_similarity_VT = negtive_similarity_VT.sum(1).unsqueeze(1).expand_as(positive_similarity_VT)
            loss_VT = positive_similarity_VT / (positive_similarity_VT + negtive_similarity_VT)
            
            negtive_similarity_VA  = exp_similarity_VA * neg_mask
            negtive_similarity_VA = negtive_similarity_VA.sum(1).unsqueeze(1).expand_as(positive_similarity_VA)
            loss_VA = positive_similarity_VA / (positive_similarity_VA + negtive_similarity_VA)
        
        # 防止对角线影响   
        loss_VT = torch.where(loss_VT == 0, torch.tensor(1.0), loss_VT)
        loss_VA = torch.where(loss_VA == 0, torch.tensor(1.0), loss_VA)
        
        # 每一个样本与其他样本的的loss和
        loss_VT = -torch.log(loss_VT).sum(1) / (batch_size - 1)
        loss_VA = -torch.log(loss_VA).sum(1) / (batch_size - 1)
        
        # 对每一个样本的loss取平均
        loss_VT = loss_VT.mean()
        loss_VA = loss_VA.mean()
        
        loss = (loss_VT + loss_VA) / 2

        return loss


class Mutiloss(torch.nn.Module):
    def __init__(
            self,
            alpha=CFG.alpha,
            beta=CFG.beta
    ):
        super(Mutiloss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.contrastive_loss = contrastive_loss()
        self.classification_loss = nn.CrossEntropyLoss()

    def forward(self, video_projection, text_projection, audio_projection, logits, labels):
        # 进行对比学习的损失计算
        contrastive_loss = self.contrastive_loss(video_projection, text_projection, audio_projection, labels)

        # 进行分类损失的计算
        classification_loss = self.classification_loss(logits, labels)
        # 总损失
        total_loss = self.alpha * contrastive_loss + self.beta * classification_loss

        return total_loss, contrastive_loss, classification_loss

