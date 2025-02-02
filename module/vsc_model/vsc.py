import torch
import torch.nn as nn
import torch.nn.functional as F
import config as CFG

class VSC(nn.Module):
    def __init__(
        self, 
        keep_number, 
        embed_dim=384,
        similarity_threshold=CFG.similarity_threshold,
        alpha=CFG.VSC_alpha
    ):
        super(VSC, self).__init__()
        self.keep_number = keep_number
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.epsilon = 1e-8    
        self.embed_dim=embed_dim
        self.similarity_threshold=similarity_threshold
        self.alpha=alpha

    def compression(self, Z_NR, Z_R):
        _, L_NR, D = Z_NR.size()
        _, L_R, _ = Z_R.size()  # 获取非相关和相关序列的维度

        # Step 1: 计算相似度矩阵
        Z_NR_normalized = F.normalize(Z_NR, dim=-1, p=2)  # 归一化
        Z_R_normalized = F.normalize(Z_R, dim=-1, p=2)    # 归一化
        similarity = torch.matmul(Z_NR_normalized, Z_R_normalized.transpose(-1, -2))  # (B, L_NR, L_R)

        # Step 2: 筛选相似度高于阈值的非相关序列
        valid_mask = similarity > self.similarity_threshold  # 筛选满足阈值的序列 (B, L_NR, L_R)
        valid_indices = valid_mask.nonzero(as_tuple=True)  # 获取满足条件的位置
        batch_indices = valid_indices[0]
        related_Z_NR_idx = valid_indices[1]
        related_Z_R_idx = valid_indices[2]
        
        # 如果没有有效的相关序列，则返回原始 Z_R
        if len(batch_indices) == 0:
            return Z_R.clone()

        # Step 3: 计算贡献度并更新相关序列
        exp_similarity = torch.exp(similarity)  # 对相似度进行指数运算
        sum_exp_similarity = exp_similarity.sum(dim=-1, keepdim=True) + self.epsilon  # 归一化分母
        theta_x = exp_similarity / sum_exp_similarity  # 贡献度值 (B, L_NR, L_R)
        
        # 根据有效索引提取贡献度权重
        theta_x_weight = theta_x[batch_indices, related_Z_NR_idx, related_Z_R_idx].unsqueeze(-1)  # (有效序列数, 1)
        weighted_NR = theta_x_weight * Z_NR[batch_indices, related_Z_NR_idx]  # (有效序列数, D)

        # 更新 Z_R
        Z_R_updated = Z_R.clone()
        Z_R_updated[batch_indices, related_Z_R_idx] = (
            self.alpha * Z_R[batch_indices, related_Z_R_idx] +  # 原始 Z_R 的比例
            (1 - self.alpha) * weighted_NR  # 非相关序列贡献
        )
        
        return Z_R_updated

    def forward(self, image_features, t_cls):
        B, L, D = image_features.size()

        # 得到图像的 CLS
        v_cls = self.global_pool(image_features.transpose(1, 2)).squeeze(-1)  # (batch_size, feature_dim)

        # 联合特征向量 m_cls
        m_cls = (v_cls + t_cls).unsqueeze(1)  # (N, 1, D)

        # 归一化
        m_cls_normalized = F.normalize(m_cls, dim=-1, p=2)
        image_norm_normalized  = F.normalize(image_features, dim=-1, p=2)

        # 注意力得分
        scort = F.softmax(torch.matmul(m_cls_normalized , image_norm_normalized.transpose(1, 2)), dim=-1).squeeze(1)
        
        # 相关序列index
        _, top_indices = torch.topk(scort, self.keep_number, dim=-1)

        # 提取 top-k 相关序列
        Z_R = torch.gather(image_features, 1, top_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))  # (B, keep_number, D)

        # 构造布尔掩码，用于标记是否为相关序列
        mask = torch.zeros(B, L, dtype=torch.bool, device=image_features.device)
        mask.scatter_(1, top_indices, True)  # 将 top-k 的位置标记为 True
        
        # 获取非相关序列
        Z_NR = image_features[~mask].view(B, -1, D) # (B, L-keep_number, D)

        # 序列压缩
        Z_R_updated = self.compression(Z_NR, Z_R)

        return Z_R