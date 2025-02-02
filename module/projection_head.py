import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg

class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,#输入的图像和文本的tensor（图像：768；文本：768）
            projection_dim=cfg.projection_dim,  #输出的tensor：256
            dropout=cfg.Projection_dropout
    ):
        super(ProjectionHead,self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
