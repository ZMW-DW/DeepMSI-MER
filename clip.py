import sys
sys.path.append('/root/autodl-tmp/MyModel')
import torch
from torch import nn
import config as CFG
from module.video_encoder_vsc import VideoEncoder_VSC
from module.projection_head import ProjectionHead
import torch.nn.functional as F

class processor(nn.Module):
    def __init__(
        self, 
        embedding_dim,
        num_projection_layers=CFG.tCLS_layers, 
        projection_dims=384, 
        dropout_rate=CFG.tCLS_dropout
    ):
        super(text_CLS, self).__init__()
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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim

        # 创建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # 添加 Batch Normalization
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))  # 使用 LeakyReLU 激活
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # 添加最后的分类层
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class Multimodel(nn.Module):
    def __init__(
            self,
            image_embedding=CFG.image_embedding,
            text_embedding=CFG.text_embedding,
            audio_embedding=CFG.audio_embedding,
            projection_dim=CFG.projection_dim,
    ):
        super().__init__()
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.audio_embedding = audio_embedding
        self.projection_dim = projection_dim

        self.video_encoder_vsc = VideoEncoder_VSC()
        # self.processor = processor(embedding_dim=(text_embedding + audio_embedding))
        self.processor = MLP(
            input_dim=(text_embedding + audio_embedding),
            hidden_dims=CFG.processor_hidden_dims,
            output_dim=384,
            dropout=CFG.processor_dropout
        )

        self.video_projection = ProjectionHead(embedding_dim=self.image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=self.text_embedding)
        self.audio_projection = ProjectionHead(embedding_dim=self.audio_embedding)

         # 分类头，将图像和文本嵌入拼接后映射到情感类别
        self.classifier =  MLP(
            input_dim=3 * self.projection_dim,
            hidden_dims=CFG.classifier_hidden_dims,
            output_dim=CFG.classifier_num_classes,
            dropout=CFG.classifier_dropout
        ) 

    def forward(self, frames, text_feature, audio_features):
        merge_feature = torch.cat([text_feature, audio_features], dim=-1)    # [batch_size, 1024 + 768]
        merge_cls = self.processor(merge_feature)

        #获取视频特征
        video_features = self.video_encoder_vsc(frames, merge_cls)

        #将两者对齐
        video_projection = self.video_projection(video_features)    # [batch_size, 256]
        text_projection  = self.text_projection(text_feature)      # [batch_size, 256]
        audio_projection = self.audio_projection(audio_features)

        # 进行特征融合以及分类
        fused_embeddings = torch.cat([video_projection, text_projection, audio_projection], dim=-1)    # [batch_size, 512]

        logits = self.classifier(fused_embeddings)

        return video_projection, text_projection, audio_projection, logits
