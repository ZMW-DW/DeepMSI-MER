import torch
from torch import nn
import config as CFG
from module.tcn_model.tcn import TCN
from image_model.swin_transformer_V2_kan import SwinTransformerV2_KAN

class VideoEncoder(nn.Module):
    def __init__(
            self,
    ):
        super(VideoEncoder, self).__init__()
        self.model = SwinTransformerV2_KAN(
            num_classes=0,
            global_pool="avg"
        )
        # 定义TCN模块
        self.tcn = TCN(
            input_size=CFG.image_input,
            num_channels=CFG.num_channels,
            kernel_size=3
        )

    def forward(self, x):
        # x的形状是 [batch_size, frames, C, H, W]  -- [8,15,3,384,384]
        batch_size, frames, C, H, W = x.size()
        all_features = []

        for i in range(frames):
            # frame的形状将是[batch_size, C, H, W]
            frame = x[:, i, :, :, :].view(-1, C, H, W)

            # 获取特征，输出的形状为 [batch_size, hidden_size]
            feature = self.model(frame)

            # 添加一帧的特征
            all_features.append(feature)

        # 将所有帧的特征堆叠起来，形成形状为        [batch_size, frames, hidden_size] -- [8, 15, 1024]
        features = torch.stack(all_features, dim=1)
        # 特征需要转换为[batch_size, hidden_size, frames]  -- [8, 1024, 15]
        features = features.permute(0, 2, 1)

        # 使用TCN进行时序特征提取
        tcn_output = self.tcn(features)  # [batch_size, hidden_size, frames]  -- [8, 786, 15]
        output = tcn_output[:, :, -1]  # [batch_size, hidden_size] -- -- [8, 786]

        # 返回最后一层的隐藏状态作为序列的表示
        return output
