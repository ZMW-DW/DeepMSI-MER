import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.layers import DropPath, to_2tuple, trunc_normal_
from module.src.efficient_kan.kan_block import KAN
from module.vsc_model.vsc import VSC
import torch.nn.functional as F
import math
from timm.layers import ndgrid
from typing import Tuple
from timm.layers import get_act_layer, LayerType, Mlp


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1):
        """
        一个基于多层感知机的分类头。
        
        参数:
        - input_dim (int): 输入特征的维度。
        - hidden_dims (list[int]): 每层隐藏层的维度列表。
        - num_classes (int): 输出类别数。
        - dropout (float): Dropout 概率。
        """
        super(ClassifierHead, self).__init__()
        layers = []
        in_dim = input_dim
        
        # 创建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # 添加最后的分类层
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量，形状为 [batch_size, input_dim]。

        返回:
        - torch.Tensor: 输出张量，形状为 [batch_size, num_classes]。
        """
        return self.mlp(x)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))## 输入进来wind形状是 64 8 8  96
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

#   W-MSA/SW-MSA
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
            self,
            dim: int,
            window_size: to_2tuple,
            num_heads: int,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            pretrained_window_size: Tuple[int, int] = (0, 0)
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = to_2tuple(pretrained_window_size)
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # 替换MLP使用KAN
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self._make_pair_wise_relative_positions()

    # 制作一个位置表 用于后续的w-MSA与SW-MSA操作
    def _make_pair_wise_relative_positions(self):
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0]).to(torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1]).to(torch.float32)
        relative_coords_table = torch.stack(ndgrid(relative_coords_h, relative_coords_w))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2


        if self.pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)


        # relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table *= self.window_size[0]  # normalize to -8, 8 或者

        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(ndgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  ## x输入形状是 64 64 96 ；对应到每个维度就是B是64，64，C是96
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        # 使用KAN替换MLP
        # relative_position_bias_table = self.cpb_kan(self.relative_coords_table.view(-1, 2))
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)


        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#   主模块
class SwinTransformerV2Block_KAN_VSC(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self,
            dim: int,
            input_resolution,
            num_heads: int,             #     8         7         7         6          6         5
                                        # 1.W-MSA -- SW-MSA    2.W-MSA -- SW-MSA    2.W-MSA -- SW-MSA
            window_reset: int,          # [  16,       14,        14,      12,         12,       10  ] 用于告知模型其有多少个token 一开始16 一轮就结束 14两轮 12两轮 10两轮
            keep_number: int,   # [  196,       -,       144,       -,        100,        -  ]  用于告知vsc模块需要保留多少个token
            window_re_size: int,

            window_size: int = 8,
            shift_size: int = 0,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: LayerType = "gelu",
            norm_layer: nn.Module = nn.LayerNorm,
            vsc_abel: bool = False, #加一个vsc判断 如果vsc为true着该stage使用vsc模块
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        act_layer = get_act_layer(act_layer)


        # 用于操作VSC的参数
        self.vsc_abel = vsc_abel
        self.window_re_size = window_re_size
        self.window_reset = window_reset
        self.keep_number = keep_number

        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        #改进处
        self.attn_vsc = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_re_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        #改进处 增加VSC模块
        self.vsc = VSC(
            embed_dim=dim,
            keep_number=keep_number,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm3 = norm_layer(dim)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        ## 3.mask
        if self.vsc_abel:
            if self.shift_size > 0:
                # calculate attention mask for SW-MSA
                H = W = self.window_re_size
                # H = W = self.window_reset
                img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_re_size),
                            slice(-self.window_re_size, -self.shift_size),
                            slice(-self.shift_size,
                                  None))  ## 生成一个元祖，第0个元素 slice(0, -7, None) 第1个元素slice(-7, -3, None) 第2个元素slice(-3, None, None) 每个元素三个分别代表 start step stop
                w_slices = (slice(0, -self.window_re_size),
                            slice(-self.window_re_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(img_mask, self.window_re_size)  # nW, window_size, window_size, 1
                mask_windows = mask_windows.view(-1, self.window_re_size * self.window_re_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None

            self.register_buffer("attn_mask", attn_mask)

        else:
            if self.shift_size > 0:
                # calculate attention mask for SW-MSA
                H, W = self.input_resolution

                img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size,None))  ## 生成一个元祖，第0个元素 slice(0, -7, None) 第1个元素slice(-7, -3, None) 第2个元素slice(-3, None, None) 每个元素三个分别代表 start step stop
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None

            self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor, t_cls: torch.Tensor) -> torch.Tensor:
        if self.vsc_abel:
            H = W = self.window_reset  # 16 14 14 12 12 10
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"

            shortcut = x
            x = x.view(B, H, W, C)

            if self.shift_size == 0:
                # 1.W-MSA
                # partition windows
                x_windows = window_partition(x, self.window_re_size)  ## 64 8 8 384  # nW*B, window_size, window_size, C
                x_windows = x_windows.view(-1, self.window_re_size * self.window_re_size,C)  ## 64 64 384  # nW*B, window_size*window_size, C
                # W-MSA
                attn_windows = self.attn_vsc(x_windows, mask=self.attn_mask)  ## attn_windows 64 64 384  # nW*B, window_size*window_size, C
                # merge windows
                attn_windows = attn_windows.view(-1, self.window_re_size, self.window_re_size, C)
                x = window_reverse(attn_windows, self.window_re_size, H, W)  # B H' W' C
                x = x.view(B, H * W, C)  # 回到原始信息shape

                # 2. LN 3. 残差
                x = x + self.drop_path1(self.norm1(x))

                # 修改处 改进处
                # 4. VSC模块  5. LN  6. 残差(删除)
                x = self.drop_path2(self.norm2(self.vsc(x, t_cls)))
                B, L, C = x.shape

                # 7.MLP改为KAN 8.LN 9.残差
                # 注意kan的输输入应该为[(batch_size, in_features)]
                # 对输入张量进行规范化，然后将其形状重塑为 [B*L, C]，其中 B 为批次大小，L 为序列长度，C 为特征维度
                # 将重塑后的张量再次重塑为 [B, L, C] 的形状，恢复原始的批次大小、序列长度和特征维度
                # x = x + self.drop_path3(self.norm3(self.kan(x.view(-1, x.size(2))).reshape(B, L, C)))
                x = x + self.drop_path3(self.norm3(self.mlp(x)))


            elif self.shift_size > 0:
                # 1.SW-MSA
                # cyclic shift
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x,self.window_re_size)  ## 64 8 8 96  # nW*B, window_size, window_size, C
                x_windows = x_windows.view(-1, self.window_re_size * self.window_re_size,C)  ## 64 64 96 ；64是bs乘以每个图片的窗口，49是一个窗口中的有多少个元素，对应到NLP中，就是有多少个单词，96是通道数，对应到NLP就是每个单词的维度  # nW*B, window_size*window_size, C
                # SW-MSA
                attn_windows = self.attn_vsc(x_windows, mask=self.attn_mask)  ## attn_windows 64 64 96，和trm没区别哈，长度49不变，toen维度96没变；  # nW*B, window_size*window_size, C
                # merge windows
                attn_windows = attn_windows.view(-1, self.window_re_size, self.window_re_size, C)  # 汇总会有的窗口
                shifted_x = window_reverse(attn_windows, self.window_re_size, H, W)  # B H' W' C
                # reverse cyclic shift
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                x = x.view(B, H * W, C)

                # 2. LN 3. 残差
                x = shortcut + self.drop_path1(self.norm1(x))

                # 4.MLP改为KAN 5.LN 6.残差
                # x = x + self.drop_path2(self.norm2(self.kan(x.view(-1, x.size(2))).reshape(B, L, C)))
                x = x + self.drop_path3(self.norm3(self.mlp(x)))

        else:
            H, W = self.input_resolution  ## 输入的x形状是:1 4096 96 得到 H是64 W是64
            B, L, C = x.shape  ## 这个是B是1，L是seq_len等于4096，C是通道数为96  原始信息
            assert L == H * W, "input feature has wrong size"

            shortcut = x  # 保留原始信息
            ## 从1 4096 96 转为  1 64 64 96  注意这个时候就从输入的一维那种向量转为了特征图，也就是一维4096，到了一个二维特征图 64 64 ，对应到原始图片是256 256
            x = x.view(B, H, W, C)

            # 1.W-MSA/SW-MSA
            # partition windows
            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  ## 64 8 8 96  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  ## 64 64 96 ；64是bs乘以每个图片的窗口，64是一个窗口中的有多少个元素，对应到NLP中，就是有多少个单词，96是通道数，对应到NLP就是每个单词的维度  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows,mask=self.attn_mask)  ## attn_windows 64 64 96，和trm没区别哈，长度64不变，toen维度96没变；  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)

            # 2.LN 3.残差
            x = shortcut + self.drop_path1(self.norm1(x))

            # 4.MLP改为KAN 5.LN 6.残差
            # x = x + self.drop_path2(self.norm2(self.kan(x.view(-1, x.size(2))).reshape(B, L, C)))
            x = x + self.drop_path3(self.norm3(self.mlp(x)))

        return x

#   下采样
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self,
            input_resolution,
            dim,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False) #其中dim应该是
        self.norm = norm_layer(4 * dim)
        """""""""
        1、  256/4 *  256/4 = 64 * 64 = 4096个 其中c为96
        2、  256/8 *  256/8 = 32 * 32 = 1024个 其中c为192
        3、 256/16 * 256/16 = 16 * 16 = 256个  其中c为384  
        
                进行vsc操作后 逐次变为 196 144 100
            14 * 14 = 196
            
            12 * 12 = 144
            
            10 * 10 = 100
            
            
        4、 256/32 * 256/32 =  8 *  8 = 64个   其中c为768  进行vsc操作后 变为 32/4=8个
        其中window为8
        最后[batch_size, 8, 768]
        """""""""


    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution #
        B, L, C = x.shape ## 第一次：输入进来x为1 4096 96
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C) ## 这里x变为了 1 64 64 96

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C ## x0形状为1 32 32 96
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C ## x1形状为1 32 32 96
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C## ## x2形状为1 32 32 96
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C ## ## x3形状为1 32 32 96
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C  ## x为[1, 32, 32, 96 * 4=384]
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C ## 1 1024 384

        x = self.norm(x)
        x = self.reduction(x) # 1 1024 192

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

#   四个stage改造点
class Stage_Layer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    
    def __init__(
            self,
            dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,

            vsc_abel: bool = False,
            window_re_size = [8, 7, 7, 6, 6, 5],
            window_reset = [16, 14, 14, 12, 12, 10],
            keep_number = [196, 196, 144, 144, 100, 100]
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.vsc_abel = vsc_abel
        self.window_re_size = window_re_size
        self.window_reset = window_reset
        self.keep_number = keep_number

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block_KAN_VSC(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,

                window_reset=self.window_reset[i],
                keep_number=self.keep_number[i],
                vsc_abel=self.vsc_abel,
                window_re_size=self.window_re_size[i]
            )
            for i in range(depth)]
        )

        # patch merging layer
        if downsample is not None:
            if self.vsc_abel:
                self.downsample = downsample((10, 10), dim=dim, norm_layer=norm_layer)
            else:
                self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, t_cls):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, t_cls)
            else:
                x = blk(x, t_cls)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

#   初始图片使用卷积做patchembedding
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 256.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
            self,
            img_size=256,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]


        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape # B是1  C3 H224  W224
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        #flatten: [B, C, H, W] -> [B, C. HW]
        #transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C  1 3136 96  3136就是我们的56*56 压扁
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

#   主要的模块
class SwinTransformerV2_KAN_VSC(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        kan_ratio (float): Ratio of KAN hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    def __init__(
            self,
            img_size=256,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            global_pool="avg",
            **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        #11 修改处
        for i_layer in range(self.num_layers):
            layer = Stage_Layer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ) if i_layer != 3  else (5, 5),  ##重大修改处 ！！！ 嘻嘻 哈哈哈哈哈 
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size if i_layer != 3 else 5, #加一个窗口大小变化
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                vsc_abel=False if i_layer != 2 else True  #加一个vsc判断 如果vsc为true着该stage使用vsc模块
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.head = ClassifierHead(
            input_dim=self.num_features,
            hidden_dims=[512, 256],
            num_classes=num_classes,
        )if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    #权重初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, t_cls):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, t_cls)  # B L C 再传递一个文本特征

        x = self.norm(x)  # B L C

        if self.global_pool == 'avg':
            x = self.avgpool(x.transpose(1, 2))  # B 1 C
            x = torch.flatten(x, 1) # [B, 1*C] ==  [B, C]
        elif self.global_pool == 'max':
            x = self.maxpool(x)
            x = torch.flatten(x, 1)
        else :
            x = x

        return x

    def forward(self, x, t_cls):
        x = self.forward_features(x, t_cls)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops