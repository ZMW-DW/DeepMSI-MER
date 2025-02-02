import numpy as np
import cv2
import torch
import sys
sys.path.append('G:\大论文实验\大论文模型')


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]



def get_video_duration(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Cannot open video: {video_path}")
        return 0
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frames / fps
    # print(f"Video duration (seconds): {duration}")
    return duration


def norm_NLC(input):
    """""""""

    公式： x = (x - x.mean)/(x.std + 1e-5)

    输入的shape为（N,L,C）
    """""""""
    input_mean = input.mean(dim=-1, keepdim=True)
    input_std  = input.std(dim=-1, keepdim=True, unbiased=False)
    input_norm = (input - input_mean) / (input_std + 1e-5)
    return input_norm


def cosin_similarity_matrix(tensor1, tensor2):
    """""""""
    
    公式： x = (tensor1 @ tensor2) / |tensor1| * |tensor2|

    输入的shape为（N,L,C）
    """""""""
    dot_matrix = torch.matmul(tensor1, tensor1.transpose(-1, -2))    #(N, L, L)
    norm1_matrix = torch.sqrt(torch.sum(tensor1 * tensor1, dim=1)).unsqueeze(2)   #(N, L, 1)
    norm2_matrix = torch.sqrt(torch.sum(tensor2 * tensor2, dim=1)).unsqueeze(1)   #(N, 1, L)
    similarity_matrix = dot_matrix / (norm1_matrix * norm2_matrix)
    return similarity_matrix #(N, L, L)

def cosin_similarity(tensor1, tensor2):
    """""""""
    
    公式： x = (tensor1 @ tensor2) / |tensor1| * |tensor2|

    输入的shape为（N,C）
    """""""""
    # 计算点积，大小 (N, N)
    dot = torch.matmul(tensor1, tensor2.transpose(-1, -2)) #(N, N)

     # 计算 L2 范数
    norm1 = torch.norm(tensor1, p=2, dim=1).unsqueeze(1)  # (batch_size, 1)
    norm2 = torch.norm(tensor2, p=2, dim=1).unsqueeze(0)  # (1, batch_size)

    print("norm1\n", norm1)
    print("norm2\n", norm2)
    print("norm1 * norm2", norm1 * norm2)

    return dot / (norm1 * norm2)


# batch_size = 1
# sequence_lentgh = 3
# feature_dim = 3
# x = torch.randn(batch_size, sequence_lentgh, feature_dim)
# # print(x)
# # print(x[:,0])
# # mean = x[:,0].mean()
# # print(f"mean = {mean}")
# # std = x[:,0].std()
# # print(f"std = {std}")
# # k = (x[0,0]-mean)/(std + 1e-5)
# # print(f"{x[0,0]} - {mean} / ({std} + {1e-5})  =  {k}")

# x = norm_NLC(x)
# print(x)
# from torch.nn import LayerNorm
# LN = LayerNorm(feature_dim)
# print(LN(x))




# ------------------------------------------------------------------------

# batch_size2 = 5
# sequence_lentgh = 3
# feature_dim2 = 3
# x1 = torch.randn(batch_size2, sequence_lentgh, feature_dim2)
# x2 = torch.randn(batch_size2, sequence_lentgh, feature_dim2)
# similarity_matrix = cosin_similarity_matrix(x1, x2)
# print(similarity_matrix)



# ------------------------------------------------------------------------
# import torch.nn.functional as F

# def main():

#     # 初始化
#     batch_size = 4

#     feature_dim = 256
#     temperature = 0.05
#     # temperature = 1

#     # labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])  # 示例标签
#     # labels = torch.tensor([0, 0, 0, 0, 0, 2, 0, 0])  # 示例标签
#     # labels = torch.tensor([0])
#     # labels = torch.tensor([0, 0])
#     # labels = torch.tensor([0, 1])
#     # labels = torch.tensor([0, 1, 2])

#     # labels = torch.tensor([0, 1, 2, 0])

#     labels = torch.tensor([1, 1, 1, 1])
    
#     print("---------------------------------------------------------------------------------------")
    
#     # 构造标签矩阵和 mask
#     labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)   # (4, 4)
#     # print("Labels Matrix:\n", labels_matrix)

#     # 正样本mask
#     pos_mask = labels_matrix.float() - torch.eye(batch_size, device=labels.device)
#     print("pos_mask:\n", pos_mask)

#     # 负样本mask
#     neg_mask =  (~labels_matrix).float()
#     print("neg_mask\n", neg_mask)

#     # neg_count = torch.sum(neg_mask == 1).item()
#     # print("neg_count\n", neg_count)
    
#     # if torch.sum(pos_mask == 1).item() == 0 or torch.sum(neg_mask == 1).item() == 0:
#     if torch.sum(pos_mask == 1).item() == 0:
#         return torch.tensor(1e-8, device=labels.device)

#     print("---------------------------------------------------------------------------------------")


#     x1 = torch.randn(batch_size, feature_dim)
#     x2 = torch.randn(batch_size, feature_dim)


#     x1 = F.normalize(x1, p=2, dim=-1)
#     x2 = F.normalize(x2, p=2, dim=-1)

#     print("---------------------------------------------------------------------------------------")

#     # 计算余弦相似度
#     similarity_matrix = torch.matmul(x1, x2.transpose(-1, -2)) / temperature
#     print("similarity_matrix\n", similarity_matrix)

#     # 指数化
#     exp_similarity = torch.exp(similarity_matrix)
#     print("exp_similarity\n", exp_similarity)

#     # 计算对比损失
#     print("---------------------------------------------------------------------------------------")    

#     positive_similarity = exp_similarity * pos_mask # 正样本相似度
#     print("positive_similarity\n", positive_similarity)

#     if torch.sum(neg_mask == 1).item() == 0:
#         per_value = positive_similarity / (positive_similarity + 1e-8)
#         print("positive_similarity / (positive_similarity + 1e-8)\n", per_value)
#     else:
#         negtive_similarity  = exp_similarity * neg_mask
#         print("negtive_similarity\n", negtive_similarity)
#         negtive_similarity = negtive_similarity.sum(1).unsqueeze(1).expand_as(positive_similarity)
#         print("negtive_similarity\n", negtive_similarity)
#         print("positive_similarity + negtive_similarity\n", positive_similarity + negtive_similarity)
#         per_value = positive_similarity / (positive_similarity + negtive_similarity)
#         print("positive_similarity / (positive_similarity + negtive_similarity)\n", per_value)   
        
#     loss = torch.where(per_value == 0, torch.tensor(1.0), per_value)
#     print("loss after 0 befor log\n", loss)
#     print("loss after log\n", -torch.log(loss))
#     loss = -torch.log(loss).sum(1) / (batch_size - 1)
#     print("loss = -torch.log(loss).sum(1) / (batch_size - 1)\n", loss)
#     loss = loss.mean()
#     print("loss.mean()\n", loss)

#     return loss

# if __name__ == "__main__":
#     loss = main()
#     print(loss)
