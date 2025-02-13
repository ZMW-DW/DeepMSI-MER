import sys
sys.path.append('/root/autodl-tmp/MyModel')
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pickle
import pandas as pd
import torch
import torch.optim
import IEMOCAP.config as CFG
from Model import Multimodel
from loss import Mutiloss
import json
import os
from datetime import datetime
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from utils import get_lr
from collections import defaultdict


label_map = {
    0: 'Happiness', 
    1: 'Sadness', 
    2: 'Neutral', 
    3: 'Anger', 
    4: 'Excited', 
    5: 'Frustration'
}

# 随机生成训练顺序
def get_ddp_generator(seed=3407):
    local_rank = torch.distributed.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

# dataset
class MyDataset(Dataset):
    def __init__(
        self,
        file,
        video_dir,
        text_dir,
        audio_dir
    ): 
        super(MyDataset, self).__init__()
        self.data = pd.read_csv(file, sep="\t", header=None, names=["File_name", "Emotion", "Text"])
        self.video_dir = video_dir
        self.text_dir = text_dir
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx]['File_name']
        
        video_path = os.path.join(self.video_dir, f'{file_name}.pkl')
        with open(video_path, 'rb') as f:
            video = pickle.load(f)
        frames = video['images']

        text_path = os.path.join(self.text_dir, f'{file_name}.pkl')
        with open(text_path, 'rb') as f:
            text = pickle.load(f)
        text_feature = text['text_feature']
        label = text['label']
        
        audio_path = os.path.join(self.audio_dir, f'{file_name}.pkl')
        with open(audio_path, 'rb') as f:
            audio = pickle.load(f)
        audio_feature = audio['audio_feature']

        return frames, text_feature, audio_feature, label

# 初始化分布式环境
def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

# 加载之前保存的结果
def load_results(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            results = json.load(f)
        print(f"Previous results loaded from {filename}")
    else:
        results = []
        print(f"No previous results found, starting fresh.")
    return results

# 分布式结果汇总
def reduce_data(data):
    # 将所有GPU的loss进行求和，避免梯度平均
    data = torch.tensor(data, device='cuda')
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    return data / dist.get_world_size()


def main(local_rank, args):
    init_ddp(local_rank)
    
    # 创建dataset以及合并
    train_dataset = MyDataset(file=CFG.IEMOCAP_train, video_dir=CFG.IEMOCAP_train_video_path, text_dir=CFG.IEMOCAP_train_text_path, audio_dir=CFG.IEMOCAP_train_audio_path)
    valid_dataset = MyDataset(file=CFG.IEMOCAP_valid, video_dir=CFG.IEMOCAP_valid_video_path, text_dir=CFG.IEMOCAP_valid_text_path, audio_dir=CFG.IEMOCAP_valid_audio_path)
    test_dataset  = MyDataset(file=CFG.IEMOCAP_test, video_dir=CFG.IEMOCAP_test_video_path, text_dir=CFG.IEMOCAP_test_text_path, audio_dir=CFG.IEMOCAP_test_audio_path)
    full_dataset  = ConcatDataset([train_dataset, valid_dataset, test_dataset])
        
    # 使用K折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
    patience_counter = 0  # 容忍计数器
    early_stopping_patience = 4
    early_stop_flag = torch.tensor(0, device='cuda')  # 提早停止的标志 (0: 不停止, 1: 停止)
        
    if local_rank == 0:                        
        # 加载之前保存的结果记录
        file_name = 'IEMOCAP'
        results = load_results(f'../results/{file_name}.json')
        
    # 每一折的最好结果
    best_accuracies = []
    best_f1_scores = []
    best_label_metrics = []  # 用于保存每一折的 label_metrics
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(full_dataset)):
        if local_rank == 0:
            print(f"\nFold {fold+1}")
            
        best_acc = 0
        best_f1 = 0
        best_lab = {}
        best_label_true = None
        best_label_pred = None

        # 创建训练和验证数据集
        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, valid_idx)
        
        train_sampler = DistributedSampler(train_subset, shuffle=True)
        g = get_ddp_generator()
        test_sampler = DistributedSampler(test_subset, shuffle=False)
        
        # 创建 DataLoader
        train_loader = DataLoader(train_subset, batch_size=CFG.batch_size, pin_memory=True, shuffle=False, sampler=train_sampler, generator=g,)
        test_loader = DataLoader(test_subset, batch_size=CFG.batch_size, pin_memory=True, shuffle=False, sampler=test_sampler)
        
        # 创建模型和损失函数
        model = Multimodel().cuda()
        criterion = Mutiloss().cuda()
        
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            model = SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True
            )
            
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CFG.lr,
            weight_decay=CFG.weight_decay
        )

        for epoch in range(CFG.epochs):
            early_stop_flag.fill_(0)  # 每轮训练开始时重置早停标志
            
            if local_rank == 0:
                print(f"epoch: {epoch + 1}")
                train_tqdm = tqdm(total=len(train_loader), desc="Train")
                            
            model.train()
            for idex, batch in enumerate(train_loader):  # 一共batch_size轮
                if batch == 1:
                    break
                batch = tuple(t.cuda(non_blocking=True) for t in batch)
                frames, text_feature, audio_feature, label = batch
                
                # 清空梯度
                optimizer.zero_grad() 
                    
                # 前向传播计算特征和 logits
                video_projection, text_projection, audio_projection, logits = model(frames, text_feature, audio_feature)

                # 计算损失
                loss, contrastive, cross_entropy = criterion(video_projection, text_projection, audio_projection, logits, label)
                
                # 反向传播计算梯度
                loss.backward()  

                # 更新模型参数 
                optimizer.step()  
                
                if local_rank == 0:
                    train_tqdm.update(1)
                    train_tqdm.set_postfix(lr=get_lr(optimizer))
                
            if local_rank == 0:
                train_tqdm.close()
                                
            if local_rank == 0:
                test_tqdm = tqdm(total=len(test_loader), desc="Test")
            
            model.eval()
            
            # 用于存储每个标签的真实标签和预测标签
            label_true = {label: [] for label in label_map.keys()}
            label_pred = {label: [] for label in label_map.keys()}
            
            for idex, batch in enumerate(test_loader):  # 一共batch_size轮
                batch = tuple(t.cuda(non_blocking=True) for t in batch)
                frames, text_feature, audio_feature, label = batch
                
                # 禁止梯度计算
                with torch.no_grad():
                    # 前向传播计算特征和 logits
                    _, _, _, logits = model(frames, text_feature, audio_feature)

                # 获取预测的标签
                pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
                true_labels = label.cpu().numpy()
                
                # 按标签存储预测和真实标签
                for i in range(len(pred_labels)):
                    true_label = true_labels[i]
                    pred_label = pred_labels[i]
                    label_true[true_label].append(true_label)
                    label_pred[true_label].append(pred_label)
                
                if local_rank == 0:
                    test_tqdm.update(1)
            
            # 计算每个标签的准确率和F1分数
            label_metrics = {}
            for label in label_map.keys():
                # 计算每个标签的准确率
                accuracy = accuracy_score(label_true[label], label_pred[label])
                # 计算每个标签的F1分数
                f1 = f1_score(label_true[label], label_pred[label], average='weighted')
                
                # 汇总信息
                accuracy = reduce_data(accuracy).item()
                f1 = reduce_data(f1).item()
                
                label_metrics[label_map[label]] = {
                    'accuracy': accuracy,
                    'f1': f1
                }
            
                
            # 计算全局的准确率和 F1 分数
            global_accuracy = accuracy_score([label for sublist in label_true.values() for label in sublist], 
                                             [label for sublist in label_pred.values() for label in sublist])
            global_f1 = f1_score([label for sublist in label_true.values() for label in sublist], 
                                 [label for sublist in label_pred.values() for label in sublist], average='weighted')
            
            global_accuracy = reduce_data(global_accuracy).item()
            global_f1 = reduce_data(global_f1).item()
                  
            if local_rank == 0:
                test_tqdm.close()
                
            if global_f1 > best_f1:
                best_f1 = global_f1
                best_acc = global_accuracy
                best_lab = label_metrics
                best_label_true = label_true
                best_label_pred = label_pred
                
                patience_counter = 0           
            else:
                patience_counter += 1
                    
            # 判断是否触发早停
            if patience_counter >= early_stopping_patience:
                early_stop_flag.fill_(1)  # 标志设为 1 (停止)
            
            # 广播早停标志到所有进程
            dist.broadcast(early_stop_flag, src=0)
            
            if local_rank == 0:
                print(f"Accuracy: {global_accuracy:.4f}, F1: {global_f1:.4f}")
            
            # 如果早停标志为 1，所有进程停止训练
            if early_stop_flag.item() == 1:
                if local_rank == 0:
                    print(f"Early stopping triggered.")
                break
        
        # 保存每折的最优结果
        best_accuracies.append(best_acc)
        best_f1_scores.append(best_f1)
        best_label_metrics.append(best_lab)
        
        # 保存每折的最好标签
        if local_rank == 0 and best_label_true is not None and best_label_pred is not None:
            fold_best_labels = {
                'true_labels': best_label_true,
                'pred_labels': best_label_pred
            }
            # 使用 numpy 保存
            np.save(f'../results/IEMOCAP/fold_{fold}_best_labels_by_class.npy', fold_best_labels)
    
    # 计算总体平均值
    avg_accuracy = np.mean(best_accuracies)
    avg_f1 = np.mean(best_f1_scores)
    print(f"Overall Average Accuracy: {avg_accuracy:.4f}")
    print(f"Overall Average F1: {avg_f1:.4f}")

    # 计算每个标签的平均值
    avg_label_metrics = {}
    for label in label_map.values():
        accuracies = [metrics[label]['accuracy'] for metrics in best_label_metrics if label in metrics]
        f1_scores = [metrics[label]['f1'] for metrics in best_label_metrics if label in metrics]
        avg_label_metrics[label] = {
            'accuracy': np.mean(accuracies),
            'f1': np.mean(f1_scores)
        }

    if local_rank == 0:
        results.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # 添加时间戳
                'accuracy': avg_accuracy,
                'f1': avg_f1,
                'classes': avg_label_metrics
            })

        with open(f'../results/{file_name}.json', 'w') as f:
            json.dump(results, f, indent=4)      
    
    dist.destroy_process_group()  # 消除进程组

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-args', help="priority", type=bool, required=False, default=True)
    parser.add_argument('-gpu', default='0,1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('-mode', help="train&test", type=str, required=False, default='train')
    # parser.add_argument('-requires_grad', help="whether to weight_decay", type= bool, required=False, default=True)
    args = parser.parse_args()
    
    os.environ['MASTER_ADDR'] = 'localhost'  # 0号机器的IP
    os.environ['MASTER_PORT'] = '19198'  # 0号机器的可用端口
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用哪些GPU
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 指定程序在分词时不并行执行

    mp.spawn(fn=main, args=(args, ), nprocs=world_size)
