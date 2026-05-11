import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset

random.seed(0)

class UserItemRatingDataset(Dataset):
    """
    包装器：将 <user, item, rating, weight> 张量转换为 PyTorch 数据集
    
    """

    def __init__(self, user_tensor, item_tensor, target_tensor, weight_tensor=None):
        """
        参数:
            user_tensor: torch.Tensor, 用户 ID 列表
            item_tensor: torch.Tensor, 物品 ID 列表
            target_tensor: torch.Tensor, 目标评分 (0 或 1)
            weight_tensor: torch.Tensor, 可选，每个交互的停留时长权重 
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor
        
        # 【修改点】：如果未提供权重（如运行原版 CiteULike），则初始化全为 1 的权重
        if weight_tensor is None:
            self.weight_tensor = torch.ones_like(self.target_tensor, dtype=torch.float32)
        else:
            self.weight_tensor = weight_tensor

    def __getitem__(self, index):
        """
        返回四个维度的信息，供训练引擎使用 
        """
        return (self.user_tensor[index], 
                self.item_tensor[index], 
                self.target_tensor[index], 
                self.weight_tensor[index])

    def __len__(self):
        return self.user_tensor.size(0)