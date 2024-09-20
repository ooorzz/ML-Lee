import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv


class COVID19Dataset(Dataset):
    """Dataset for loading and preprocessing the COVID19 dataset."""

    def __init__(self, path, mode="train", target_only=False):
        self.mode = mode
        with open(path, "r") as fp:
            # 以列表的形式先将数据集读入 [2701×[95]]
            data = list(csv.reader(fp))
            # 把第一行的属性名称去掉，同时把第一列的index去掉 -> [2700,94]
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            # 就是一个0~92的list
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            pass

        if mode == "test":
            data = data[:, feats]
            # Tensor: [893,93]
            self.data = torch.FloatTensor(data)
        else:
            # train的时候有真值，这里target就是取的所有样本的真值
            target = data[:, -1]
            # 让data只保留feature部分、不含target
            data = data[:, feats]
            if mode == "train":
                # 所有样本的index（去掉是10的倍数的部分）
                # 这里是用来划分训练集和验证集的，验证集就是10的倍数的那部分
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == "dev":
                indices = [i for i in range(len(data)) if i % 10 == 0]
            # training set的Tensor: [2430,93]  validation: [270,93]
            self.data = torch.FloatTensor(data[indices])
            # train: [2430] val:[230]
            self.target = torch.FloatTensor(target[indices])
        # 对除了前40个用于表示州编号的feature以外的所有feature进行标准化处理
        self.data[:, 40:] = (
            self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)
        ) / self.data[:, 40:].std(dim=0, keepdim=True)
        self.dim = self.data.shape[1] # dim:93
        print(
            f"Finished reading the {mode} set of COVID19 Dataset ({len(self.data)} samples found, each dim = {self.dim})"
        )

    def __getitem__(self, index):
        if self.mode in ["train", "dev"]:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)
    # dataset由两部分 dataset.data和 dataset.target组成
    # DataLoader用于将数据加载到模型中
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),  # 是否打乱数据集
        drop_last=False,  # 当数据集总数不能被batch_size整除时，是否去掉分出的最后一个batch
        num_workers=n_jobs,  # n_jobs>=1的时候多线程处理
        pin_memory=True,  # 是否存储于固定内存
    )
    return dataloader
