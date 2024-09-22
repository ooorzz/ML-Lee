import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


class COVID19Dataset(Dataset):
    """Dataset for loading and preprocessing the COVID19 dataset."""

    def __init__(self, path, mode="train", feature_select="None", feats=[]):
        self.mode = mode
        self.feats = feats
        # train模式下不会有feats传入，而是需要自己计算
        if mode == "train":
            df = pd.read_csv(path, index_col=0)
            feature_list = list(df.columns)
            # 因为下面要只针对train set进行特征选取的计算，所以先分出train和val
            df = df[df.index % 10 != 0]

            X = df.drop("tested_positive.2", axis=1)
            y = df["tested_positive.2"]

            # Feature Selection
            if feature_select == "Correlation":
                correlation_matrix = df.corr()
                # 筛选与目标相关性大于0.8的
                filtered_target_corr = correlation_matrix[
                    correlation_matrix["tested_positive.2"] > 0.8
                ]
                corr_selected_features = list(filtered_target_corr.index)
                corr_selected_features.remove("tested_positive.2")
                feats = [feature_list.index(item) for item in corr_selected_features]
                print("用与目标变量的相关性选定的特征：\n", feats)

            elif feature_select == "Lasso":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                # 创建LassoCV实例，使用交叉验证来选择最佳的alpha
                lasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train)

                # 通过检查系数不为零的特征来进行特征选择
                lasso_selected_features = [
                    feature
                    for feature, coef in zip(X.columns, lasso.coef_)
                    if coef != 0
                ]
                feats = [feature_list.index(item) for item in lasso_selected_features]
                # 输出选择的特征
                print("Lasso选中的特征:\n", feats)

            elif feature_select == "KBest+f":
                # 创建 SelectKBest 对象，选择最佳的10个特征
                selector_f_reg = SelectKBest(f_regression, k=15)
                # 拟合并转换数据
                X_new_f = selector_f_reg.fit_transform(X, y)
                feats = list(selector_f_reg.get_support(indices=True))
                print("f_regression选中的特征:\n", feats)

            elif feature_select == "KBest+mutual":
                selector_mutual = SelectKBest(mutual_info_regression, k=15)
                X_new_mutual = selector_mutual.fit_transform(X, y)
                feats = list(selector_mutual.get_support(indices=True))
                print("mutual_info_regression选中的特征:\n", feats)

            else:
                feats = list(range(93))
            # 把选好的feature更新到self.feats中
            self.feats = feats

            selected_df = df.reset_index(drop=True)
            selected_X = selected_df.iloc[:, feats]
            # 交叉5重验证，判断特征选择的泛化性如何
            kf = KFold(n_splits=5)
            model = LinearRegression()
            KF_MSE_list = []
            for train_index, test_index in kf.split(selected_X):
                X_train, X_test = (
                    selected_X.iloc[train_index],
                    selected_X.iloc[test_index],
                )  # 按行索引选择
                y_train, y_test = (
                    y.iloc[train_index],
                    y.iloc[test_index],
                )  # 同样对 y 使用 iloc
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                KF_MSE_list.append(mean_squared_error(y_test, predictions))
            mean_mse = np.mean(KF_MSE_list)
            std_mse = np.std(KF_MSE_list)

            print(f"KF Mean MSE: {mean_mse}")
            print(f"KF Standard Deviation of MSE: {std_mse}")

            data = df.to_numpy().astype(float)
            # train的时候有真值，这里target就是取的所有样本的真值
            target = data[:, -1]
            # 让data只保留feature部分、不含target
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
            self.target = torch.FloatTensor(target)
            # # 所有样本的index（去掉是10的倍数的部分）
            # # 这里是用来划分训练集和验证集的，验证集就是10的倍数的那部分
            # indices = [i for i in range(len(data)) if i % 10 != 0]
            # # training set的Tensor: [2430,93]  validation: [270,93]
            # self.data = torch.FloatTensor(data[indices])
            # # train: [2430] val:[230]
            # self.target = torch.FloatTensor(target[indices])

        elif mode == "dev":
            with open(path, "r") as fp:
                # 以列表的形式先将数据集读入 [2701×[95]]
                data = list(csv.reader(fp))
                # 把第一行的属性名称去掉，同时把第一列的index去掉 -> [2700,94]
                data = np.array(data[1:])[:, 1:].astype(float)
            target = data[:, -1]
            data = data[:, feats]
            # 取index为10的倍数的部分作为val set
            indices = [i for i in range(len(data)) if i % 10 == 0]
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        elif mode == "test":
            with open(path, "r") as fp:
                # 以列表的形式先将数据集读入 [2701×[95]]
                data = list(csv.reader(fp))
                # 把第一行的属性名称去掉，同时把第一列的index去掉 -> [2700,94]
                data = np.array(data[1:])[:, 1:].astype(float)
            data = data[:, feats]
            # Tensor: [893,93]
            self.data = torch.FloatTensor(data)

        # else:
        #     # train的时候有真值，这里target就是取的所有样本的真值
        #     target = data[:, -1]
        #     # 让data只保留feature部分、不含target
        #     data = data[:, feats]
        #     if mode == "train":
        #         # 所有样本的index（去掉是10的倍数的部分）
        #         # 这里是用来划分训练集和验证集的，验证集就是10的倍数的那部分
        #         indices = [i for i in range(len(data)) if i % 10 != 0]
        #     elif mode == "dev":
        #         indices = [i for i in range(len(data)) if i % 10 == 0]
        #     # training set的Tensor: [2430,93]  validation: [270,93]
        #     self.data = torch.FloatTensor(data[indices])
        #     # train: [2430] val:[230]
        #     self.target = torch.FloatTensor(target[indices])

        # 进行标准化处理
        self.data = (self.data - self.data.mean(dim=0, keepdim=True)) / self.data.std(
            dim=0, keepdim=True
        )
        self.dim = self.data.shape[1]  # dim:93
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


def prep_dataloader(path, mode, batch_size, n_jobs=0, feature_select="None", feats=[]):
    dataset = COVID19Dataset(
        path, mode=mode, feature_select=feature_select, feats=feats
    )
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
    if mode == "train":
        return dataloader, dataset.feats
    else:
        return dataloader
