import torch


def train(tr_set, dv_set, model, config, device):
    n_epochs = config["n_epochs"]
    # getattr(object,string)是用于获取object名为string的属性的
    # **config["optim_hparas"]是用于将字典config["optim_hparas"]展开作为关键词传递给优化器
    # 总之这行是用来初始化优化器，相当于
    #   optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 其中model.parameters()是前面定义的模型中的所有可优化参数（应该就是两个线性层的矩阵）
    optimizer = getattr(torch.optim, config["optimizer"])(
        model.parameters(), **config["optim_hparas"]
    )
    min_mse = 1000.0
    loss_record = {"train": [], "dev": []}
    early_stop_cnt = 0
    epoch = 0

    while epoch < n_epochs:
        model.train()
        # tr_set是DataLoader类型的，会自动分好batch
        # 这里x,y就分别是一个batch的data和target
        # x: Tensor(270,93) y: Tensor(270)
        for x, y in tr_set:
            # 把loss对每个可训练参数的梯度重置为0（因为Pytorch默认在每次 backward()调用时将梯度累加到现有梯度上
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            # 前向传播：根据现有的model的参数，输入x，得到模型预测结果pred
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
            # 反向传播：计算loss函数对每一个可训练参数的梯度，并保存在每个参数的.grad属性里
            mse_loss.backward()
            # 根据计算出的梯度，根据优化方法，更新所有参数
            optimizer.step()
            # detach()的作用：从计算图中分离出这个张量，防止它参与后续的梯度计算
            #               Pytorch中所有张量默认都在计算图里，就会跟踪其梯度信息、用于反向传播
            #               这里我们只需要拿到loss的值，不需要它参与梯度计算，遂移除
            # .cpu()的作用：把mse_loss移动到CPU上
            #              如果训练在GPU上进行，则 mse_loss是以张量形式存储的，需要将其转换为常规的 Python 数值(如int,float)才能在CPU上处理
            # item()的作用：将张量转换为 Python 标量值
            loss_record["train"].append(mse_loss.detach().cpu().item())

        dev_mse = dev(dv_set, model, device)
        # min_mse存储训练过程中最小的验证损失 eval时最小的MSE
        if dev_mse < min_mse:
            min_mse = dev_mse
            # 将最小eval损失对应的model保存下来、作为最终用于测试的model，并且暂停early_stop的计数
            print(f"Saving model (epoch = {epoch + 1:4d}, loss = {min_mse:.4f})")
            torch.save(model.state_dict(), config["save_path"])
            early_stop_cnt = 0
        else:
            # 如果验证损失没有变小，则早停计数+1
            early_stop_cnt += 1

        epoch += 1
        loss_record["dev"].append(dev_mse)
        # 如果 early_stop_cnt 超过了配置中指定的早停次数 config["early_stop"]，说明模型已经在多个 epoch 中没有提升，因此终止训练。
        if early_stop_cnt > config["early_stop"]:
            break

    print(f"Finished training after {epoch} epochs")
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        # torch.no_grad()禁用梯度计算，eval过程中参数既定、无需计算梯度来更新
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        # 这里是根据单个batch的MSE计算出其总SE，再加到 total_loss中，最后所有batch一起算平均SE
        # total_loss: 所有batch的总方差
        total_loss += mse_loss.detach().cpu().item() * len(x)
    return total_loss / len(dv_set.dataset)
