import torch
import csv


def test(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            # pred是当前batch的预测结果 Tensor(270)
            pred = model(x)
            # preds把每个batch的预测结果分别保存 [Num_of_batch ×Tensor(270)]
            preds.append(pred.detach().cpu())
    #   将所有批次的预测结果在维度 0上拼接成一个完整的张量Tensor(893)，再转为ndarry
    return torch.cat(preds, dim=0).numpy()


def save_pred(preds, file):
    print(f"Saving results to {file}")
    with open(file, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "tested_positive"])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
