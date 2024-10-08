import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch

def plot_learning_curve(loss_record, title=""):
    total_steps = len(loss_record["train"])
    x_1 = range(total_steps)
    x_2 = x_1[:: len(loss_record["train"]) // len(loss_record["dev"])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record["train"], c="tab:red", label="train")
    plt.plot(x_2, loss_record["dev"], c="tab:cyan", label="dev")
    plt.ylim(0.0, 5.0)
    plt.xlabel("Training steps")
    plt.ylabel("MSE loss")
    plt.title(f"Learning curve of {title}")
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35.0, preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c="r", alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c="b")
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel("ground truth value")
    plt.ylabel("predicted value")
    plt.title("Ground Truth vs. Prediction")
    plt.show()
