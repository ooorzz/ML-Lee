import os
import torch
from src.dataset import prep_dataloader
from src.model import NeuralNet
from src.train import train, dev
from src.test import test, save_pred
from src.utils import plot_learning_curve, plot_pred

# Set seed and paths
myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

device = "cuda" if torch.cuda.is_available() else "cpu"
tr_path = "data/covid.train.csv"
tt_path = "data/covid.test.shuffle.csv"
os.makedirs("models", exist_ok=True)

# Hyperparameters
config = {
    "n_epochs": 3000,
    "batch_size": 270,
    "optimizer": "SGD",
    "optim_hparas": {"lr": 0.001, "momentum": 0.9},
    "early_stop": 200,
    "save_path": "models/model.pth",
}

# Load data
# feature_select的选项：Correlation, Lasso, KBest+f, KBest+mutual, None(默认)
tr_set, selected_feats = prep_dataloader(
    tr_path, "train", config["batch_size"], feature_select="Correlation"
)
dv_set = prep_dataloader(tr_path, "dev", config["batch_size"], feats=selected_feats)
tt_set = prep_dataloader(tt_path, "test", config["batch_size"], feats=selected_feats)

# Train model
model = NeuralNet(tr_set.dataset.dim).to(device)
# 这里返回的 model_loss是最小的验证 MSE
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
plot_learning_curve(model_loss_record, title="deep model")

# Load best model and make predictions
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config["save_path"], map_location="cpu")
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)

# Test and save predictions
preds = test(tt_set, model, device)
save_pred(preds, "pred.csv")
