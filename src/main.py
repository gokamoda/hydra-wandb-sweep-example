import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

import wandb


# ===== 簡単な線形回帰データ生成 =====
def make_dataset(n=256):
    x = np.linspace(-1, 1, n).reshape(-1, 1)
    y = 3.0 * x + 0.5 + 0.1 * np.random.randn(n, 1)  # 真の式 y = 3x + 0.5 + ノイズ
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ===== 線形モデル =====
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(config_path="config", config_name="defaults", version_base="1.3")
def run(cfg: DictConfig):
    os.environ.setdefault("WANDB_START_METHOD", cfg.wandb.start_method)
    set_seed(cfg.seed)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=cfg_dict,
    )

    # データ作成
    x, y = make_dataset(cfg.train.n_samples)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True
    )

    # モデル・最適化
    model = SimpleLinear()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.train.lr)
    criterion = nn.MSELoss()

    for epoch in range(cfg.train.epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        wandb.log({"train/loss": loss.item(), "epoch": epoch})

    # 最後に学習したパラメータを記録
    W, b = model.fc.weight.item(), model.fc.bias.item()
    wandb.log({"learned/W": W, "learned/b": b})
    wandb.finish()


if __name__ == "__main__":
    run()
