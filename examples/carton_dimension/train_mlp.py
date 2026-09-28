"""Train a 9-feature MLP from an .npz dataset: features (N,9), targets (N,3)."""
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="npz with features and targets; targets use mm or your chosen unit")
parser.add_argument("--output", default="carton_mlp.pt")
parser.add_argument("--epochs", type=int, default=300)
args = parser.parse_args()
data = np.load(args.dataset)
x, y = data["features"].astype("float32"), data["targets"].astype("float32")
if x.ndim != 2 or x.shape[1] != 9 or y.shape != (len(x), 3):
    raise SystemExit("Expected features with shape (N, 9) and targets with shape (N, 3).")
mean, std = x.mean(0), x.std(0).clip(1e-6)
target_scale = y.std(0).clip(1e-6)
model = torch.nn.Sequential(torch.nn.Linear(9, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
inputs, targets = torch.tensor((x - mean) / std), torch.tensor(y / target_scale)
for _ in range(args.epochs):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
torch.save({"model": model.state_dict(), "feature_mean": torch.tensor(mean), "feature_std": torch.tensor(std), "target_scale": torch.tensor(target_scale)}, args.output)
print(f"saved {args.output}; final normalized MSE={loss.item():.6f}")
