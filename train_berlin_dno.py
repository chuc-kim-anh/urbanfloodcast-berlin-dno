"""Train the public UrbanFloodCast DNO on the prepared Berlin II pilot split."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from p2m_data import build_model_input, get_reference, load_sample


def resample_spatiotemporal(x, size):
    # x: [B,Y,X,T,1,C] -> [B,Y',X',T,1,C]
    b, y, xx, t, ti, c = x.shape
    z = x.permute(0, 3, 4, 5, 1, 2).reshape(b, t * ti * c, y, xx)
    z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
    return z.reshape(b, t, ti, c, *size).permute(0, 4, 5, 1, 2, 3).contiguous()


def resample_target(y, size, mask=None):
    # y/mask: [B,Y,X,T,C]
    b, h, w, t, c = y.shape
    z = y.permute(0, 3, 4, 1, 2).reshape(b, t * c, h, w)
    z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
    y2 = z.reshape(b, t, c, *size).permute(0, 3, 4, 1, 2).contiguous()
    if mask is None:
        return y2
    m = mask.float().permute(0, 3, 4, 1, 2).reshape(b, t * c, h, w)
    m = F.interpolate(m, size=size, mode="nearest")
    m = m.reshape(b, t, c, *size).permute(0, 3, 4, 1, 2).bool().contiguous()
    return y2, m


def load_case(path, downsample):
    sample = load_sample(path, torch)
    x, horizon = build_model_input(sample, torch)
    y, mask = get_reference(sample, torch, horizon)
    if downsample > 1:
        size = (max(96, sample.shape[0] // downsample), max(96, sample.shape[1] // downsample))
        x = resample_spatiotemporal(x, size)
        y, mask = resample_target(y, size, mask)
    return x, y, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/converted")
    p.add_argument("--output", default="models/berlin_ii_dno.pt")
    p.add_argument("--repo", default="vendor/UrbanFloodCast")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--downsample", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    args = p.parse_args()
    repo_dno = (Path(args.repo).resolve() / "DNO")
    if not (repo_dno / "models" / "DNO.py").exists():
        raise SystemExit(f"Missing official DNO code: {repo_dno}")
    sys.path.insert(0, str(repo_dno))
    from models.DNO import DNO

    root = Path(args.data)
    train_files = sorted((root / "train").glob("*.pt"))
    valid_files = sorted((root / "valid").glob("*.pt"))
    if len(train_files) < 2 or not valid_files:
        raise SystemExit(f"Need >=2 train and >=1 validation cases; found {len(train_files)} and {len(valid_files)}")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNO(num_channels=5, width=10, initial_step=1, pad=False, factor=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device}; train={len(train_files)}; valid={len(valid_files)}; epochs={args.epochs}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for path in train_files:
            x, y, mask = load_case(path, args.downsample)
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            if pred.shape != y.shape:
                raise RuntimeError(f"DNO output {tuple(pred.shape)} != target {tuple(y.shape)}")
            loss = ((pred - y).square() * mask).sum() / mask.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu())

        model.eval()
        valid_loss = 0.0
        with torch.inference_mode():
            for path in valid_files:
                x, y, mask = load_case(path, args.downsample)
                pred = model(x.to(device)).cpu()
                valid_loss += float((((pred - y).square()) * mask).sum() / mask.sum().clamp_min(1))
        train_loss /= len(train_files)
        valid_loss /= len(valid_files)
        print(f"epoch={epoch}/{args.epochs} train_mse={train_loss:.6f} valid_mse={valid_loss:.6f}", flush=True)
        if valid_loss < best:
            best = valid_loss
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "valid_mse": best,
                        "architecture": "UrbanFloodCast DNO", "downsample_train": args.downsample}, output)
    print(f"saved={output.resolve()} best_valid_mse={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
