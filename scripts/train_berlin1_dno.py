import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

app_dir = Path("C:/Users/admin/Documents/Codex/2026-10-08/ngh/outputs/urbanfloodcast_app")
repo_dno = Path("C:/Users/admin/Documents/Codex/2026-10-08/ngh/work/UrbanFloodCast/DNO")
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(repo_dno))

from models.DNO import DNO
from p2m_data import build_model_input, get_reference, load_sample

def resample_spatiotemporal(x, size):
    b, y, xx, t, ti, c = x.shape
    z = x.permute(0, 3, 4, 5, 1, 2).reshape(b, t * ti * c, y, xx)
    z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
    return z.reshape(b, t, ti, c, *size).permute(0, 4, 5, 1, 2, 3).contiguous()

def resample_target(y, size, mask=None):
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

def load_case(path, downsample=4):
    sample = load_sample(path, torch)
    x, horizon = build_model_input(sample, torch)
    y, mask = get_reference(sample, torch, horizon)
    if downsample > 1:
        size = (max(96, sample.shape[0] // downsample), max(96, sample.shape[1] // downsample))
        x = resample_spatiotemporal(x, size)
        y, mask = resample_target(y, size, mask)
    return x, y, mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data_dir = app_dir / "data/berlin1_converted"
    train_files = sorted((data_dir / "train").glob("*.pt"))
    test_files = sorted((data_dir / "test").glob("*.pt"))

    print(f"Berlin I: Found {len(train_files)} training cases and {len(test_files)} test cases.")
    if len(train_files) < 10:
        print("Waiting for data conversion to complete...")
        return

    print("Pre-loading Berlin I tensors into RAM...")
    downsample = 4
    train_data = []
    for pth in train_files:
        x, y, mask = load_case(pth, downsample=downsample)
        train_data.append((x, y, mask))

    test_data = []
    for pth in test_files:
        x, y, mask = load_case(pth, downsample=downsample)
        test_data.append((x, y, mask))
    print("Pre-loading complete.")

    model = DNO(num_channels=5, width=16, initial_step=1, pad=False, factor=1).to(device)
    print(f"Initialized DNO model (width=16, params={sum(p.numel() for p in model.parameters())})")

    lr = 2e-4
    epochs = 80
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_test_loss = float("inf")
    save_path = app_dir / "models/berlin_i_dno_30events.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting training on Berlin I for {epochs} epochs...")
    start_time = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for x, y, mask in train_data:
            x_b, y_b, mask_b = x.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_b)

            diff_sq = (pred - y_b).square()
            loss_domain = (diff_sq * mask_b).sum() / mask_b.sum().clamp_min(1)
            
            wet_mask = mask_b & (y_b[..., 0:1] > 0.01)
            loss_wet = (diff_sq[..., 0:1] * wet_mask).sum() / wet_mask.sum().clamp_min(1) if wet_mask.any() else 0.0

            loss = loss_domain + 2.0 * loss_wet
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += float(loss.item())

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_data)

        # Evaluate on test set
        model.eval()
        total_test_loss = 0.0
        with torch.inference_mode():
            for x, y, mask in test_data:
                x_b, y_b, mask_b = x.to(device), y.to(device), mask.to(device)
                pred = model(x_b)
                loss_v = ((pred - y_b).square() * mask_b).sum() / mask_b.sum().clamp_min(1)
                total_test_loss += float(loss_v.item())

        avg_test_loss = total_test_loss / len(test_data)

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            elapsed = time.time() - start_time
            print(f"Epoch [{ep:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.5f} | Test Loss: {avg_test_loss:.5f} | Time: {elapsed:.1f}s")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), save_path)

    total_time = time.time() - start_time
    print(f"Berlin I training finished in {total_time:.1f}s. Best Test Loss: {best_test_loss:.5f}")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    main()
