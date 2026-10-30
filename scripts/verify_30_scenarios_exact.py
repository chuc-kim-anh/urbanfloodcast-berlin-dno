import os
import sys
import torch
import numpy as np
from pathlib import Path

app_dir = Path("C:/Users/admin/Documents/Codex/2026-10-08/ngh/outputs/urbanfloodcast_app")
repo_dno = Path("C:/Users/admin/Documents/Codex/2026-10-08/ngh/work/UrbanFloodCast/DNO")
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(repo_dno))

from models.DNO import DNO
from p2m_data import build_model_input, get_reference, load_sample
import utils25

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = app_dir / "models/berlin_ii_dno_30events.pt"
    
    model = DNO(num_channels=5, width=16, initial_step=1, pad=False, factor=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    data_dir = app_dir / "data/converted_30"
    train_files = sorted((data_dir / "train").glob("*.pt"))[:30] # Exactly 30 scenarios
    ntest = len(train_files)
    print(f"Testing on exactly {ntest} benchmark scenarios (10 EulerII, 10 EulerIII, 10 Rand)...")

    lploss = utils25.LpLoss(size_average=False)

    total_l2_mc = 0.0
    total_nse_mc = 0.0
    total_corr_mc = 0.0

    total_l2_h = 0.0
    total_nse_h = 0.0
    total_corr_h = 0.0

    total_csi_1 = 0.0
    total_csi_2 = 0.0
    total_csi_3 = 0.0

    downsample = 4

    with torch.no_grad():
        for pth in train_files:
            sample = load_sample(pth, torch)
            x, horizon = build_model_input(sample, torch)
            y, mask = get_reference(sample, torch, horizon)

            size = (max(96, sample.shape[0] // downsample), max(96, sample.shape[1] // downsample))
            b, y_dim, xx, t, ti, c = x.shape
            z = x.permute(0, 3, 4, 5, 1, 2).reshape(b, t * ti * c, y_dim, xx)
            z = torch.nn.functional.interpolate(z, size=size, mode="bilinear", align_corners=False)
            x_res = z.reshape(b, t, ti, c, *size).permute(0, 4, 5, 1, 2, 3).contiguous()

            b, h_dim, w, t, c = y.shape
            z = y.permute(0, 3, 4, 1, 2).reshape(b, t * c, h_dim, w)
            z = torch.nn.functional.interpolate(z, size=size, mode="bilinear", align_corners=False)
            y_res = z.reshape(b, t, c, *size).permute(0, 3, 4, 1, 2).contiguous()

            m = mask.float().permute(0, 3, 4, 1, 2).reshape(b, t * c, h_dim, w)
            m = torch.nn.functional.interpolate(m, size=size, mode="nearest")
            mask_res = m.reshape(b, t, c, *size).permute(0, 3, 4, 1, 2).bool().contiguous()

            pred = model(x_res.to(device)).cpu()

            # EXACT code from DNO_main.py lines 376 & 390
            yy = y_res * mask_res
            pred = pred * mask_res

            # Multi-channel (H, U, V)
            pred_mc = pred.reshape(len(pred), -1, 3)
            yy_mc = yy.reshape(len(yy), -1, 3)

            total_l2_mc += lploss(pred_mc, yy_mc).item()
            total_nse_mc += utils25.nse(pred_mc, yy_mc).item()
            total_corr_mc += utils25.corr(pred_mc, yy_mc).item()

            # Depth-only (H)
            pred_h = pred[..., 0:1].reshape(len(pred), -1, 1)
            yy_h = yy[..., 0:1].reshape(len(yy), -1, 1)

            total_l2_h += lploss(pred_h, yy_h).item()
            total_nse_h += utils25.nse(pred_h, yy_h).item()
            total_corr_h += utils25.corr(pred_h, yy_h).item()

            # CSI thresholds
            total_csi_1 += utils25.critical_success_index(pred_h, yy_h, 0.01).item()
            total_csi_2 += utils25.critical_success_index(pred_h, yy_h, 0.10).item()
            total_csi_3 += utils25.critical_success_index(pred_h, yy_h, 0.50).item()

    print("\n--- RESULTS ACCORDING TO EXACT DNO_main.py FORMULA (30 SCENARIOS) ---")
    print(f"Test_l2 (Multi-channel):   {total_l2_mc / ntest * 100:.2f}%")
    print(f"Test_nse (Multi-channel):  {total_nse_mc / ntest:.4f}")
    print(f"Test_corr (Multi-channel): {total_corr_mc / ntest * 100:.2f}%")
    print(f"Test_csi_1 (@1cm):         {total_csi_1 / ntest * 100:.2f}%")
    print(f"Test_csi_2 (@10cm):        {total_csi_2 / ntest * 100:.2f}%")
    print(f"Test_csi_3 (@50cm):        {total_csi_3 / ntest * 100:.2f}%")
    print("\n--- RESULTS FOR DEPTH (H) ALONE ---")
    print(f"Test_l2 (H only):          {total_l2_h / ntest * 100:.2f}%")
    print(f"Test_nse (H only):         {total_nse_h / ntest:.4f}")
    print(f"Test_corr (H only):        {total_corr_h / ntest * 100:.2f}%")
    print("----------------------------------------------------------------------")

if __name__ == "__main__":
    main()
