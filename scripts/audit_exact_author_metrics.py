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
    print(f"Device: {device}")

    ckpt_path = app_dir / "models/berlin_ii_dno_30events.pt"
    model = DNO(num_channels=5, width=16, initial_step=1, pad=False, factor=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    data_dir = app_dir / "data/converted_30"
    # Test on the valid set (independent events not in the 30 training cases)
    valid_files = sorted((data_dir / "valid").glob("*.pt"))
    train_files = sorted((data_dir / "train").glob("*.pt"))
    all_files = valid_files + train_files # All 36 events

    print(f"Auditing on {len(all_files)} total events ({len(valid_files)} test/valid + {len(train_files)} train)...")

    lploss = utils25.LpLoss(size_average=False)

    # 1. Author's multi-channel metrics (H, U, V combined as in DNO_main.py lines 392-394)
    mc_l2, mc_nse, mc_corr = [], [], []
    # 2. Depth-only metrics (H only)
    h_l2, h_nse, h_corr = [], [], []
    # 3. CSI metrics (as in DNO_main.py lines 395-400)
    csi_1cm, csi_10cm, csi_50cm = [], [], []

    downsample = 4

    with torch.no_grad():
        for pth in all_files:
            sample = load_sample(pth, torch)
            x, horizon = build_model_input(sample, torch)
            y, mask = get_reference(sample, torch, horizon)

            # Downsample to model resolution
            size = (max(96, sample.shape[0] // downsample), max(96, sample.shape[1] // downsample))
            # Resample x
            b, y_dim, xx, t, ti, c = x.shape
            z = x.permute(0, 3, 4, 5, 1, 2).reshape(b, t * ti * c, y_dim, xx)
            z = torch.nn.functional.interpolate(z, size=size, mode="bilinear", align_corners=False)
            x_res = z.reshape(b, t, ti, c, *size).permute(0, 4, 5, 1, 2, 3).contiguous()

            # Resample y and mask
            b, h_dim, w, t, c = y.shape
            z = y.permute(0, 3, 4, 1, 2).reshape(b, t * c, h_dim, w)
            z = torch.nn.functional.interpolate(z, size=size, mode="bilinear", align_corners=False)
            y_res = z.reshape(b, t, c, *size).permute(0, 3, 4, 1, 2).contiguous()

            m = mask.float().permute(0, 3, 4, 1, 2).reshape(b, t * c, h_dim, w)
            m = torch.nn.functional.interpolate(m, size=size, mode="nearest")
            mask_res = m.reshape(b, t, c, *size).permute(0, 3, 4, 1, 2).bool().contiguous()

            pred = model(x_res.to(device)).cpu()

            # EXACT code from DNO_main.py lines 376 & 390:
            yy = y_res * mask_res
            pred = pred * mask_res

            # Reshape for multi-channel (B, -1, 3)
            pred_mc = pred.reshape(len(pred), -1, 3)
            yy_mc = yy.reshape(len(yy), -1, 3)

            val_mc_l2 = lploss(pred_mc, yy_mc).item()
            val_mc_nse = utils25.nse(pred_mc, yy_mc).item()
            val_mc_corr = utils25.corr(pred_mc, yy_mc).item()

            mc_l2.append(val_mc_l2)
            mc_nse.append(val_mc_nse)
            mc_corr.append(val_mc_corr)

            # Reshape for Depth only (B, -1, 1)
            pred_h = pred[..., 0:1].reshape(len(pred), -1, 1)
            yy_h = yy[..., 0:1].reshape(len(yy), -1, 1)

            val_h_l2 = lploss(pred_h, yy_h).item()
            val_h_nse = utils25.nse(pred_h, yy_h).item()
            val_h_corr = utils25.corr(pred_h, yy_h).item()

            h_l2.append(val_h_l2)
            h_nse.append(val_h_nse)
            h_corr.append(val_h_corr)

            # CSI EXACTLY as DNO_main.py lines 395-400
            c1 = utils25.critical_success_index(pred_h, yy_h, 0.01).item()
            c2 = utils25.critical_success_index(pred_h, yy_h, 0.10).item()
            c3 = utils25.critical_success_index(pred_h, yy_h, 0.50).item()

            csi_1cm.append(c1)
            csi_10cm.append(c2)
            csi_50cm.append(c3)

    print("\n================ EXACT AUTHOR IMPLEMENTATION AUDIT (DNO_main.py) ================")
    print("1. CACH DO CHINH THUC CUA BAI BAO (Gop ca 3 kenh H, U, V nhu DNO_main.py dong 392-394):")
    print(f"   - Relative L2 Error (LpLoss) : {np.mean(mc_l2)*100:.2f}%  (Paper claim: ~20.0%)")
    print(f"   - NSE (Nash-Sutcliffe)       : {np.mean(mc_nse):.4f}     (Paper claim: 0.88 - 0.94)")
    print(f"   - Pearson Correlation (Corr) : {np.mean(mc_corr)*100:.2f}%  (Paper claim: 93.0%)")

    print("\n2. CACH DO BOC TACH RIENG KENH DO SAU NGAP (Chi kenh H):")
    print(f"   - Relative L2 Error (H only) : {np.mean(h_l2)*100:.2f}%")
    print(f"   - NSE (H only)               : {np.mean(h_nse):.4f}")
    print(f"   - Pearson Corr (H only)      : {np.mean(h_corr)*100:.2f}%")

    print("\n3. DO CHINH XAC RANH GIOI NGAP (CSI - DNO_main.py dong 395-400, kenh H):")
    print(f"   - CSI @ 1 cm  (Ngap nong)    : {np.mean(csi_1cm)*100:.2f}%  (Paper claim: 87.5%)")
    print(f"   - CSI @ 10 cm (Ngap duong)   : {np.mean(csi_10cm)*100:.2f}%  (Paper claim: 70.0%)")
    print(f"   - CSI @ 50 cm (Ngap sau)     : {np.mean(csi_50cm)*100:.2f}%  (Paper claim: 45.0%)")
    print("==================================================================================")

if __name__ == "__main__":
    main()
