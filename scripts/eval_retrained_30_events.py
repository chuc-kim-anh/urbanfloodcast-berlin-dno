import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import zoom
import torch

app_dir = Path("C:/Users/admin/Documents/Codex/2026-10-08/ngh/outputs/urbanfloodcast_app")
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(Path("C:/Users/admin/Documents/Codex/2026-10-08/ngh/work/UrbanFloodCast/DNO")))

from p2m_data import model_grid_size, resample_model_input, resample_prediction
from models.DNO import DNO

def read_rain(event_dir: Path, ntime: int = 25, dt_seconds: int = 300) -> np.ndarray:
    rain = np.zeros(ntime, dtype=np.float32)
    candidates = list(event_dir.glob("*.txt"))
    if not candidates:
        return rain
    for line in candidates[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            idx = int(float(parts[0]) / dt_seconds)
            val = float(parts[1])
        except ValueError:
            continue
        if 0 <= idx < ntime:
            rain[idx] = val
    return rain

def main():
    print("================================================================================")
    print("      EVALUATING RETRAINED 30-EVENT DNO MODEL VS MIKE 21 GROUND TRUTH           ")
    print("================================================================================")

    dataset_root = Path("C:/Users/admin/Downloads/UrbanFloodCast_Dataset/UrbanFloodCast_Dataset/BerlinII")
    ckpt_path = app_dir / "models/berlin_ii_dno_30events.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} does not exist yet. Please train first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNO(num_channels=5, width=16, initial_step=1, pad=False, factor=1).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded retrained model from {ckpt_path.name}")

    target_shape = (433, 692)
    dem_raw_path = dataset_root / "Input_BerlinII/moa_bottom.tif"
    bld_raw_path = dataset_root / "Input_BerlinII/moa_building.tif"

    dem_raw = tifffile.imread(dem_raw_path).astype(np.float32)
    dem_raw = np.nan_to_num(dem_raw, nan=-99999, posinf=-99999, neginf=-99999)
    nodata = dem_raw < -2000
    if nodata.any():
        dem_raw[nodata] = float(np.max(dem_raw[~nodata]) + 30)
    dem_res = zoom(dem_raw, (target_shape[0] / dem_raw.shape[0], target_shape[1] / dem_raw.shape[1]), order=1).astype(np.float32)[:target_shape[0], :target_shape[1]]

    bld_raw = tifffile.imread(bld_raw_path).astype(np.float32)
    bld_res = zoom(bld_raw, (target_shape[0] / bld_raw.shape[0], target_shape[1] / bld_raw.shape[1]), order=0)[:target_shape[0], :target_shape[1]] > 0.5

    # 36 test events
    euler2 = sorted([p for p in (dataset_root / "EulerII").glob("*/*") if p.is_dir() and list(p.glob("*_0H.tif"))])[:12]
    euler3 = sorted([p for p in (dataset_root / "EulerIII").glob("*/*") if p.is_dir() and list(p.glob("*_0H.tif"))])[:12]
    rand = sorted([p for p in (dataset_root / "Rand").glob("*") if p.is_dir() and list(p.glob("*_0H.tif"))])[:12]
    all_events = euler2 + euler3 + rand

    horizon = 24
    inf_size = model_grid_size(*target_shape, factor=4)
    dem_t = torch.from_numpy(dem_res).unsqueeze(2).repeat(1, 1, horizon).unsqueeze(-1)
    dem_t = torch.nn.functional.normalize(dem_t).unsqueeze(-1)

    csi_1_masked, csi_10_masked, csi_50_masked = [], [], []
    corrs_masked, nses_masked, rmses_masked, l2s_masked = [], [], [], []

    for idx, event_dir in enumerate(all_events, 1):
        name = event_dir.name
        rain_series = read_rain(event_dir, ntime=25, dt_seconds=300)

        h0 = np.nan_to_num(tifffile.imread(event_dir / f"{name}_0H.tif").astype(np.float32), nan=0.0)
        u0 = np.nan_to_num(tifffile.imread(event_dir / f"{name}_0U.tif").astype(np.float32), nan=0.0)
        v0 = np.nan_to_num(tifffile.imread(event_dir / f"{name}_0V.tif").astype(np.float32), nan=0.0)
        state0 = np.stack([h0, u0, v0], axis=-1)

        state = torch.from_numpy(state0).unsqueeze(2).repeat(1, 1, horizon, 1).unsqueeze(-2)
        rain_t = torch.from_numpy(rain_series[1:25]).view(1, 1, horizon, 1).expand(target_shape[0], target_shape[1], -1, -1)
        rain_t = torch.log1p(torch.clamp(rain_t, min=0) / 1e-2) / 10.0
        rain_t = rain_t.unsqueeze(-1)

        x = torch.cat((state, rain_t, dem_t), dim=-1).unsqueeze(0).float()
        x_model = resample_model_input(x, torch, inf_size)

        with torch.inference_mode():
            pred = model(x_model.to(device))
        pred = resample_prediction(pred[0].cpu(), torch, target_shape)

        p_h = np.maximum(pred[..., 0].numpy(), 0)
        p_h[bld_res] = 0.0

        t_h = np.zeros((target_shape[0], target_shape[1], horizon), dtype=np.float32)
        domain_mask = None

        for s_idx, sec in enumerate(range(300, 7500, 300)):
            raw_raster = tifffile.imread(event_dir / f"{name}_{sec}H.tif").astype(np.float32)
            if domain_mask is None:
                domain_mask = ~np.isnan(raw_raster)
            h_gt = np.nan_to_num(raw_raster, nan=0.0)
            h_gt[bld_res] = 0.0
            t_h[..., s_idx] = np.maximum(h_gt, 0.0)

        # Apply official domain mask
        domain_3d = np.repeat(domain_mask[:, :, np.newaxis], horizon, axis=2)
        p_m = p_h[domain_3d]
        t_m = t_h[domain_3d]

        # CSI
        for th, arr in [(0.01, csi_1_masked), (0.10, csi_10_masked), (0.50, csi_50_masked)]:
            pb = p_m >= th
            tb = t_m >= th
            tp = np.sum(pb & tb)
            fp = np.sum(pb & ~tb)
            fn = np.sum(~pb & tb)
            csi = tp / (tp + fp + fn + 1e-8)
            arr.append(float(csi))

        # Corr
        corr = np.corrcoef(p_m, t_m)[0, 1]
        corrs_masked.append(float(corr))

        # RMSE
        mse = np.mean((p_m - t_m) ** 2)
        rmses_masked.append(float(np.sqrt(mse)))

        # NSE
        var_t = np.var(t_m)
        nse = 1.0 - (mse / (var_t + 1e-8))
        nses_masked.append(float(nse))

        # Relative L2
        rel_l2 = np.linalg.norm(p_m - t_m) / (np.linalg.norm(t_m) + 1e-8)
        l2s_masked.append(float(rel_l2))

        print(f"[{idx:02d}/36] {name:28} | CSI@1cm: {csi_1_masked[-1]*100:4.1f}% | CSI@10cm: {csi_10_masked[-1]*100:4.1f}% | Corr: {corrs_masked[-1]:.3f} | NSE: {nses_masked[-1]:.3f}")

    print("\n================== BENCHMARK SUMMARY (RETRAINED DNO 30 EVENTS) ==================")
    print(f"CSI @ 1cm  (Masked): {np.mean(csi_1_masked)*100:.2f}% ± {np.std(csi_1_masked)*100:.2f}%")
    print(f"CSI @ 10cm (Masked): {np.mean(csi_10_masked)*100:.2f}% ± {np.std(csi_10_masked)*100:.2f}%")
    print(f"CSI @ 50cm (Masked): {np.mean(csi_50_masked)*100:.2f}% ± {np.std(csi_50_masked)*100:.2f}%")
    print(f"Corr       (Masked): {np.mean(corrs_masked)*100:.2f}% ± {np.std(corrs_masked)*100:.2f}%")
    print(f"NSE        (Masked): {np.mean(nses_masked):.4f} ± {np.std(nses_masked):.4f}")
    print(f"RMSE       (Masked): {np.mean(rmses_masked):.4f}m ± {np.std(rmses_masked):.4f}m")
    print(f"Rel L2     (Masked): {np.mean(l2s_masked)*100:.2f}% ± {np.std(l2s_masked)*100:.2f}%")
    print("==================================================================================")

if __name__ == "__main__":
    main()
