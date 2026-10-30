import os
import sys
import time
from pathlib import Path
import numpy as np
import tifffile
import torch

app_dir = Path("C:/Users/admin/Documents/Codex/2026-10-08/ngh/outputs/urbanfloodcast_app")
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(Path("C:/Users/admin/Documents/Codex/2026-10-08/ngh/work/UrbanFloodCast/DNO")))

from models.DNO import DNO
from p2m_data import resample_prediction
import utils25

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
    print("    EVALUATION ON BERLIN I (SEEN REGIONS, UNSEEN RAINFALL EVENTS - TEST SET)    ")
    print("             AT ORIGINAL FULL RESOLUTION (454 x 535)                            ")
    print("================================================================================")

    root = Path("C:/Users/admin/Downloads/UrbanFloodCast_Dataset/UrbanFloodCast_Dataset/BerlinI")
    ckpt_path = app_dir / "models/berlin_i_dno_30events.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNO(num_channels=5, width=16, initial_step=1, pad=False, factor=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()

    dem_path = root / "Input_BerlinI/DEM.tif"
    dem = tifffile.imread(dem_path).astype(np.float32)
    target_shape = dem.shape # (454, 535)

    dem = np.nan_to_num(dem, nan=-99999, posinf=-99999, neginf=-99999)
    nodata = dem < -2000
    if nodata.any():
        dem[nodata] = float(np.max(dem[~nodata]) + 30) if (~nodata).any() else 30.0

    test_dir = root / "Seen regions and unseen rainfall events/Test"
    test_events = sorted([p for p in test_dir.iterdir() if p.is_dir() and list(p.glob("*_0H.tif"))])
    print(f"Found {len(test_events)} official test events in Berlin I.")

    horizon = 24
    downsample = 4
    model_size = (max(96, target_shape[0] // downsample), max(96, target_shape[1] // downsample))

    dem_t = torch.from_numpy(dem).unsqueeze(2).repeat(1, 1, horizon).unsqueeze(-1)
    dem_t = torch.nn.functional.normalize(dem_t).unsqueeze(-1)

    csi_1_list, csi_10_list, csi_50_list = [], [], []
    corrs, nses, l2s, rmses = [], [], [], []

    for idx, ev in enumerate(test_events, 1):
        name = ev.name
        rain_series = read_rain(ev, ntime=25, dt_seconds=300)

        h0 = np.nan_to_num(tifffile.imread(ev / f"{name}_0H.tif").astype(np.float32), nan=0.0)
        u0 = np.nan_to_num(tifffile.imread(ev / f"{name}_0U.tif").astype(np.float32), nan=0.0)
        v0 = np.nan_to_num(tifffile.imread(ev / f"{name}_0V.tif").astype(np.float32), nan=0.0)
        state0 = np.stack([h0, u0, v0], axis=-1)

        state = torch.from_numpy(state0).unsqueeze(2).repeat(1, 1, horizon, 1).unsqueeze(-2)
        rain_t = torch.from_numpy(rain_series[1:25]).view(1, 1, horizon, 1).expand(target_shape[0], target_shape[1], -1, -1)
        rain_t = torch.log1p(torch.clamp(rain_t, min=0) / 1e-2) / 10.0
        rain_t = rain_t.unsqueeze(-1)

        x = torch.cat((state, rain_t, dem_t), dim=-1).unsqueeze(0).float()

        # Resample to model size
        b, y_d, x_d, t_d, ti_d, c_d = x.shape
        z = x.permute(0, 3, 4, 5, 1, 2).reshape(b, t_d * ti_d * c_d, y_d, x_d)
        z = torch.nn.functional.interpolate(z, size=model_size, mode="bilinear", align_corners=False)
        x_res = z.reshape(b, t_d, ti_d, c_d, *model_size).permute(0, 4, 5, 1, 2, 3).contiguous()

        with torch.inference_mode():
            pred = model(x_res.to(device))

        # Resample back to original full resolution (454, 535)
        pred_full = resample_prediction(pred[0].cpu(), torch, target_shape)
        p_h = np.maximum(pred_full[..., 0].numpy(), 0)

        t_h = np.zeros((target_shape[0], target_shape[1], horizon), dtype=np.float32)
        domain_mask = None

        times = list(range(300, 25 * 300, 300))
        for s_idx, sec in enumerate(times):
            raw_raster = tifffile.imread(ev / f"{name}_{sec}H.tif").astype(np.float32)
            if domain_mask is None:
                domain_mask = ~np.isnan(raw_raster)
            h_gt = np.nan_to_num(raw_raster, nan=0.0)
            t_h[..., s_idx] = np.maximum(h_gt, 0.0)

        domain_3d = np.repeat(domain_mask[:, :, np.newaxis], horizon, axis=2)
        p_m = p_h[domain_3d]
        t_m = t_h[domain_3d]

        # CSI at original resolution
        for th, arr in [(0.01, csi_1_list), (0.10, csi_10_list), (0.50, csi_50_list)]:
            pb = p_m >= th
            tb = t_m >= th
            tp = np.sum(pb & tb)
            fp = np.sum(pb & ~tb)
            fn = np.sum(~pb & tb)
            csi = tp / (tp + fp + fn + 1e-8)
            arr.append(float(csi))

        corr = np.corrcoef(p_m, t_m)[0, 1]
        corrs.append(float(corr))

        mse = np.mean((p_m - t_m) ** 2)
        rmses.append(float(np.sqrt(mse)))

        var_t = np.var(t_m)
        nse = 1.0 - (mse / (var_t + 1e-8))
        nses.append(float(nse))

        rel_l2 = np.linalg.norm(p_m - t_m) / (np.linalg.norm(t_m) + 1e-8)
        l2s.append(float(rel_l2))

        print(f"[{idx:02d}/12] {name:20} | CSI@1cm: {csi_1_list[-1]*100:4.1f}% | CSI@10cm: {csi_10_list[-1]*100:4.1f}% | Corr: {corrs[-1]:.3f} | NSE: {nses[-1]:.3f}")

    print("\n================== BERLIN I OFFICIAL TEST SET SUMMARY (FULL RESOLUTION) ==================")
    print(f"CSI @ 1cm  (Original Res): {np.mean(csi_1_list)*100:.2f}% ± {np.std(csi_1_list)*100:.2f}%  (Paper claim: ~87.5%)")
    print(f"CSI @ 10cm (Original Res): {np.mean(csi_10_list)*100:.2f}% ± {np.std(csi_10_list)*100:.2f}%  (Paper claim: ~70.0%)")
    print(f"CSI @ 50cm (Original Res): {np.mean(csi_50_list)*100:.2f}% ± {np.std(csi_50_list)*100:.2f}%  (Paper claim: ~45.0%)")
    print(f"Corr       (Original Res): {np.mean(corrs)*100:.2f}% ± {np.std(corrs)*100:.2f}%  (Paper claim: >93.0%)")
    print(f"NSE        (Original Res): {np.mean(nses):.4f} ± {np.std(nses):.4f}  (Paper claim: 0.88 - 0.94)")
    print(f"RMSE       (Original Res): {np.mean(rmses):.4f}m ± {np.std(rmses):.4f}m")
    print(f"Rel L2     (Original Res): {np.mean(l2s)*100:.2f}% ± {np.std(l2s)*100:.2f}%")
    print("===========================================================================================")

if __name__ == "__main__":
    main()
