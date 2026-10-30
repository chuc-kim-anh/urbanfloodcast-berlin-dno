import os
import sys
import re
from pathlib import Path
import numpy as np
import tifffile
import torch
from scipy.ndimage import zoom

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

def convert_event(event_dir: Path, dem: np.ndarray, output_path: Path):
    name = event_dir.name
    sample_shape = (433, 692)
    times = list(range(0, 25 * 300, 300))
    channels = []
    for seconds in times:
        frame = []
        for suffix in ("H", "U", "V"):
            p = event_dir / f"{name}_{seconds}{suffix}.tif"
            if not p.is_file():
                raise FileNotFoundError(f"Missing raster: {p}")
            arr = tifffile.imread(p).astype(np.float32)
            frame.append(arr)
        channels.append(np.stack(frame, axis=-1))
    state = np.stack(channels, axis=2) # [y, x, time, 3]
    rain_series = read_rain(event_dir)
    rain = np.broadcast_to(rain_series[None, None, :, None], (*sample_shape, len(times), 1)).copy()
    elevation = np.broadcast_to(dem[:, :, None, None], (*sample_shape, len(times), 1)).copy()
    tensor = np.concatenate((state, rain, elevation), axis=-1).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(tensor), output_path)
    return tensor.shape

def main():
    root = Path("C:/Users/admin/Downloads/UrbanFloodCast_Dataset/UrbanFloodCast_Dataset/BerlinII")
    out_root = Path("C:/Users/admin/Documents/Codex/2026-10-08/ngh/outputs/urbanfloodcast_app/data/converted_30")
    dem_path = root / "Input_BerlinII/moa_bottom.tif"
    
    print("[1/3] Loading and preparing base DEM...")
    dem = tifffile.imread(dem_path).astype(np.float32)
    dem = np.nan_to_num(dem, nan=-99999, posinf=-99999, neginf=-99999)
    nodata = dem < -2000
    if nodata.any():
        dem[nodata] = float(np.max(dem[~nodata]) + 30) if (~nodata).any() else 30.0
    sample_shape = (433, 692)
    if dem.shape != sample_shape:
        dem = zoom(dem, (sample_shape[0] / dem.shape[0], sample_shape[1] / dem.shape[1]), order=1).astype(np.float32)
        dem = dem[:sample_shape[0], :sample_shape[1]]

    # 2. Select 30 train events (10 EulerII, 10 EulerIII, 10 Rand) and 6 valid events (2 each)
    print("[2/3] Selecting 30 Train and 6 Valid events...")
    e2_dirs = sorted([p for p in root.glob("EulerII/*/*") if p.is_dir() and list(p.glob("*_0H.tif"))])
    e3_dirs = sorted([p for p in root.glob("EulerIII/*/*") if p.is_dir() and list(p.glob("*_0H.tif"))])
    rand_dirs = sorted([p for p in root.glob("Rand/*") if p.is_dir() and list(p.glob("*_0H.tif"))])

    train_events = e2_dirs[:10] + e3_dirs[:10] + rand_dirs[:10]
    valid_events = e2_dirs[10:12] + e3_dirs[10:12] + rand_dirs[10:12]

    print(f"Selected: {len(train_events)} Train events, {len(valid_events)} Valid events.")

    # 3. Convert
    print("[3/3] Converting Train events to .pt tensors...")
    for idx, ev in enumerate(train_events, 1):
        out_p = out_root / "train" / f"{ev.parent.parent.name}_{ev.name}.pt"
        if not out_p.exists():
            sh = convert_event(ev, dem, out_p)
            print(f"  [Train {idx:02d}/30] {ev.name} -> {out_p.name} ({sh})")
        else:
            print(f"  [Train {idx:02d}/30] {ev.name} already exists.")

    print("Converting Valid events to .pt tensors...")
    for idx, ev in enumerate(valid_events, 1):
        out_p = out_root / "valid" / f"{ev.parent.parent.name}_{ev.name}.pt"
        if not out_p.exists():
            sh = convert_event(ev, dem, out_p)
            print(f"  [Valid {idx:02d}/06] {ev.name} -> {out_p.name} ({sh})")
        else:
            print(f"  [Valid {idx:02d}/06] {ev.name} already exists.")

    print("ALL 36 EVENTS CONVERTED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
