"""Convert downloaded Berlin II benchmark event folders to DNO tensor samples."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import zoom
import tifffile
import torch


def read_rain(event_dir: Path, stem: str, ntime: int = 25, dt_seconds: int = 300) -> np.ndarray:
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


def convert_event(event_dir: Path, dem_path: Path, output_path: Path):
    name = event_dir.name
    dem = tifffile.imread(dem_path).astype(np.float32)
    dem = np.nan_to_num(dem, nan=-99999, posinf=-99999, neginf=-99999)
    nodata = dem < -2000
    if nodata.any():
        # Match the upstream model's masked-DEM treatment: assign nodata above terrain.
        dem[nodata] = float(np.max(dem[~nodata]) + 30) if (~nodata).any() else 30.0

    # Berlin II supplies high-resolution terrain alongside coarser event rasters.
    # The DNO sample tensor must share the event grid; retain this aligned DEM
    # channel while the raw high-resolution layer remains available separately.
    first_h = event_dir / f"{name}_0H.tif"
    sample_shape = tifffile.imread(first_h).shape
    if dem.shape != sample_shape:
        dem = zoom(dem, (sample_shape[0] / dem.shape[0], sample_shape[1] / dem.shape[1]), order=1).astype(np.float32)
        dem = dem[:sample_shape[0], :sample_shape[1]]

    times = list(range(0, 25 * 300, 300))
    channels = []
    for seconds in times:
        frame = []
        for suffix in ("H", "U", "V"):
            path = event_dir / f"{name}_{seconds}{suffix}.tif"
            if not path.is_file():
                raise FileNotFoundError(f"Thiếu raster benchmark: {path}")
            arr = tifffile.imread(path).astype(np.float32)
            if arr.shape != dem.shape:
                raise ValueError(f"Lưới {path.name} {arr.shape} khác DEM {dem.shape}")
            frame.append(arr)
        channels.append(np.stack(frame, axis=-1))
    state = np.stack(channels, axis=2)  # [y,x,time,3]
    rain_series = read_rain(event_dir, name)
    rain = np.broadcast_to(rain_series[None, None, :, None], (*dem.shape, len(times), 1)).copy()
    elevation = np.broadcast_to(dem[:, :, None, None], (*dem.shape, len(times), 1)).copy()
    tensor = np.concatenate((state, rain, elevation), axis=-1).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(tensor), output_path)
    return tensor.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--output", default="data/converted")
    parser.add_argument("--dem", default="data/raw/Input_BerlinII/moa_bottom.tif")
    args = parser.parse_args()
    root, target, dem = Path(args.raw), Path(args.output), Path(args.dem)
    event_dirs = sorted(p for p in root.rglob("*") if p.is_dir() and list(p.glob("*_0H.tif")))
    if not event_dirs:
        raise SystemExit(f"Không tìm thấy thư mục sự kiện dưới {root}")
    for event in event_dirs:
        # Berlin II publishes Euler scenarios in Valid/Test folders, with no
        # dedicated Train split. This project creates a transparent development
        # split by storm total (not the benchmark's official split): 60/70/80 train,
        # 90 validation, and 110 test.
        match = re.search(r"(\d+)mm", event.name)
        mm = int(match.group(1)) if match else -1
        split = "train" if 0 < mm < 90 else "valid" if mm == 90 else "test"
        out = target / split / f"{event.parent.parent.name}_{event.name}.pt"
        shape = convert_event(event, dem, out)
        print(f"{split}: {event.name} -> {out.name}, tensor={shape}", flush=True)


if __name__ == "__main__":
    main()
