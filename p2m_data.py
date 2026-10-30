"""Data adapters for the public UrbanFloodCast Berlin benchmark.

This project uses the authors' DNO/TL-DNO implementation (not the unrelated
coastal-flood P2M paper). Benchmark .pt samples contain the raster forcing and
reference series used by the official training pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np


@dataclass
class EventData:
    path: Path
    shape: tuple[int, int]
    steps: int
    timestep_minutes: float


def find_events(data_dir: str | Path) -> list[EventData]:
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        return []
    events = []
    for path in sorted(root.rglob("*.pt")):
        events.append(EventData(path=path, shape=(0, 0), steps=25, timestep_minutes=5.0))
    return events


def load_sample(path: str | Path, torch):
    """Load a converted official benchmark tensor [y,x,time,5]."""
    try:
        sample = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        sample = torch.load(Path(path), map_location="cpu")
    if not torch.is_tensor(sample) or sample.ndim != 4 or sample.shape[-1] < 5:
        raise ValueError("Mẫu cần là tensor .pt dạng [y, x, time, >=5 channels] của UrbanFloodCast.")
    if sample.shape[2] < 2:
        raise ValueError("Mẫu cần có ít nhất hai bước thời gian.")
    # TIFF-derived flow/depth channels contain NaNs outside the simulated
    # hydraulic domain. The public DNO has no explicit spatial mask input, and
    # NaNs otherwise propagate through every convolution. Treat those cells as
    # dry/zero-flow for the emulator input and supervised target.
    return torch.nan_to_num(sample.float(), nan=0.0, posinf=0.0, neginf=0.0).contiguous()


def make_rain_series(total_mm: float, duration_min: int, steps: int, pattern: str) -> np.ndarray:
    """Generate a piecewise-constant rain series in mm/h at model time steps."""
    dt_min = 5.0  # UrbanFloodCast Berlin samples use 300 s forcing intervals.
    values = np.zeros(steps, dtype=np.float32)
    active = max(1, min(steps, int(math.ceil(duration_min / dt_min))))
    rate = total_mm * 60.0 / max(duration_min, 1)
    if pattern == "Euler II":
        weights = np.linspace(1.8, 0.2, active, dtype=np.float32)
    elif pattern == "Euler III":
        weights = np.exp(-0.5 * ((np.arange(active) - (active - 1) / 2) / max(active / 5, 0.5)) ** 2).astype(np.float32)
    else:
        weights = np.ones(active, dtype=np.float32)
    weights /= weights.sum()
    # Convert event depth allocation to equivalent interval intensity.
    values[:active] = weights * total_mm * 60.0 / dt_min
    return values


def build_model_input(sample, torch, total_mm: float | None = None,
                      duration_min: int = 60, pattern: str = "Benchmark event"):
    """Apply the official one-shot feature transforms to a benchmark sample.

    For a custom storm, replace only the rainfall channel. Initial water state,
    terrain, and spatial grid stay from the selected Berlin sample.
    """
    T_IN = 1
    horizon = sample.shape[2] - T_IN
    # Official one-shot features: initial h/qx/qy, log rainfall, normalized DEM.
    state = sample[..., :T_IN, :3]
    state = state.unsqueeze(-3).repeat(1, 1, horizon, 1, 1)
    rain = sample[..., T_IN:T_IN + horizon, 3:4].clone()
    if total_mm is not None:
        series = make_rain_series(total_mm, duration_min, horizon, pattern)
        # Rain forcing in the benchmark is spatially uniform per time slice.
        rain = torch.as_tensor(series, dtype=sample.dtype).view(1, 1, horizon, 1).expand(
            sample.shape[0], sample.shape[1], -1, -1)
    rain = torch.log1p(torch.clamp(rain, min=0) / 1e-2) / 10.0
    rain = rain.unsqueeze(-1)
    dem = sample[..., T_IN:T_IN + horizon, 4:5].clone()
    dem = torch.nan_to_num(dem, nan=float(torch.nan_to_num(dem).max()) + 30.0)
    # Match the official preprocessing, which applies torch.normalize with its
    # default dimension (dimension 1 of [y,x,time,channel]).
    dem = torch.nn.functional.normalize(dem)
    dem = dem.unsqueeze(-1)
    x = torch.cat((state, rain, dem), dim=-1)
    # Official network expects [batch,y,x,forecast_time,input_time,channels].
    return x.unsqueeze(0), horizon


def model_grid_size(height: int, width: int, factor: int = 4) -> tuple[int, int]:
    """Spatial grid used for both training and inference of the pilot checkpoint."""
    return max(96, height // factor), max(96, width // factor)


def resample_model_input(x, torch, size: tuple[int, int]):
    """Resize [B,Y,X,T,1,C] inputs using the same interpolation as training."""
    batch, height, width, steps, input_steps, channels = x.shape
    z = x.permute(0, 3, 4, 5, 1, 2).reshape(
        batch, steps * input_steps * channels, height, width
    )
    z = torch.nn.functional.interpolate(z, size=size, mode="bilinear", align_corners=False)
    return z.reshape(batch, steps, input_steps, channels, *size).permute(
        0, 4, 5, 1, 2, 3
    ).contiguous()


def resample_prediction(prediction, torch, size: tuple[int, int]):
    """Resize [Y,X,T,C] DNO output to the event grid for mapping."""
    height, width, steps, channels = prediction.shape
    z = prediction.permute(2, 3, 0, 1).reshape(1, steps * channels, height, width)
    z = torch.nn.functional.interpolate(z, size=size, mode="bilinear", align_corners=False)
    return z.reshape(1, steps, channels, *size).permute(0, 3, 4, 1, 2)[0].contiguous()


def get_reference(sample, torch, horizon: int):
    """Reference simulation depth and validity mask, first output channel is depth."""
    target = sample[..., 1:1 + horizon, :3].clone().unsqueeze(0)
    dry_mask = target[..., 0:1] == 0
    target[..., 1:2][dry_mask] = 0
    target[..., 2:3][dry_mask] = 0
    mask = torch.isfinite(target)
    return torch.nan_to_num(target), mask
