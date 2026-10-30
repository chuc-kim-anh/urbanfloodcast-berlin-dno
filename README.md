# UrbanFloodCast DNO — Berlin II

Streamlit app for the public UrbanFloodCast Berlin II benchmark. It runs the HydroPML DNO emulator and displays predicted flood depth on an Esri satellite basemap.

## Quick start (Windows)

Requirements: Git, Python 3.11, PowerShell, and an internet connection. Initial setup downloads selected benchmark data from Zenodo and the upstream DNO source from GitHub. The full Zenodo archive is about 7.3 GB; the downloader uses HTTP range requests to fetch only the scenarios used by this project.

```powershell
git clone https://github.com/chuc-kim-anh/urbanfloodcast-berlin-dno.git
Set-Location urbanfloodcast-berlin-dno
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\setup_project.ps1
powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
```

When Streamlit starts, open <http://127.0.0.1:8504/>. The included checkpoint is `models/berlin_ii_dno.pt`. The setup script clones the original HydroPML code into `vendor/UrbanFloodCast`, downloads selected Berlin II rasters from Zenodo, and converts them to tensors.

For CUDA, install the PyTorch build matching the local GPU from the [official PyTorch selector](https://pytorch.org/get-started/locally/) instead of the generic `pip install torch` command.

## Data and model

- Benchmark dataset: [UrbanFloodCast on Zenodo](https://doi.org/10.5281/zenodo.15700880).
- Upstream model code: [HydroPML/UrbanFloodCast](https://github.com/HydroPML/UrbanFloodCast).
- The converter creates 5 training, 2 validation, and 1 held-out event tensors for a small development experiment. This is not the official benchmark split.
- `models/berlin_ii_dno.pt` is the locally trained Berlin II checkpoint used by the app. It is included so the app can run inference without retraining.
- To retrain: `.\.venv\Scripts\python.exe train_berlin_dno.py --epochs 12 --downsample 4`.

## Important limitations

- This is UrbanFloodCast DNO, not the unrelated coastal Predict-to-Map model.
- The checkpoint is trained for Berlin II. Replacing its DEM with another city does not make it valid there; new reference simulations and retraining or fine-tuning are needed.
- Custom rainfall is a spatially uniform synthetic forcing. The benchmark reference only matches its original storm.
- The DNO inputs include initial depth/flow, rainfall, and elevation. Roughness, sewer networks, roads, and satellite imagery are not DNO input channels. The satellite imagery is a visual basemap only.
- Benchmark reference maps are hydraulic simulation outputs, not field observations. This small prototype has not been validated for operational warnings.

## Source files

- `app_p2m.py` — Streamlit dashboard.
- `p2m_data.py`, `p2m_model.py` — data and inference adapters.
- `fetch_subset.py`, `prepare_urbanfloodcast.py` — download and convert benchmark data.
- `train_berlin_dno.py` — training entry point.
- `scripts/` — additional evaluation and preparation scripts.
