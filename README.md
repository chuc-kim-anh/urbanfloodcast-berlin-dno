# UrbanFloodCast Berlin II dashboard

This app wraps HydroPML's public **DNO** implementation for the Berlin II urban benchmark. It is not the LSU coastal P2M paper. The requested `moa_bottom.tif`, `moa_rough.tif`, and the Euler II/III scenarios are all from **Berlin II**; Berlin I instead uses `DEM.tif` and `Strickler_bln1.tif`.

## What is ready

- Streamlit dashboard with rainfall controls, a 5-minute timeline, a georeferenced Esri satellite basemap with a three-level overlay, building-roof masking, a selector for DNO forecast versus benchmark reference, depth error map, and threshold statistics.
- Adapter to the upstream DNO class/checkpoint format.
- Berlin II raster-to-tensor converter and a training entry point.
- Data handling distinguishes benchmark-simulation reference maps from field observations. Δh is only shown for the selected sample's native rainfall.
- Downloaded Berlin II terrain, roughness, and building layers plus 8 selected Euler events; converted 5 train, 2 validation, and 1 test event.
- Trained `models/berlin_ii_dno.pt` locally for 12 epochs; validation MSE reached 0.007353 on the development split.

## Run this workspace app

The launcher is [run_dashboard.ps1](run_dashboard.ps1). It uses the project `.venv` and opens the dashboard at Streamlit's local URL.

## Dataset/checkpoint status

The full Zenodo archive is 7.3 GB compressed. Berlin II has no dedicated `Train` directory: it provides Euler II, Euler III, and random design storms under `Valid` and `Test`. The preparation script creates a **development split** by storm total (60/70/80 mm train, 90 mm validation, 110 mm test); this is not the official benchmark split and should not be reported as official test accuracy. It uses the exact Berlin II DEM and keeps the original hydrodynamic simulation outputs as labels.

UrbanFloodCast does not publish a pretrained checkpoint. The checkpoint was trained locally on a grid downsampled by a factor of four because this machine has a 4 GB GPU. Inference uses the same grid size, then bilinearly resamples predicted maps to the event grid. Predictions from this small development subset are for a working prototype, not a validated operational forecast.

## Data-preparation and training commands

This workspace uses HTTP range reads to fetch the Berlin II input rasters and selected storm folders from Zenodo rather than downloading the entire 7.3 GB archive. Data preparation and initial model training are complete. To retrain:

```powershell
Set-Location outputs/urbanfloodcast_app
.\.venv\Scripts\python.exe prepare_urbanfloodcast.py
.\.venv\Scripts\python.exe train_berlin_dno.py --epochs 12 --downsample 4
.\run_dashboard.ps1
```

The app defaults to `data/converted/test` and `models/berlin_ii_dno.pt`. Override those locations with `URBANFLOODCAST_SAMPLE_DIR` and `URBANFLOODCAST_CHECKPOINT` if needed. DNO source code defaults to `../../work/UrbanFloodCast/DNO`.

## Model/input limitations

- Official raster events are at 300-second intervals. The model emits the next 24 frames after the initial state.
- The public DNO tensor channels contain flow state, rainfall, and elevation. While `moa_rough.tif` exists, upstream DNO preprocessing does not feed roughness as an additional network channel; adding that feature requires retraining a changed architecture.
- Custom rainfall is an idealized, spatially uniform hyetograph. It reuses the selected sample's initial flow state and terrain. The benchmark reference map is not valid for a different synthetic storm.
- The DEM displayed in the app is aligned to the event grid and preprocessed, not a CRS-aware high-resolution basemap.
- The interactive satellite basemap uses Esri World Imagery tiles in the browser and needs an internet connection. It is a visual backdrop only; the DNO still uses the Berlin II benchmark inputs, not satellite pixels.

## Source files

- `app_p2m.py` — dashboard.
- `p2m_data.py` — tensor loading and rainfall/feature transforms.
- `p2m_model.py` — official DNO inference adapter.
- `prepare_urbanfloodcast.py` — Berlin II raster conversion.
- `train_berlin_dno.py` — local DNO training.
