from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from p2m_data import (
    build_model_input,
    find_events,
    get_reference,
    load_sample,
    model_grid_size,
    resample_model_input,
    resample_prediction,
)
from p2m_model import load_predictor, predict_flood_depth


st.set_page_config(page_title="UrbanFloodCast | Berlin II", page_icon="🌧️", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_predictor(repo_root: str, checkpoint_path: str, checkpoint_mtime: float):
    """Keep the loaded DNO on device between Streamlit reruns."""
    import torch
    return load_predictor(repo_root, checkpoint_path, torch)


def get_satellite_bounds():
    """Convert Berlin II GeoTIFF bounds (UTM 33N) to Leaflet lat/lon bounds."""
    import rasterio
    from rasterio.warp import transform_bounds

    dem_path = Path("data/raw/Input_BerlinII/moa_bottom.tif")
    if not dem_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy DEM có tọa độ: {dem_path}")
    with rasterio.open(dem_path) as src:
        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    return [[south, west], [north, east]], [(south + north) / 2, (west + east) / 2]


@st.cache_data(show_spinner=False)
def get_building_mask(shape: tuple[int, int]) -> np.ndarray:
    """Align the Berlin II building footprint mask to the event raster grid."""
    import tifffile
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

    path = Path("data/raw/Input_BerlinII/moa_building.tif")
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy mặt nạ công trình: {path}")
    with rasterio.open(path) as src:
        destination = np.zeros(shape, dtype=np.uint8)
        # Event H rasters have no GeoTIFF tags. Assign their grid the exact
        # DEM footprint, then align the building mask with a nearest-neighbor
        # reprojection rather than image resizing.
        event_transform = from_bounds(*src.bounds, width=shape[1], height=shape[0])
        reproject(
            source=tifffile.imread(path).astype(np.uint8),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=event_transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest,
        )
    return destination.astype(bool)


def depth_rgba(depth: np.ndarray) -> np.ndarray:
    """Build a transparent three-level flood overlay."""
    rgba = np.zeros((*depth.shape, 4), dtype=np.uint8)
    # Ignore sub-2 cm numerical noise; dry cells stay transparent.
    low = np.isfinite(depth) & (depth >= 0.02) & (depth < 0.20)
    medium = np.isfinite(depth) & (depth >= 0.20) & (depth <= 0.40)
    high = np.isfinite(depth) & (depth > 0.40)
    rgba[low] = (0, 190, 110, 190)       # green: 2–20 cm
    rgba[medium] = (255, 190, 0, 210)    # amber: 20–40 cm
    rgba[high] = (235, 45, 55, 225)      # red: >40 cm
    return rgba


def render_satellite_map(depth: np.ndarray, selected_minute: int, event_name: str, layer_label: str):
    import folium

    bounds, center = get_satellite_bounds()
    buildings = get_building_mask(depth.shape)
    # The benchmark's building mask marks roofs as non-inundatable cells.
    depth = np.where(buildings, np.nan, depth)
    fmap = folium.Map(location=center, zoom_start=15, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Sources: Esri, Maxar, Earthstar Geographics",
        name="Ảnh vệ tinh Esri",
        overlay=False,
        control=True,
    ).add_to(fmap)
    folium.raster_layers.ImageOverlay(
        image=depth_rgba(depth), bounds=bounds, opacity=0.92, name=f"Ngập: {layer_label}",
        interactive=True, cross_origin=False, zindex=10, show=True,
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    legend = f"""
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:rgba(12,22,35,.94);
                color:#fff;padding:12px 14px;border:1px solid #63758b;border-radius:8px;
                font:13px/1.6 Arial,sans-serif;box-shadow:0 2px 8px #0008">
      <b>{layer_label} · phút {selected_minute}</b><br>
      <span style="color:#00be6e">■</span> Mức 1 · 2–&lt;20 cm<br>
      <span style="color:#ffbe00">■</span> Mức 2 · 20–40 cm<br>
      <span style="color:#eb2d37">■</span> Mức 3 · &gt;40 cm<br>
      <span style="color:#b8c2ce">Ô trong suốt: &lt;2 cm / không ngập<br>và mặt nạ mái nhà</span>
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend))
    return fmap


st.markdown("""
<style>
.stApp { background: #07111c; color: #e7eff8; }
[data-testid="stSidebar"] { background: #0b1825; }
h1,h2,h3 { letter-spacing: -.025em; }
div[data-testid="stMetric"] { background: #0e1d2b; border: 1px solid #20354a; padding: 14px; border-radius: 12px; }
.caption { color:#8da6bd; }
</style>
""", unsafe_allow_html=True)

st.title("Dự báo ngập đô thị · Berlin II")
st.caption("UrbanFloodCast DNO · dự báo độ sâu ngập từ forcing mưa, trạng thái ban đầu và đặc trưng địa hình đã mã hóa trong mẫu benchmark.")

with st.sidebar:
    st.header("Thiết lập dữ liệu")
    code_root = Path(os.getenv("URBANFLOODCAST_CODE", "../../work/UrbanFloodCast/DNO")).resolve()
    if code_root.name == "DNO":
        code_repo_root = code_root.parent
    else:
        code_repo_root = code_root
    sample_dir = Path(os.getenv("URBANFLOODCAST_SAMPLE_DIR", "data/converted/test")).expanduser()
    checkpoint = Path(os.getenv("URBANFLOODCAST_CHECKPOINT", "models/berlin_ii_dno.pt")).expanduser()
    st.text_input("Thư mục test (.pt đã chuyển đổi)", str(sample_dir), key="sample_dir")
    st.text_input("Checkpoint DNO (.pt)", str(checkpoint), key="checkpoint")
    st.caption("Checkpoint được huấn luyện trực tiếp trên các kịch bản Berlin II đã chuyển đổi.")

events = find_events(st.session_state.sample_dir)
left, right = st.columns([1, 2.1], gap="large")
with left:
    st.subheader("Kịch bản mưa")
    scenario = st.selectbox("Chọn dạng mưa", ["Mưa benchmark mẫu chọn", "Euler II", "Euler III", "Mưa cực đoan", "Tùy chỉnh"])
    defaults = {"Mưa benchmark mẫu chọn": (45.0, 45), "Euler II": (45.0, 60), "Euler III": (70.0, 60), "Mưa cực đoan": (110.0, 60), "Tùy chỉnh": (50.0, 45)}
    default_total, default_duration = defaults[scenario]
    total_mm = st.slider("Tổng lượng mưa (mm)", 10, 150, int(default_total), 1, disabled=scenario == "Mưa benchmark mẫu chọn")
    duration_min = st.slider("Thời lượng mưa (phút)", 15, 120, default_duration, 5, disabled=scenario == "Mưa benchmark mẫu chọn")
    if events:
        labels = [e.path.name for e in events]
        selected = st.selectbox("Mẫu Berlin II làm trạng thái ban đầu / tham chiếu", labels)
        sample_path = next(e.path for e in events if e.path.name == selected)
    else:
        sample_path = None
        st.warning("Chưa tìm thấy tệp .pt. Hãy chuyển dữ liệu benchmark theo TIFF2PT của repo chính thức, rồi trỏ thư mục test vào đây.")
    run = st.button("▶  Chạy mô hình DNO", type="primary", use_container_width=True, disabled=not bool(sample_path))
    st.caption("Bước thời gian của bộ dữ liệu gốc là 5 phút; thời lượng/độ dài dự báo phụ thuộc chuỗi mẫu và checkpoint.")

custom_rain = scenario != "Mưa benchmark mẫu chọn"
run_signature = (
    sample_path.name if sample_path else None,
    scenario,
    float(total_mm) if custom_rain else None,
    int(duration_min) if custom_rain else None,
)

if run:
    import torch
    status = st.status("Đang bắt đầu dự báo…", expanded=True)
    try:
        total_start = time.perf_counter()
        status.write("Đang đọc mẫu raster đã chuyển đổi…")
        sample = load_sample(sample_path, torch)
        status.write(f"Đã đọc mẫu {tuple(sample.shape)}. Đang tạo đầu vào…")
        # The official data adapter's one-shot feature construction is used.
        x, horizon = build_model_input(sample, torch, total_mm=float(total_mm) if custom_rain else None, duration_min=duration_min, pattern=scenario)
        checkpoint_path = Path(st.session_state.checkpoint).expanduser()
        checkpoint_mtime = checkpoint_path.stat().st_mtime if checkpoint_path.is_file() else 0.0
        status.write("Đang nạp/khởi tạo DNO (lần đầu có thể mất thêm thời gian)…")
        model, device = cached_predictor(str(code_repo_root), str(checkpoint_path), checkpoint_mtime)
        original_size = tuple(sample.shape[:2])
        inference_size = model_grid_size(*original_size, factor=4)
        x_model = resample_model_input(x, torch, inference_size)
        status.write(
            f"Đang suy luận trên {device.type.upper()} cho {horizon} bước ở lưới "
            f"{inference_size[0]}×{inference_size[1]} (độ phân giải lúc huấn luyện)…"
        )
        inference_start = time.perf_counter()
        prediction = predict_flood_depth(model, device, x_model, torch)
        prediction = resample_prediction(prediction, torch, original_size)
        inference_elapsed = time.perf_counter() - inference_start
        status.write("Đang chuẩn bị bản đồ và thống kê…")
        reference, valid_mask = get_reference(sample, torch, horizon)
        prediction = prediction[..., 0].numpy()  # [y,x,t]
        truth = reference[0, ..., 0].numpy()
        valid = valid_mask[0, ..., 0].numpy()
        # Only the benchmark's original forcing has a valid paired reference.
        is_benchmark_rain = not custom_rain
        dem_map = sample[..., 0, 4].numpy()
        initial_depth = sample[..., 0, 0].numpy()
        st.session_state.result = (prediction, truth, valid, inference_elapsed, horizon, is_benchmark_rain, sample_path.name, dem_map, initial_depth)
        st.session_state.result_signature = run_signature
        total_elapsed = time.perf_counter() - total_start
        status.update(label=f"Dự báo hoàn tất · GPU/CPU {inference_elapsed:.1f}s · tổng {total_elapsed:.1f}s", state="complete", expanded=False)
    except Exception as exc:
        status.update(label=f"Dự báo thất bại: {exc}", state="error", expanded=True)
        st.exception(exc)

result = st.session_state.get("result")
if result is not None and st.session_state.get("result_signature") != run_signature:
    result = None
    st.info("Bạn đã đổi mẫu hoặc kịch bản mưa. Bấm **Chạy mô hình DNO** để tính lại; bản đồ cũ đã được ẩn để tránh nhầm.")
with right:
    if result is None:
        st.subheader("Bản đồ kết quả")
        st.info("Chọn mẫu benchmark và checkpoint đã huấn luyện, sau đó bấm **Chạy mô hình DNO**. Ứng dụng không tạo bản đồ thay thế khi thiếu mô hình.")
        st.markdown("**Các bước cần có**\n1. Tải raster Berlin II và chuyển sang tensor `.pt`.\n2. Huấn luyện DNO trên các mẫu Berlin II đã chia theo tổng mưa.\n3. Đặt `URBANFLOODCAST_SAMPLE_DIR` và `URBANFLOODCAST_CHECKPOINT`, rồi khởi động lại ứng dụng.")
    else:
        pred, truth, valid, elapsed, horizon, paired, event_name, dem_map, initial_depth = result
        dt = 5
        selected_min = st.slider("Thời gian dự báo", 0, min(horizon * dt, 120), min(60, horizon * dt), dt)
        st.caption("0 phút là trạng thái ban đầu; chọn 65 phút để xem dự báo sau 65 phút tính từ lúc bắt đầu mưa. Mốc này độc lập với thời lượng mưa đã nhập.")
        t = min(max(selected_min // dt - 1, 0), horizon - 1)
        pred_map = np.maximum(initial_depth, 0) if selected_min == 0 else np.maximum(pred[:, :, t], 0)
        ref_map = np.maximum(initial_depth, 0) if selected_min == 0 else np.maximum(truth[:, :, t], 0)
        valid2 = np.isfinite(pred_map) & np.isfinite(ref_map)
        masked_pred = np.where(valid2, pred_map, np.nan)
        err_map = np.where(valid2, pred_map - ref_map, np.nan) if paired else None
        a, b, c, d, e, f = st.columns(6)
        a.metric("Độ sâu ngập lớn nhất", f"{np.nanmax(masked_pred):.2f} m")
        n_valid = max(np.isfinite(masked_pred).sum(), 1)
        finite_pred = np.isfinite(masked_pred)
        b.metric("Diện tích ngập ≥2 cm", f"{100 * (finite_pred & (masked_pred >= .02)).sum() / n_valid:.1f}%")
        c.metric("Mức 1 · 2–<20 cm", f"{100 * (finite_pred & (masked_pred >= .02) & (masked_pred < .2)).sum() / n_valid:.1f}%")
        d.metric("Mức 2 · 20–40 cm", f"{100 * (finite_pred & (masked_pred >= .2) & (masked_pred <= .4)).sum() / n_valid:.1f}%")
        e.metric("Mức 3 · >40 cm", f"{100 * (finite_pred & (masked_pred > .4)).sum() / n_valid:.1f}%")
        f.metric("Thời gian suy luận", f"{elapsed:.2f} s")
        if not paired:
            st.warning("Mưa tùy chỉnh không có bản đồ tham chiếu tương ứng trong benchmark; không hiển thị Δh để tránh so sánh sai kịch bản.")
        tab1, tab2, tab3, tab4 = st.tabs(["Ảnh vệ tinh + ngập", "DEM đầu vào", "DNO dự báo", "Sai lệch Δh"])
        import plotly.graph_objects as go
        def heatmap(z, title, colors, zmin=None, zmax=None):
            fig = go.Figure(go.Heatmap(z=z, colorscale=colors, zmin=zmin, zmax=zmax, colorbar=dict(title="m")))
            fig.update_layout(title=title, height=570, margin=dict(l=5, r=5, t=50, b=5), paper_bgcolor="#07111c", plot_bgcolor="#07111c", font_color="#dce8f4")
            fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, title="")
            fig.update_xaxes(title="")
            return fig
        with tab1:
            map_layer = "DNO dự báo"
            map_depth = masked_pred
            st.caption(f"Ảnh nền vệ tinh · {map_layer} · phút {selected_min} · {event_name}")
            pred_med_pct = 100 * np.count_nonzero(valid2 & (pred_map >= 0.20) & (pred_map <= 0.40)) / max(np.count_nonzero(valid2), 1)
            pred_high_pct = 100 * np.count_nonzero(valid2 & (pred_map > 0.40)) / max(np.count_nonzero(valid2), 1)
            ref_med_pct = 100 * np.count_nonzero(valid2 & (ref_map >= 0.20) & (ref_map <= 0.40)) / max(np.count_nonzero(valid2), 1)
            ref_high_pct = 100 * np.count_nonzero(valid2 & (ref_map > 0.40)) / max(np.count_nonzero(valid2), 1)
            if paired and pred_med_pct + pred_high_pct == 0 and ref_med_pct + ref_high_pct > 0:
                st.warning(
                    f"Checkpoint DNO chưa dự báo được ô ngập từ 20 cm trở lên tại phút {selected_min}; "
                    f"kết quả benchmark mô phỏng tham chiếu có {ref_med_pct:.1f}% mức 2 và {ref_high_pct:.1f}% mức 3. "
                    "Benchmark chỉ dùng làm đối chiếu, không phải quan trắc thực địa."
                )
            try:
                fmap = render_satellite_map(map_depth, selected_min, event_name, map_layer)
                components.html(fmap.get_root().render(), height=650, scrolling=False)
                st.caption("Ảnh nền Esri chỉ để tham khảo vị trí; ảnh không phải đầu vào của mô hình. Lớp DNO dự báo dùng ba màu theo độ sâu; vùng khô và mái nhà để trong suốt.")
            except Exception as exc:
                st.warning(f"Không dựng được bản đồ vệ tinh: {exc}. Các bản đồ raster bên dưới vẫn dùng được.")
                st.plotly_chart(heatmap(masked_pred, f"Độ sâu dự báo · t = {selected_min} phút", "Turbo", 0), use_container_width=True, key="satellite_fallback_depth")
        with tab2:
            st.plotly_chart(heatmap(dem_map, "Cao trình DEM trong mẫu benchmark (đã tiền xử lý)", "Earth"), use_container_width=True, key="dem_input_heatmap")
            st.caption("Giá trị đang hiển thị là kênh cao trình sau tiền xử lý của tensor benchmark, không phải cao trình mét gốc.")
        with tab3:
            st.plotly_chart(heatmap(masked_pred, f"Độ sâu dự báo · t = {selected_min} phút", "Turbo", 0), use_container_width=True, key="dno_prediction_heatmap")
        with tab4:
            if paired:
                lim = float(np.nanpercentile(np.abs(err_map), 98)) if np.isfinite(err_map).any() else 1
                st.plotly_chart(heatmap(err_map, "DNO − tham chiếu (m)", "RdBu", -lim, lim), use_container_width=True, key="depth_error_heatmap")
            else:
                st.info("Không có Δh hợp lệ cho trường hợp mưa tùy chỉnh.")

st.divider()
st.caption("Benchmark Berlin được xây từ mô phỏng thủy lực. Bản đồ tham chiếu là đầu ra benchmark/mô phỏng, không mặc định là quan trắc ngập thực địa. Kết quả phụ thuộc checkpoint và kịch bản đầu vào.")
