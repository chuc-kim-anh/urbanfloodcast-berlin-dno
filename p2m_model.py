"""Runtime adapter for the official HydroPML UrbanFloodCast DNO checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path


def load_predictor(repo_root: str | Path, checkpoint_path: str | Path, torch):
    dno_root = Path(repo_root).expanduser() / "DNO"
    if not (dno_root / "models" / "DNO.py").is_file():
        raise FileNotFoundError(f"Không thấy mã DNO chính thức tại {dno_root}. Hãy đặt URBANFLOODCAST_CODE trỏ tới thư mục DNO.")
    if str(dno_root) not in sys.path:
        sys.path.insert(0, str(dno_root))
    from models.DNO import DNO

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Chưa có checkpoint đã huấn luyện: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNO(num_channels=5, width=10, initial_step=1, pad=False, factor=1).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model, device


def predict_flood_depth(model, device, model_input, torch):
    """Return official DNO output [time,y,x,channels] with channel 0 as depth."""
    with torch.inference_mode():
        prediction = model(model_input.to(device))
    if prediction.ndim != 5:
        raise ValueError(f"DNO trả tensor shape không nhận diện được: {tuple(prediction.shape)}")
    # DNO repository convention: [batch,y,x,time,channels].
    return prediction[0].detach().float().cpu()
