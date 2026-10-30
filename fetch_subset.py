from remotezip import RemoteZip
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, struct, zlib, bz2, zipfile
import time

URL = "https://zenodo.org/api/records/15700880/files/UrbanFloodCast_Dataset.zip/content"
ROOT = Path("data/raw")
STATIC = [
    "UrbanFloodCast_Dataset/BerlinII/Input_BerlinII/moa_bottom.tif",
    "UrbanFloodCast_Dataset/BerlinII/Input_BerlinII/moa_rough.tif",
    "UrbanFloodCast_Dataset/BerlinII/Input_BerlinII/moa_building.tif",
]
DIRS = [
    "UrbanFloodCast_Dataset/BerlinII/EulerII/Test/EulerIIm_60mm_1h/",
    "UrbanFloodCast_Dataset/BerlinII/EulerII/Test/EulerIIm_70mm_1h/",
    "UrbanFloodCast_Dataset/BerlinII/EulerII/Test/EulerIIm_80mm_1h/",
    "UrbanFloodCast_Dataset/BerlinII/EulerII/Test/EulerIIm_90mm_1h/",
    "UrbanFloodCast_Dataset/BerlinII/EulerII/Valid/EulerIIm_110mm_1h/",
    "UrbanFloodCast_Dataset/BerlinII/EulerIII/Test/EulerIIIm_60mm_1h/",
    "UrbanFloodCast_Dataset/BerlinII/EulerIII/Test/EulerIIIm_70mm_1h/",
    "UrbanFloodCast_Dataset/BerlinII/EulerIII/Test/EulerIIIm_80mm_1h/",
    "UrbanFloodCast_Dataset/BerlinII/EulerIII/Test/EulerIIIm_90mm_1h/",
    "UrbanFloodCast_Dataset/BerlinII/Valid/EulerIIIm_110mm_1h/",
]


def fetch_group(group):
    label, infos = group
    missing = []
    for info in infos:
        dest = ROOT / info.filename.split("UrbanFloodCast_Dataset/BerlinII/", 1)[-1]
        if not (dest.exists() and dest.stat().st_size == info.file_size):
            missing.append((info, dest))
    if not missing:
        return label, 0, "cached"

    first = min(info.header_offset for info, _ in missing)
    # Local headers have the same UTF-8 filename length plus a small extra field.
    # Include a generous tail margin so one range request covers every member.
    last = max(info.header_offset + 30 + len(info.filename.encode("utf-8")) + 4096 + info.compress_size
               for info, _ in missing)
    response = None
    for attempt in range(6):
        response = requests.get(URL, headers={"Range": f"bytes={first}-{last - 1}"}, timeout=240)
        if response.status_code == 206:
            break
        if response.status_code == 429:
            time.sleep(min(10 * (attempt + 1), 60))
            continue
        raise RuntimeError(f"ZIP range failed for {label}: {response.status_code}")
    if response is None or response.status_code != 206:
        raise RuntimeError(f"ZIP range rate-limited for {label}")
    block = response.content

    for info, dest in missing:
        local = info.header_offset - first
        header = block[local:local + 30]
        sig, _, _, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
        if sig != 0x04034B50:
            raise RuntimeError(f"Invalid local ZIP header for {info.filename}")
        begin_abs = info.header_offset + 30 + name_len + extra_len
        end_abs = begin_abs + info.compress_size
        begin, end = begin_abs - first, end_abs - first
        compressed = block[begin:end]
        if method == zipfile.ZIP_STORED:
            data = compressed
        elif method == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(compressed, -15)
        elif method == zipfile.ZIP_BZIP2:
            data = bz2.decompress(compressed)
        else:
            raise RuntimeError(f"Unsupported ZIP compression method {method}")
        if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
            raise RuntimeError(f"CRC/size check failed for {info.filename}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return label, len(block), f"{len(missing)} files"


def main():
    groups = []
    with RemoteZip(URL) as archive:
        entries = archive.infolist()
        for directory in DIRS:
            infos = [i for i in entries if i.filename.startswith(directory) and not i.is_dir()]
            if infos:
                groups.append((directory, infos))
        infos = [i for i in entries if i.filename in STATIC]
        if infos:
            groups.append(("BerlinII inputs", infos))
    print(f"Fetching {len(groups)} raster groups with HTTP range reads", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch_group, group) for group in groups]
        for future in as_completed(futures):
            label, size, status = future.result()
            print(f"{label.rsplit('/', 1)[-1]}: {size / 1e6:.1f} MB, {status}", flush=True)
    print("Berlin II subset fetched and CRC-checked", flush=True)


if __name__ == "__main__":
    main()
