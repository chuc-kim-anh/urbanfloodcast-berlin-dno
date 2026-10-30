from pathlib import Path
import bz2
import requests
import struct
import zlib
import zipfile
from remotezip import RemoteZip

URL = "https://zenodo.org/api/records/15700880/files/UrbanFloodCast_Dataset.zip/content"
ROOT = Path("data/raw/Input_BerlinII")
NAMES = ["moa_bottom.tif", "moa_rough.tif", "moa_building.tif"]
CHUNK = 8 * 1024 * 1024


def get_range(start, end):
    response = requests.get(URL, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=180)
    if response.status_code != 206 or len(response.content) != end - start:
        raise RuntimeError(f"Range failure {response.status_code}: requested {end-start}, got {len(response.content)}")
    return response.content


def fetch(info):
    target = ROOT / info.filename.rsplit("/", 1)[-1]
    if target.exists() and target.stat().st_size == info.file_size:
        print(f"cached {target.name}", flush=True)
        return
    header = get_range(info.header_offset, info.header_offset + 30)
    sig, _, _, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
    if sig != 0x04034B50:
        raise RuntimeError(f"Invalid ZIP local header for {info.filename}")
    start = info.header_offset + 30 + name_len + extra_len
    compressed = bytearray()
    for pos in range(start, start + info.compress_size, CHUNK):
        end = min(pos + CHUNK, start + info.compress_size)
        compressed.extend(get_range(pos, end))
        print(f"{target.name}: {len(compressed) / 1e6:.1f}/{info.compress_size / 1e6:.1f} MB", flush=True)
    if method == zipfile.ZIP_STORED:
        data = bytes(compressed)
    elif method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif method == zipfile.ZIP_BZIP2:
        data = bz2.decompress(compressed)
    else:
        raise RuntimeError(f"Unsupported ZIP method {method}")
    if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise RuntimeError(f"CRC/size validation failed for {target.name}")
    ROOT.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"saved {target} ({len(data) / 1e6:.1f} MB; CRC OK)", flush=True)


def main():
    with RemoteZip(URL) as archive:
        infos = [archive.getinfo(f"UrbanFloodCast_Dataset/BerlinII/Input_BerlinII/{name}") for name in NAMES]
    for info in infos:
        fetch(info)


if __name__ == "__main__":
    main()
