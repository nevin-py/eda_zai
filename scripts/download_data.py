from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SOURCES = {
    "train_FD001.txt": [
        "https://raw.githubusercontent.com/umbertogriffo/Predictive-Maintenance-using-LSTM/master/Dataset/PM_train.txt",
        "https://raw.githubusercontent.com/Adithya-Thonse/RUL-Prediction/main/data/train_FD001.txt",
        "https://raw.githubusercontent.com/wilsonj809/Jet-Engine-RUL-Prediction/main/data/train_FD001.txt",
    ],
    "test_FD001.txt": [
        "https://raw.githubusercontent.com/umbertogriffo/Predictive-Maintenance-using-LSTM/master/Dataset/PM_test.txt",
        "https://raw.githubusercontent.com/Adithya-Thonse/RUL-Prediction/main/data/test_FD001.txt",
        "https://raw.githubusercontent.com/wilsonj809/Jet-Engine-RUL-Prediction/main/data/test_FD001.txt",
    ],
    "RUL_FD001.txt": [
        "https://raw.githubusercontent.com/umbertogriffo/Predictive-Maintenance-using-LSTM/master/Dataset/PM_truth.txt",
        "https://raw.githubusercontent.com/Adithya-Thonse/RUL-Prediction/main/data/RUL_FD001.txt",
        "https://raw.githubusercontent.com/wilsonj809/Jet-Engine-RUL-Prediction/main/data/RUL_FD001.txt",
    ],
}


def download_file(url: str, target_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=45) as response:
        payload = response.read()
    if not payload:
        raise ValueError(f"Downloaded zero bytes from {url}")
    target_path.write_bytes(payload)


def download_fd001(data_dir: Path, overwrite: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    for file_name, url_candidates in SOURCES.items():
        target_path = data_dir / file_name

        if target_path.exists() and target_path.stat().st_size > 0 and not overwrite:
            print(f"[skip] {file_name} already exists at {target_path}")
            continue

        downloaded = False
        for url in url_candidates:
            try:
                print(f"[try ] {file_name} <- {url}")
                download_file(url, target_path)
                print(f"[ ok ] saved {file_name} ({target_path.stat().st_size} bytes)")
                downloaded = True
                break
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                print(f"[fail] {url} :: {exc}")

        if not downloaded:
            raise RuntimeError(
                "Unable to download required CMAPSS file "
                f"{file_name}. Please place it manually into {data_dir}."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CMAPSS FD001 dataset files.")
    parser.add_argument("--data-dir", default="data", help="Directory for dataset files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)

    try:
        download_fd001(data_dir=data_dir, overwrite=args.overwrite)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n[error] {exc}")
        return 1

    print("\nCMAPSS FD001 dataset is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
