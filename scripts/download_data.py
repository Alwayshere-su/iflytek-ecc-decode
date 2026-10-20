#!/usr/bin/env python3
"""Download POLAR(64,32) training shards from the HuggingFace mirror.

The data is not redistributed in this repository. By default this downloads
the two shards used by the reference pipeline (000 and 001) into ./train/.

Examples:
    python scripts/download_data.py --shards 0-1
    python scripts/download_data.py --shards 0-9 --workers 4
    python scripts/download_data.py --shards 0-499 --workers 8
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "https://hf-mirror.com/datasets/aprofeta/ecc-dataset/resolve/main/train"
PREFIXES = ("train_codeword_x", "train_noisy_y")


def parse_shards(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def remote_size(url: str) -> int:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=30) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else -1
    except Exception:
        return -1


def download_one(url: str, dest: Path, force: bool) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    if not force and dest.exists() and expected > 0 and dest.stat().st_size == expected:
        return f"skip  {dest.name}"

    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl:
        cmd = [
            curl,
            "-L",
            "--retry", "5",
            "--retry-all-errors",
            "--retry-delay", "3",
            "--connect-timeout", "20",
            "--max-time", "1800",
            "--continue-at", "-",
            "-o", str(dest),
            url,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    else:
        # Fallback without resume support. curl is available on Windows 10+,
        # Linux and macOS in practice; this path is for minimal environments.
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        tmp.replace(dest)
    return f"ok    {dest.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", default="0-1", help="e.g. 0-1, 0-9, 0-499")
    parser.add_argument("--out", type=Path, default=ROOT / "train")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="redownload even if size matches")
    args = parser.parse_args()

    shards = parse_shards(args.shards)
    if not shards:
        parser.error("no shards selected")

    tasks = []
    for shard in shards:
        for prefix in PREFIXES:
            name = f"{prefix}_shard_{shard:03d}.csv"
            tasks.append((f"{args.base_url}/{name}", args.out / name))

    print(f"mirror : {args.base_url}")
    print(f"shards : {shards[0]}..{shards[-1]} ({len(shards)} shard(s))")
    print(f"output : {args.out}")
    print(f"files  : {len(tasks)}")

    start = time.time()
    ok = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(download_one, url, dest, args.force): dest for url, dest in tasks}
        for future in as_completed(futures):
            try:
                message = future.result()
                if message.startswith("skip"):
                    skipped += 1
                else:
                    ok += 1
            except Exception as exc:
                failed += 1
                print(f"FAIL {futures[future].name}: {exc}", file=sys.stderr)
            done = ok + skipped + failed
            if done % 10 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] ok={ok} skip={skipped} fail={failed} "
                      f"{time.time() - start:.0f}s", flush=True)

    print(f"done in {time.time() - start:.0f}s: ok={ok} skip={skipped} fail={failed}")
    if failed:
        print("Some files failed; re-run the same command to resume.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
