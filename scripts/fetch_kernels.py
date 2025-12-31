#!/usr/bin/env python3
"""
Download SPICE kernels (BSP/TLS/TPC/BPC) and generate a reproducibility manifest.

Usage:
  python scripts/fetch_kernels.py
  python scripts/fetch_kernels.py --force
  python scripts/fetch_kernels.py --base-dir data/kernels
  python scripts/fetch_kernels.py --verify-only

Notes:
- SHA256 is generated locally after download (NAIF doesn't publish them).
- The manifest is written to data/manifest.yml by default.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import yaml  # pip install pyyaml
except ImportError:
    yaml = None


# ----------------------------- Configuration -----------------------------

@dataclass(frozen=True)
class KernelItem:
    group: str          # e.g. "general", "galileo"
    url: str            # download URL
    filename: str       # local filename (case-sensitive for metakernel!)
    note: str = ""      # optional short note


KERNELS: List[KernelItem] = [
    # General
    KernelItem("general", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls", "naif0012.tls"),
    KernelItem("general", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc", "pck00010.tpc"),
    KernelItem("general", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp", "de440s.bsp"),
    KernelItem("general", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_1962_250826_2125_combined.bpc",
               "earth_1962_250826_2125_combined.bpc"),

    # Galileo
    KernelItem("galileo", "https://naif.jpl.nasa.gov/pub/naif/GLL/kernels/spk/s970311a.bsp", "s970311a.bsp"),

    # Cassini
    KernelItem("cassini", "https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/spk/000331R_SK_V2P12_EP15.bsp",
               "000331R_SK_V2P12_EP15.bsp"),

    # Messenger (PDS)
    KernelItem("messenger",
               "https://naif.jpl.nasa.gov/pub/naif/pds/data/mess-e_v_h-spice-6-v1.0/messsp_1000/data/spk/msgr_040803_071001_120401.bsp",
               "msgr_040803_071001_120401.bsp"),

    # NEAR
    KernelItem("near", "https://naif.jpl.nasa.gov/pub/naif/NEAR/misc/pds/individual/NEAR_CRUISE_NAV_V1.BSP",
               "NEAR_CRUISE_NAV_V1.BSP"),

    # Juno
    KernelItem("juno", "https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/spk/spk_merge_110805_171017_130515.bsp",
               "spk_merge_110805_171017_130515.bsp"),

    # Rosetta
    KernelItem("rosetta", "https://naif.jpl.nasa.gov/pub/naif/ROSETTA/kernels/spk/ORER_______________00031.BSP",
               "ORER_______________00031.BSP"),
    KernelItem("rosetta", "https://naif.jpl.nasa.gov/pub/naif/ROSETTA/kernels/spk/ORFR_______________00067.BSP",
               "ORFR_______________00067.BSP"),
    KernelItem("rosetta", "https://naif.jpl.nasa.gov/pub/naif/ROSETTA/kernels/spk/ORGR_______________00096.BSP",
               "ORGR_______________00096.BSP"),
]


# ----------------------------- Helpers -----------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Stream download with a basic progress indicator (bytes)
    req = Request(url, headers={"User-Agent": "flyby-proxy-kernel-fetch/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        total = resp.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None

        tmp = dest.with_suffix(dest.suffix + ".part")
        bytes_done = 0
        t0 = time.time()

        with tmp.open("wb") as out:
            while True:
                buf = resp.read(1024 * 256)
                if not buf:
                    break
                out.write(buf)
                bytes_done += len(buf)

                # Lightweight progress (no fancy deps)
                if total_int:
                    pct = 100.0 * bytes_done / total_int
                    dt = max(time.time() - t0, 1e-6)
                    rate = bytes_done / dt / (1024 * 1024)
                    sys.stdout.write(f"\r  {dest.name}: {pct:6.2f}%  {rate:6.2f} MB/s")
                    sys.stdout.flush()

        if total_int:
            sys.stdout.write("\n")

        tmp.replace(dest)


def load_manifest(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    if yaml is None:
        raise RuntimeError("PyYAML is required to read/write manifest.yml. Install with: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
        if not isinstance(data, list):
            raise ValueError("manifest.yml must contain a YAML list at top level.")
        return data


def save_manifest(path: Path, entries: List[Dict]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read/write manifest.yml. Install with: pip install pyyaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort entries for stable diffs
    entries_sorted = sorted(entries, key=lambda e: (e.get("group", ""), e.get("filename", "")))
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(entries_sorted, f, sort_keys=False, width=120)


def index_manifest(entries: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    idx: Dict[Tuple[str, str], Dict] = {}
    for e in entries:
        key = (e.get("group", ""), e.get("filename", ""))
        idx[key] = e
    return idx


# ----------------------------- Main workflow -----------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Download SPICE kernels and create/verify a SHA256 manifest.")
    p.add_argument("--base-dir", default="data/kernels", help="Base directory for downloaded kernels.")
    p.add_argument("--manifest", default="data/manifest.yml", help="Path to manifest YAML.")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    p.add_argument("--verify-only", action="store_true", help="Do not download; only verify existing files.")
    p.add_argument("--timeout", type=int, default=120, help="Download timeout seconds.")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    manifest_path = Path(args.manifest)

    existing = load_manifest(manifest_path)
    idx = index_manifest(existing)

    updated_entries: List[Dict] = []
    problems: List[str] = []

    for item in KERNELS:
        local_path = base_dir / item.group / item.filename

        # Download if needed
        if args.verify_only:
            if not local_path.exists():
                problems.append(f"Missing: {local_path}")
                continue
        else:
            if local_path.exists() and not args.force:
                print(f"[skip] {local_path} (exists)")
            else:
                print(f"[get ] {item.url}")
                try:
                    download_file(item.url, local_path, timeout=args.timeout)
                except Exception as ex:
                    problems.append(f"Download failed for {item.url}: {ex}")
                    continue

        # Hash + size
        try:
            h = sha256_file(local_path)
            size_bytes = local_path.stat().st_size
        except Exception as ex:
            problems.append(f"Hash/stat failed for {local_path}: {ex}")
            continue

        # Verify against manifest if present
        key = (item.group, item.filename)
        prior = idx.get(key)
        if prior and "sha256" in prior and prior["sha256"]:
            if str(prior["sha256"]).lower() != h.lower():
                problems.append(
                    f"SHA256 mismatch for {local_path}: manifest={prior['sha256']} actual={h}"
                )

        entry = {
            "group": item.group,
            "filename": item.filename,
            "url": item.url,
            "sha256": h,
            "size_bytes": int(size_bytes),
        }
        if item.note:
            entry["note"] = item.note

        updated_entries.append(entry)
        print(f"[ok  ] {local_path}  sha256={h[:12]}...  size={size_bytes/1024/1024:.2f} MB")

    # Write manifest (unless verify-only and no yaml)
    if not args.verify_only:
        save_manifest(manifest_path, updated_entries)
        print(f"\nWrote manifest: {manifest_path}")

    if problems:
        print("\nProblems:")
        for pr in problems:
            print(f"  - {pr}")
        return 2

    print("\nAll kernels present and hashed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
