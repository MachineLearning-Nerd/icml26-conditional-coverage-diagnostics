#!/usr/bin/env python3
"""Prepare the eight Appendix-H classifier inputs from immutable public files.

The upstream repository deliberately omits ``experiments/data``.  This script
does not invent replacement datasets: it hashes each downloaded source and
checks the exact row/predictor counts published in Appendix H before emitting
the CSV contract expected by the upstream driver (a ``target`` column).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import arff
import pandas as pd


SPECS = {
    "protein": {
        "url": "https://www.openml.org/data/v1/download/22045534/physicochemical-protein.arff",
        "raw_name": "protein.arff",
        "sha256": "66d49b53fbe9f15e38136cf8253153ed787eb720882ebb3d77450515977c3a37",
        "target": "RMSD", "rows": 45730, "features": 9,
    },
    "deliverytime": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/rajatkumar30/food-delivery-time?datasetVersionNumber=1",
        "raw_name": "food-delivery-time-v1.zip",
        "sha256": "198f9179124285c82d3aec1b1c54a647b16c8ec5f4a0e9dd0dc4def6d9cdc978",
        "target": "Time_taken(min)", "rows": 45593, "features": 10,
        "extract": "deliverytime.csv",
    },
    "diamonds": {
        "url": "https://www.openml.org/data/v1/download/21792853/diamonds.arff",
        "raw_name": "diamonds.arff",
        "sha256": "0bbef7a24061e5cc4e8591eea59417948a2d2286401675b57163a555e1849b40",
        "target": "price", "rows": 53940, "features": 9,
    },
    "superconductivity": {
        "url": "https://www.openml.org/data/v1/download/22101847/superconduct.arff",
        "raw_name": "superconductivity.arff",
        "sha256": "d5bf995d3129f3f3fbcd97a293557fe83bf7a9deeb07d17bfc5493840d4b0ad6",
        "target": "critical_temp", "rows": 21263, "features": 81,
    },
    "ailerons": {
        "url": "https://www.openml.org/data/v1/download/52060/Ailerons.arff",
        "raw_name": "ailerons.arff",
        "sha256": "4aecc3e23a49a9a6300cb72c7dd57d6412fb16a0843c5aeb20508604e4fbb3f1",
        "target": "goal", "rows": 13750, "features": 40,
    },
    "o11": {
        "url": "https://www.openml.org/data/v1/download/1674848/QSAR-TID-11.sparse_arff",
        "raw_name": "o11.arff",
        "sha256": "177dd4c56df846ffff3f266b8384333f47c50150bfa94290fc27d41fc669444d",
        "target": "MEDIAN_PXC50", "rows": 5742, "features": 1025,
    },
    "miami2016": {
        "url": "https://www.openml.org/data/v1/download/22047757/MiamiHousing2016.arff",
        "raw_name": "miami2016.arff",
        "sha256": "db5125c57ff34a627f1841f3d29b59089c96144fd32bc5044a48762b2fd0a7a0",
        "target": "SALE_PRC", "rows": 13932, "features": 16,
    },
    "winequality": {
        "url": "https://www.openml.org/data/v1/download/22125275/wine_quality.arff",
        "raw_name": "winequality.arff",
        "sha256": "84076cebcb9229a7ef1a9b37693c92ac80cb39d384725b6cbf18123c7932289f",
        "target": "median_wine_quality", "rows": 6497, "features": 12,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(spec: dict, raw_dir: Path) -> Path:
    path = raw_dir / spec["raw_name"]
    if not path.exists():
        print(f"Downloading {path.name}")
        urllib.request.urlretrieve(spec["url"], path)
    observed = sha256(path)
    if observed != spec["sha256"]:
        raise RuntimeError(f"hash mismatch for {path}: {observed}")
    return path


def load_frame(spec: dict, raw_dir: Path) -> pd.DataFrame:
    raw = download(spec, raw_dir)
    if "extract" in spec:
        import zipfile
        with zipfile.ZipFile(raw) as archive:
            with archive.open(spec["extract"]) as handle:
                frame = pd.read_csv(handle)
    else:
        with raw.open() as handle:
            parsed = arff.load(handle)
        frame = pd.DataFrame(parsed["data"], columns=[item[0] for item in parsed["attributes"]])
    return frame.drop(columns=spec.get("drop", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SPECS), action="append")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in args.dataset or list(SPECS):
        spec = SPECS[name]
        frame = load_frame(spec, args.raw_dir)
        target = spec["target"]
        if target not in frame:
            raise RuntimeError(f"{name} lacks target {target!r}")
        features = frame.drop(columns=[target])
        if (len(frame), len(features.columns)) != (spec["rows"], spec["features"]):
            raise RuntimeError(
                f"{name} shape {(len(frame), len(features.columns))} != "
                f"{(spec['rows'], spec['features'])}"
            )
        output = pd.concat([features, frame[[target]].rename(columns={target: "target"})], axis=1)
        destination = args.output_dir / f"{name}.csv"
        output.to_csv(destination, index=False)
        manifest[name] = {
            "source_url": spec["url"],
            "raw_file": spec["raw_name"],
            "raw_sha256": spec["sha256"],
            "prepared_file": destination.name,
            "prepared_sha256": sha256(destination),
            "rows": len(output),
            "features": len(features.columns),
            "target": target,
        }
        print(f"prepared {name}: {len(output)} rows, {len(features.columns)} features")
    (args.output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
