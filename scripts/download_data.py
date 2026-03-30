"""Download data files from Zenodo."""
import os
import urllib.request
import json
from config import ZENODO_RECORD_ID, DATA_ROOT

# Files needed for each notebook (excluding network models)
DATA_FILES = [
    "summary_stats.h5",
    "posteriors_ecc.h5",
    "posteriors_qc.h5",
    "posteriors_prec.h5",
    "ecc_pop.pkl",
    "prior_samples.pkl",
    "egw_conversions.h5",
    "glitch_GW190701.h5",
    "glitch_GW231114_043211.h5",
    "glitch_GW231223_032836.h5",
]


def get_zenodo_file_urls():
    """Fetch file URLs from the Zenodo API."""
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    with urllib.request.urlopen(url) as resp:
        record = json.loads(resp.read())
    return {f["key"]: f["links"]["self"] for f in record["files"]}


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return
    print(f"  Downloading {os.path.basename(dest_path)}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Saved to {dest_path}")


def download_all(data_root=None, include_networks=False):
    """Download all data files from Zenodo."""
    if data_root is None:
        data_root = DATA_ROOT

    os.makedirs(data_root, exist_ok=True)
    os.makedirs(os.path.join(data_root, "glitch_marginalization"), exist_ok=True)

    print(f"Downloading from Zenodo record {ZENODO_RECORD_ID}...")
    file_urls = get_zenodo_file_urls()

    for fname in DATA_FILES:
        if fname in file_urls:
            # Glitch files go into subdirectory
            if fname.startswith("glitch_"):
                dest = os.path.join(
                    data_root, "glitch_marginalization", fname.replace("glitch_", "")
                )
            else:
                dest = os.path.join(data_root, fname)
            download_file(file_urls[fname], dest)
        else:
            print(f"  WARNING: {fname} not found in Zenodo record")

    if include_networks:
        for fname, url in file_urls.items():
            if fname.startswith("SEOBNRv5") or fname == "MODEL_MANIFEST.md":
                dest = os.path.join(data_root, "networks", fname)
                download_file(url, dest)

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download data from Zenodo")
    parser.add_argument(
        "--output", "-o", default=DATA_ROOT, help="Output directory"
    )
    parser.add_argument(
        "--include-networks", action="store_true", help="Also download network models"
    )
    args = parser.parse_args()
    download_all(data_root=args.output, include_networks=args.include_networks)
