import os

# New Zenodo record for the full-reproduction release (event posteriors +
# selection function + hierarchical velocity-dispersion inference). Set this once
# separate for the previous paper version.
ZENODO_RECORD_ID = os.environ.get("DINGO_ECC_ZENODO_RECORD_ID", "21221337")
ZENODO_DOI = f"10.5281/zenodo.{ZENODO_RECORD_ID}"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"

# The Zenodo data lives in zenodo_data/ one level up from code/ (i.e. two levels
# up from scripts/); download_data.py fetches it there. Override with
# DINGO_ECC_DATA_ROOT to point at a copy elsewhere.
DATA_ROOT = os.environ.get(
    "DINGO_ECC_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "zenodo_data"),
)

# Subdirectories of the release zenodo_data tree (folders + descriptive names).
EVENT_DATA_DIR = os.path.join(DATA_ROOT, "event_data")
GLITCH_DIR = os.path.join(DATA_ROOT, "glitch")
SELECTION_DIR = os.path.join(DATA_ROOT, "selection_function")
HIERARCHICAL_DIR = os.path.join(DATA_ROOT, "hierarchical_inference")

# Trained DINGO neural networks (pulled with download_data.py --include-networks).
# Layout: NETWORKS_DIR/<approximant>/<model>/model_latest.pt (+ train_settings.yaml).
# Not needed for the figures; used only to (re)analyze events with the network.
# See networks/MODEL_MANIFEST.md for the model table.
NETWORKS_DIR = os.path.join(DATA_ROOT, "networks")
