import os

ZENODO_RECORD_ID = "19340012"
ZENODO_DOI = "10.5281/zenodo.19340012"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"

# Set DATA_ROOT to point to wherever you downloaded the Zenodo data.
# Override with the DINGO_ECC_DATA_ROOT environment variable, or place
# the Zenodo data in a sibling directory called "zenodo_data".
DATA_ROOT = os.environ.get(
    "DINGO_ECC_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "zenodo_data"),
)
