"""Download the release data files from Zenodo into the foldered layout.

Zenodo stores files flat; this maps each Zenodo key to its local path under
zenodo_data/ (event_data/, glitch/, selection_function/, hierarchical_inference/).
"""
import os
import urllib.request
import json
from config import ZENODO_RECORD_ID, DATA_ROOT

# {zenodo flat key : local path under DATA_ROOT}
ZENODO_FILES = {
    # event_data/
    "posteriors_eccentric.h5": "event_data/posteriors_eccentric.h5",
    "posteriors_quasicircular.h5": "event_data/posteriors_quasicircular.h5",
    "posteriors_precessing.h5": "event_data/posteriors_precessing.h5",
    "summary_statistics.h5": "event_data/summary_statistics.h5",
    "egw_conversions.h5": "event_data/egw_conversions.h5",
    "e_zeta_prior_hull.h5": "event_data/e_zeta_prior_hull.h5",
    "posteriors_log_uniform_eccentric.h5": "event_data/posteriors_log_uniform_eccentric.h5",
    "posteriors_population_reweighted.h5": "event_data/posteriors_population_reweighted.h5",
    # glitch/
    "glitch_GW190701.h5": "glitch/GW190701.h5",
    "glitch_GW231114_043211.h5": "glitch/GW231114_043211.h5",
    "glitch_GW231223_032836.h5": "glitch/GW231223_032836.h5",
    # selection_function/
    "injection_p_draw.h5": "selection_function/injection_p_draw.h5",
    "survival_function.h5": "selection_function/survival_function.h5",
    "fixed_injection_ecc_sweep.h5": "selection_function/fixed_injection_ecc_sweep.h5",
    # hierarchical_inference/
    "capture_ecc_table.h5": "hierarchical_inference/capture_ecc_table.h5",
    "gwtc4_hyperposterior.h5": "hierarchical_inference/gwtc4_hyperposterior.h5",
    "sigma_posterior.h5": "hierarchical_inference/sigma_posterior.h5",
    "branching_fraction_posterior.h5": "hierarchical_inference/branching_fraction_posterior.h5",
}


# Trained DINGO networks (optional, --include-networks). Zenodo keys are flat
# "<approximant>_<model>_<file>" names; locally they live under
# networks/<approximant>/<model>/ (the layout config.NETWORKS_DIR documents and
# dingo expects when loading a model).
NETWORK_FILES = {
    "MODEL_MANIFEST.md": "networks/MODEL_MANIFEST.md",
    "SEOBNRv5EHM_O4_HL_0_model_latest.pt": "networks/SEOBNRv5EHM/O4_HL_0/model_latest.pt",
    "SEOBNRv5EHM_O4_HL_0_train_settings.yaml": "networks/SEOBNRv5EHM/O4_HL_0/train_settings.yaml",
    "SEOBNRv5EHM_O4_HL_0_time_model_latest.pt": "networks/SEOBNRv5EHM/O4_HL_0_time/model_latest.pt",
    "SEOBNRv5EHM_O4_HL_0_time_train_settings.yaml": "networks/SEOBNRv5EHM/O4_HL_0_time/train_settings.yaml",
    "SEOBNRv5EHM_O4_HL_2_model_latest.pt": "networks/SEOBNRv5EHM/O4_HL_2/model_latest.pt",
    "SEOBNRv5EHM_O4_HL_2_train_settings.yaml": "networks/SEOBNRv5EHM/O4_HL_2/train_settings.yaml",
    "SEOBNRv5EHM_O4_HL_2_time_model_latest.pt": "networks/SEOBNRv5EHM/O4_HL_2_time/model_latest.pt",
    "SEOBNRv5EHM_O4_HL_2_time_train_settings.yaml": "networks/SEOBNRv5EHM/O4_HL_2_time/train_settings.yaml",
    "SEOBNRv5HM_O4_HL_0_model_latest.pt": "networks/SEOBNRv5HM/O4_HL_0/model_latest.pt",
    "SEOBNRv5HM_O4_HL_0_train_settings.yaml": "networks/SEOBNRv5HM/O4_HL_0/train_settings.yaml",
    "SEOBNRv5HM_O4_HL_1_model_latest.pt": "networks/SEOBNRv5HM/O4_HL_1/model_latest.pt",
    "SEOBNRv5HM_O4_HL_1_train_settings.yaml": "networks/SEOBNRv5HM/O4_HL_1/train_settings.yaml",
    "SEOBNRv5HM_O4_HL_1_time_2_model_latest.pt": "networks/SEOBNRv5HM/O4_HL_1_time_2/model_latest.pt",
    "SEOBNRv5HM_O4_HL_1_time_2_train_settings.yaml": "networks/SEOBNRv5HM/O4_HL_1_time_2/train_settings.yaml",
    "SEOBNRv5PHM_v5PHM_H1L1_model_latest.pt": "networks/SEOBNRv5PHM/v5PHM_H1L1/model_latest.pt",
    "SEOBNRv5PHM_v5PHM_H1L1_train_settings.yaml": "networks/SEOBNRv5PHM/v5PHM_H1L1/train_settings.yaml",
    "SEOBNRv5PHM_v5PHM_H1L1_time_model_latest.pt": "networks/SEOBNRv5PHM/v5PHM_H1L1_time/model_latest.pt",
}


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


def download_all(include_networks=False):
    """Download all release data files from Zenodo into the foldered layout."""
    data_root = DATA_ROOT
    os.makedirs(data_root, exist_ok=True)

    print(f"Downloading from Zenodo record {ZENODO_RECORD_ID}...")
    file_urls = get_zenodo_file_urls()

    for key, rel in ZENODO_FILES.items():
        if key in file_urls:
            download_file(file_urls[key], os.path.join(data_root, rel))
        else:
            print(f"  WARNING: {key} not found in Zenodo record")

    if include_networks:
        for key, rel in NETWORK_FILES.items():
            if key in file_urls:
                download_file(file_urls[key], os.path.join(data_root, rel))
            else:
                print(f"  WARNING: {key} not found in Zenodo record")

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download data from Zenodo")
    parser.add_argument("--include-networks", action="store_true",
                        help="Also download network models")
    args = parser.parse_args()
    download_all(include_networks=args.include_networks)
