import pathlib

REPO_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = REPO_PATH / "data"
BFI_PATH = REPO_PATH / "data" / "bfi"
RESULTS_PATH = REPO_PATH / "data" / "segmentation_and_pvr_results"
FIGURE_PATH = REPO_PATH / "data" / "figures"
FIGURE_PATH.mkdir(exist_ok=True, parents=True)
