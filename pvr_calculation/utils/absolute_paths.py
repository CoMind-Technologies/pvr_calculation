import pathlib

REPO_PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = REPO_PATH / "data"
WAVEFORM_PATH = REPO_PATH / "data" / "waveforms"
IRF_PATH = REPO_PATH / "data" / "irf"
FIGURE_PATH = REPO_PATH / "data" / "figures"
FIGURE_PATH.mkdir(exist_ok=True, parents=True)
