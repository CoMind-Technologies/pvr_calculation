from dataclasses import dataclass

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


@dataclass
class Waveform:
    db_brain: np.ndarray
    db_scalp: np.ndarray
    mua_brain: np.ndarray
    mua_scalp: np.ndarray
    times: np.ndarray

    def __post_init__(self):
        self.get_mean_and_amps()

    @classmethod
    def from_file(cls, filepath):
        with h5py.File(filepath, "r") as f:
            db_brain = f["db_brain"][:]
            db_scalp = f["db_scalp"][:]
            mua_brain = f["mua_brain"][:]
            mua_scalp = f["mua_scalp"][:]
            times = f["times"][:]
        return cls(db_brain, db_scalp, mua_brain, mua_scalp, times)

    def get_mean_and_amps(self):
        self.db_brain_amp = np.ptp(self.db_brain)
        self.db_scalp_amp = np.ptp(self.db_scalp)
        self.db_brain_mean = np.mean(self.db_brain)
        self.db_scalp_mean = np.mean(self.db_scalp)
        self.mua_brain_amp = np.ptp(self.mua_brain)
        self.mua_scalp_amp = np.ptp(self.mua_scalp)
        self.mua_brain_mean = np.mean(self.mua_brain)
        self.mua_scalp_mean = np.mean(self.mua_scalp)

    def downsample(self, interval):
        self.db_brain = self.db_brain[::interval]
        self.db_scalp = self.db_scalp[::interval]
        self.mua_brain = self.mua_brain[::interval]
        self.mua_scalp = self.mua_scalp[::interval]
        self.times = self.times[::interval]
        self.get_mean_and_amps()

    def trim(self, start, end):
        self.db_brain = self.db_brain[start:end]
        self.db_scalp = self.db_scalp[start:end]
        self.mua_brain = self.mua_brain[start:end]
        self.mua_scalp = self.mua_scalp[start:end]
        self.times = self.times[start:end]
        self.get_mean_and_amps()

    def save(self, filepath):
        with h5py.File(filepath, "w") as f:
            f.create_dataset("db_brain", data=self.db_brain)
            f.create_dataset("db_scalp", data=self.db_scalp)
            f.create_dataset("mua_brain", data=self.mua_brain)
            f.create_dataset("mua_scalp", data=self.mua_scalp)
            f.create_dataset("times", data=self.times)

    def load(self, filepath):
        with h5py.File(filepath, "r") as f:
            self.db_brain = f["db_brain"][:]
            self.db_scalp = f["db_scalp"][:]
            self.mua_brain = f["mua_brain"][:]
            self.mua_scalp = f["mua_scalp"][:]
            self.times = f["times"][:]
        self.get_mean_and_amps()


def get_troughs(wf, plot_on=False):
    wf_norm = wf.db_brain - np.min(wf.db_brain)
    wf_norm /= np.max(wf_norm)
    ind, _ = find_peaks(-1 * ((wf_norm) - np.max(wf_norm)), height=0.6, distance=30)
    if plot_on:
        fig, ax = plt.subplots(1, 1)
        ax.plot(wf.db_brain)
        ax.scatter(ind, wf.db_brain[ind], c="r")
        fig.show()
    return ind
