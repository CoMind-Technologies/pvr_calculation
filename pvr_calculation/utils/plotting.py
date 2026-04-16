import datetime
import os

from matplotlib import pyplot as plt
from matplotlib import rc

from mc_analysis.utils.absolute_paths import FIGURE_PATH


def date_str():
    return datetime.datetime.now().strftime("%Y_%m_%d")


def save_fig(fig, save_name, dpi=500, save_folder=FIGURE_PATH):
    """Save a matplotlib figure with specified dpi."""
    save_full_path = os.path.join(save_folder, save_name)
    fig.savefig(save_full_path, dpi=dpi)
    print(f"Figure saved to {save_full_path}")


def set_default_fontsizes(fts, fts_tick=None, fts_labels=None, fts_title=None, fts_txt=None, fts_leg=None):
    """
    Set a default fontsize scheme for matplotlib; will apply to any new figure
    created after this call.

    If only fts is supplied, the tick size will be set to 0.8 * fts, and all
    other fontsizes (x & y labels, title, text, legend) to fts. If other values
    are provided, they will be used.

    Parameters
    ----------
    fts : float
        Main fontsize to use.
    fts_tick : float, optional
        Fontsize for the tick values. The default is None.
    fts_labels : float, optional
        Fontsize for the axis labels. The default is None.
    fts_title : float, optional
        Fontsize for the title. The default is None.
    fts_txt : float, optional
        Fontsize for text in the figure. The default is None.
    fts_leg : float, optional
        Fontsize for the legend. The default is None.

    Returns
    -------
    None.

    """

    # default values
    if fts_tick is None:
        fts_tick = 0.8 * fts

    # fts for all the rest:
    if fts_labels is None:
        fts_labels = fts

    if fts_title is None:
        fts_title = fts

    if fts_txt is None:
        fts_txt = fts

    if fts_leg is None:
        fts_leg = fts

    plt.rc("xtick", labelsize=fts_tick)  # fontsize of the x tick values
    plt.rc("ytick", labelsize=fts_tick)  # fontsize of the y tick values

    plt.rc("axes", labelsize=fts_labels)  # fontsize of the x and y axis labels
    plt.rc("axes", titlesize=fts_title)  # fontsize of the title

    plt.rc("font", size=fts_txt)  # default text size

    plt.rc("figure", titlesize=fts_title)  # fontsize of the figure title

    plt.rc("legend", fontsize=fts_leg)  # fontsize of the legend

    return


def set_plot_style():
    set_default_fontsizes(9)
    rc("font", **{"family": "serif", "serif": ["Times"]})
    return
