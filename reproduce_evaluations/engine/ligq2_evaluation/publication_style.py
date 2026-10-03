"""Readable, locally scoped typography for retained publication figures."""

from functools import wraps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def font_settings(font_size=16):
    if not 6 <= font_size <= 40:
        raise ValueError("Font size must be between 6 and 40 points.")
    return {
        "font.size": font_size,
        "axes.labelsize": font_size + 2,
        "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size,
    }


def readable_labels(function):
    """Accept an optional font_size keyword without altering plotted data."""
    @wraps(function)
    def wrapped(*args, font_size=16, **kwargs):
        with plt.rc_context(font_settings(font_size)):
            return function(*args, **kwargs)
    return wrapped
