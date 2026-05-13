"""Shared matplotlib typography defaults for project plotting scripts."""

from __future__ import annotations

BASE_FONT_SIZE = 13
AXES_TITLE_SIZE = 15
AXES_LABEL_SIZE = 14
TICK_LABEL_SIZE = 13
LEGEND_FONT_SIZE = 12
FIGURE_TITLE_SIZE = 17

ANNOTATION_FONT_SIZE = 11
SMALL_ANNOTATION_FONT_SIZE = 10
DENSE_ANNOTATION_FONT_SIZE = 9


def apply_large_plot_text(plt) -> None:
    """Apply consistent large text defaults for readability in generated plots."""
    plt.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": AXES_TITLE_SIZE,
            "axes.labelsize": AXES_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "figure.titlesize": FIGURE_TITLE_SIZE,
        }
    )