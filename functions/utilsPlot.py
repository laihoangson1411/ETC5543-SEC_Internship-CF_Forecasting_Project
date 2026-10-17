"""
Plot utilities shared by every notebook.

Colour palette, plot style, figure titles and table captions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import ScaledTranslation
from IPython.display import display, Markdown

# ── Colours: wind / main series = purple, solar / contrast = orange ──
PURPLE = '#391B56'
ORANGE = '#E67E22'
GREY = '#888888'
ASSET_COLORS = ['#2171B5', '#D62728', '#7B3FA0', '#E6B800']    # one colour per asset in a panel

# Season code (1=Summer, 2=Autumn, 3=Winter, 4=Spring) -> (name, colour, marker)
SEASON_STYLE = {1: ('Summer', '#C44E52', 'o'), 2: ('Autumn', '#DD8452', 's'),
                3: ('Winter', '#4C72B0', 'D'), 4: ('Spring', '#55A868', '^')}

# Feature importance bars: signed coefficient > 0 / < 0 (linear models), share of importance (tree models)
IMPORTANCE_COLORS = {'Raises CF (linear models)': '#2ca02c', 'Lowers CF (linear models)': '#d62728',
                     'Share of importance (tree models)': 'steelblue'}

# Correlation colour map: -1 = orange, 0 = white, +1 = purple
CORR_CMAP = LinearSegmentedColormap.from_list('orange_purple', [ORANGE, 'white', PURPLE])

BASE_RC = {'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': 0.3,
           'axes.spines.top': False, 'axes.spines.right': False}
SMALL_FONTS_RC = {'font.size': 9, 'axes.titlesize': 9.5, 'xtick.labelsize': 8,
                  'ytick.labelsize': 8, 'legend.fontsize': 8}


def set_style(small_fonts: bool = False, **rc):
    """Apply the project plot style; small_fonts adds the compact font sizes, rc overrides single settings."""
    plt.rcParams.update({**BASE_RC, **(SMALL_FONTS_RC if small_fonts else {}), **rc})


def add_titles(fig, title, subtitle=None):
    """Bold figure title with an optional one-line insight (subtitle) below it.

    Call after fig.tight_layout(); the text is placed just above the top of the figure.
    """
    def put(text, offset_pt, **kw):
        # figure-fraction x, y = top edge of the figure, then shifted up by offset_pt points
        shift = fig.transFigure + ScaledTranslation(0, offset_pt / 72, fig.dpi_scale_trans)
        fig.text(0.5, 1.0, text, transform=shift, ha='center', va='bottom', **kw)
    put(title, 8 + (16 if subtitle else 0), fontsize=12, fontweight='bold')
    if subtitle:
        put(subtitle, 6, fontsize=10, color='#555555', style='italic')


def bottom_legend(fig, handles=None, labels=None, axes=None, ncol=None, **kw):
    """One legend below the figure; call after fig.tight_layout(), like add_titles.

    Without `handles`, it collects the labelled artists of `axes` (default: every axes of the figure, twin axes
    included), shows each label once and removes the per-axes legends. With `axes`, the legend is centred under
    those axes (one legend per panel when the panels have different labels); otherwise under the whole figure.
    """
    if handles is None:
        handles, labels = [], []
        for ax in (axes if axes is not None else fig.axes):
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in labels:
                    handles.append(h)
                    labels.append(lab)
    elif labels is None:
        labels = [h.get_label() for h in handles]
    x = 0.5
    if axes is not None:
        boxes = [ax.get_position() for ax in axes]
        x = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    kw = {'frameon': False, 'fontsize': 8, **kw}
    return fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(x, 0), ncol=ncol or min(len(labels), 4), **kw)


def table_title(title, note=None):
    """Print a bold table caption (and an optional italic note) above display(df)."""
    display(Markdown(f'**{title}**' + (f'  \n*{note}*' if note else '')))


def fy_separators(ax, first_fy, last_fy, labels=False):
    """Dotted line at 1 July of each financial year first_fy..last_fy; labels=True writes 'FY27' at the top."""
    for fy in range(first_fy, last_fy + 1):
        x = pd.Timestamp(year=fy - 1, month=7, day=1)
        ax.axvline(x, color='#999999', lw=0.7, ls=':', alpha=0.6, zorder=0)
        if labels:
            ax.text(x + pd.Timedelta(days=5), 0.99, f'FY{fy % 100}', transform=ax.get_xaxis_transform(),
                    fontsize=8, color='#777777', va='top')


def annotated_heatmap(ax, table, cmap, vmin, vmax, fontsize=9, rotate_x=False):
    """Heatmap of a small table with its value written in each cell (white text on dark cells).

    Returns the image, for a colour bar.
    """
    im = ax.imshow(table.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=45 if rotate_x else 0, ha='right' if rotate_x else 'center')
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.grid(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if np.isfinite(v):
                dark = abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.2
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=fontsize, color='white' if dark else '#222222')
    return im


def importance_bars(ax, imp, signed):
    """Horizontal bars of a feature importance, largest at the top, with the value at the end of each bar.

    Signed importances (linear models) are green when positive and red when negative; shares (tree models) are blue.
    """
    imp = imp.reindex(imp.abs().sort_values().index)
    pos, neg, share = IMPORTANCE_COLORS.values()
    colors = [pos if v >= 0 else neg for v in imp.values] if signed else share
    ax.barh(imp.index, imp.values, color=colors, alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.6)
    for y, v in enumerate(imp.values):
        ax.annotate(f'{v:+.3f}' if signed else f'{v:.3f}', xy=(v, y), xytext=(3 if v >= 0 else -3, 0),
                    textcoords='offset points', ha='left' if v >= 0 else 'right', va='center', fontsize=7)
    ax.margins(x=0.2)                                   # room for the numbers at both ends
    ax.grid(axis='y', visible=False)
