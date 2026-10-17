"""
Plot utilities shared by every notebook.

Colour palette, plot style, figure titles, table captions and the data-dictionary tables of the slides.
"""

import csv
import html
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import ScaledTranslation
from IPython.display import display, Markdown, HTML

# ── Colours: wind / main series = purple, solar / contrast = orange ──
PURPLE = '#391B56'
ORANGE = '#E67E22'
GREY = '#888888'
ASSET_COLORS = ['#2171B5', '#D62728', '#7B3FA0', '#E6B800']    # one colour per asset in a panel

# Season code (1=Summer, 2=Autumn, 3=Winter, 4=Spring) -> (name, colour, marker)
SEASON_STYLE = {1: ('Summer', '#C44E52', 'o'), 2: ('Autumn', '#DD8452', 's'),
                3: ('Winter', '#4C72B0', 'D'), 4: ('Spring', '#55A868', '^')}

# Feature importance bars (theme colours): signed coefficient > 0 / < 0 (linear models), share of importance (tree models)
IMPORTANCE_COLORS = {'Raises CF (linear models)': PURPLE, 'Lowers CF (linear models)': ORANGE,
                     'Share of importance (tree models)': '#7D6B98'}

# One colour per model (forecast plots of WindCF / SolarCF / CF_Forecast); baselines in grey
MODEL_COLORS = {'Linear': '#4C72B0', 'Ridge': '#E67E22', 'RF': '#55A868', 'XGBoost': '#C44E52',
                'Ensemble': '#8172B3', 'Exponential': '#AAAAAA', 'Seasonal Naive': '#777777'}

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


def importance_bars(ax, imp, signed, stars=None, fontsize=7):
    """Horizontal bars of a feature importance, largest at the top, with the value at the end of each bar.

    Signed importances (linear models) are purple when positive and orange when negative; shares (tree models) are
    lilac (IMPORTANCE_COLORS). `stars` (feature -> '*', '**', ...) is written after the value.
    """
    imp = imp.reindex(imp.abs().sort_values().index)
    pos, neg, share = IMPORTANCE_COLORS.values()
    colors = [pos if v >= 0 else neg for v in imp.values] if signed else share
    ax.set_axisbelow(True)                              # grid lines behind the bars
    ax.barh(imp.index, imp.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.6)
    for y, (feat, v) in enumerate(imp.items()):
        text = f'{v:+.3f}' if signed else f'{v:.3f}'
        if stars is not None and stars.get(feat):
            text = f'{text} {stars[feat]}' if v >= 0 else f'{stars[feat]} {text}'
        ax.annotate(text, xy=(v, y), xytext=(3 if v >= 0 else -3, 0), textcoords='offset points',
                    ha='left' if v >= 0 else 'right', va='center', fontsize=fontsize)
    ax.margins(x=0.25)                                  # room for the numbers (and stars) at both ends
    ax.grid(axis='y', visible=False)


def _cell_html(text):
    """Cell text to HTML: **text** is bold; a cell whose lines all start with '- ' is a bulleted list."""
    lines = [re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html.escape(line.strip(), quote=False))
             for line in text.strip().splitlines()]
    if lines and all(line.startswith('- ') for line in lines):
        return '<ul>' + ''.join(f'<li>{line[2:]}</li>' for line in lines) + '</ul>'
    return '<br>'.join(lines)


def dictionary_table(path, bold_cols=2, css_class='dd'):
    """Data-dictionary table of the slides, read from a CSV file (the style is in css/presentation.css).

    The CSV stays plain text:
      - an empty header extends the header on its left (colspan);
      - an empty cell extends the cell above it (rowspan), e.g. the group name is written once;
      - the first `bold_cols` columns are bold: by default the group and the ID (classes grp and id);
        bold_cols=1 for a table whose first column is the only label (e.g. the model summary).
    """
    with open(path, newline='', encoding='utf-8') as f:
        header, *rows = csv.reader(f)
    n_cols = len(header)
    rows = [r + [''] * (n_cols - len(r)) for r in rows]

    heads = []                                        # [label, colspan]
    for h in header:
        if h.strip() or not heads:
            heads.append([h.strip(), 1])
        else:
            heads[-1][1] += 1
    out = ['<tr>' + ''.join(f'<th colspan="{c}">{_cell_html(h)}</th>' if c > 1 else f'<th>{_cell_html(h)}</th>'
                            for h, c in heads) + '</tr>']

    span = [[0] * n_cols for _ in rows]               # rowspan of each cell; 0 = covered by the cell above
    for j in range(n_cols):
        top = 0
        for i, row in enumerate(rows):
            if i == 0 or row[j].strip():
                top = i
            span[top][j] += 1
    cls = {j: ' class="grp"' if j == 0 else ' class="id"' for j in range(bold_cols)}
    for i, row in enumerate(rows):
        cells = []
        for j in range(n_cols):
            if span[i][j]:
                rowspan = f' rowspan="{span[i][j]}"' if span[i][j] > 1 else ''
                cells.append(f'<td{cls.get(j, "")}{rowspan}>{_cell_html(row[j])}</td>')
        out.append('<tr>' + ''.join(cells) + '</tr>')
    # data-quarto-disable-processing: Quarto keeps the HTML table as it is (otherwise it sets equal column widths)
    return HTML(f'<table class="{css_class}" data-quarto-disable-processing="true">' + ''.join(out) + '</table>')


def mae_ranking_table(mae, baselines, models):
    """MAE table of the slides: the baselines first, then the models from the lowest MAE (shaded purple, darker =
    lower MAE; the lowest is bold). Colours are inline, so the table looks the same in a notebook and on a slide."""
    head = f'background:{PURPLE}; color:#fff; font-weight:700; text-align:left; padding:6px 12px;'
    cell = 'padding:6px 12px; border:0;'
    ranked = mae.reindex(models).dropna().sort_values()
    lo, hi = ranked.min(), ranked.max()
    shade = LinearSegmentedColormap.from_list('mae', ['#EBE7F0', '#B5A3C6'])      # worst -> best

    def row(name, value, bg='#fff', bold=False):
        v = f'<b>{value:.3f}</b>' if bold else f'{value:.3f}'
        return (f'<tr><td style="{cell} color:{PURPLE};">{name}</td>'
                f'<td style="{cell} background:{bg}; text-align:center;">{v}</td></tr>')

    out = [f'<tr><th style="{head}">Benchmark</th><th style="{head} text-align:center;">MAE</th></tr>']
    out += [row(b, mae[b]) for b in baselines]
    out.append(f'<tr><th style="{head}">Model</th><th style="{head} text-align:center;">MAE</th></tr>')
    for m, v in ranked.items():
        t = 1 - (v - lo) / (hi - lo) if hi > lo else 1
        r, g, b, _ = shade(t)
        out.append(row(m, v, f'rgb({r * 255:.0f},{g * 255:.0f},{b * 255:.0f})', bold=v == lo))
    return HTML('<table class="mae" style="border-collapse:collapse;" data-quarto-disable-processing="true">'
                + ''.join(out) + '</table>')
