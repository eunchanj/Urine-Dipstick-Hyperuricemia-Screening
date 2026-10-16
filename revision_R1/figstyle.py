"""One visual language for every figure in the uric acid R1 revision.

Ported from the liver JBI audit (module_liver/.../v1.0/working/figstyle.py) so the programme's papers share
one look. Every main figure is exactly 190 mm wide and is saved without a tight bounding box, so widths are
identical on the page. Fonts are embedded as TrueType (pdf.fonttype 42) so the PDF text stays editable.
Panel letters sit at the same offset from each axes' top-left corner, titles are left-aligned above their
axes, and font sizes come from one table (FS).
"""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

# Shared programme palette (same hex values as the liver paper).
NAVY, RED, GREEN, GREY, AMBER, INK = "#1f3b63", "#c0392b", "#2e7d5b", "#7f8c8d", "#e08e0b", "#2b2b2b"
WIDTH_IN = 190 / 25.4
FS = dict(title=9.5, label=8.5, tick=8, legend=7.4, note=7.0, panel=13)
# Figures go to <revision folder>/figures; PROJ_U_REVISION_DIR overrides the folder (original: parent of this one).
OUT = Path(os.environ.get("PROJ_U_REVISION_DIR", Path(__file__).resolve().parent.parent)) / "figures"


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FS["tick"],
        "axes.titlesize": FS["title"], "axes.labelsize": FS["label"],
        "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"], "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333", "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def figure(height_in, nrows=1, ncols=1, wspace=0.06, **gridspec):
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_IN, height_in),
                             constrained_layout=True, gridspec_kw=gridspec or None)
    eng = fig.get_layout_engine()
    eng.set(w_pad=0.04, h_pad=0.04, wspace=wspace, hspace=0.12)
    return fig, axes


def title(ax, text):
    ax.set_title(text, fontsize=FS["title"], fontweight="bold", color=NAVY, pad=7, loc="left")


def panel(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-28, 6),
                textcoords="offset points", fontsize=FS["panel"], fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)


def grid(ax, axis="y"):
    ax.grid(axis=axis, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  {name}  {WIDTH_IN:.2f} x {fig.get_figheight():.2f} in")
