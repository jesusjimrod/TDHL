"""Figs. 4, 6, 7 and 8 of the paper.

Each figure is a page of fixed size; the axes boxes and the positions of the labels, legends and qubit glyphs are given
in page points (origin top left). The plots keep the styles of the notebooks, drawn at a figure-specific scale (1.2 for
Fig. 4, 0.5625 for Fig. 6, 0.85 for Fig. 7, 0.475 for Fig. 8). Axis limits are left to matplotlib, as in the notebooks,
except where the notebooks fix them. Labels are typeset with LaTeX.

Usage: python make_figures.py ; reads ../data, writes ../new_reconstruction.pdf (Fig. 4), ../probe_analysis.pdf (Fig. 6),
../scaling.pdf (Fig. 7) and ../full.pdf (Fig. 8).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdhl_numerics as tn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "..", "data"), os.path.join(HERE, "..")

CHAIN_RC = {"text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif", "axes.labelsize": 16,
            "axes.titlesize": 15, "font.size": 12, "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
            "pdf.fonttype": 42}                                      # TrueType, not Type 3, for journal production
COMPLEXITY_RC = dict(CHAIN_RC, **{"axes.labelsize": 14, "axes.titlesize": 14})
LATEX = r"\usepackage{amsmath}\usepackage{bm}"

BLUE, LIGHT = "#2a8bdb", "#a7d0ed"                                   # reconstruction; S = 9 in Fig. 6
SHADE_EARLY, SHADE_LATE, DIVERGENCE_TIME = "#f9fdfe", "#fcf5f6", 1.2  # Fig. 6(c) shading and guide line
N_COLORS = {2: "#d6e7f3", 4: "#a3bfd4", 6: "#7196b5", 8: "#3e6e96"}   # Fig. 7(a) lines and markers
N_BANDS = {2: "#edf4fa", 4: "#d6e2ec", 6: "#bfd0de", 8: "#a8bed0"}    # opaque spread bands
B_COLORS = ["#f8e1cb", "#e7c2a0", "#d6a374", "#c68448", "#b5651d"]     # Fig. 7(b), N_tot = 1e6..1e10
B_BANDS = ["#fcf2e8", "#f4e4d4", "#edd6c0", "#e5c8ad", "#deba99"]

# glyph gradients (axial, top-left corner to bottom-right corner), sampled at u = 0, 0.1, ..., 1
PINK = "ffdbea ffdbea ffdbea ffe6f1 fff6fa ffffff ffffff ffffff ffffff ffffff ffffff"
GRAD_FIG4 = "639adb 76a6df 88b2e4 9bbee8 aecaec c6daf2 dfeaf8 f8fafd ffffff ffffff ffffff"
GRAD_FIG8 = "819bb6 90a7bf 9fb3c7 aebfd0 bdcbd9 d1dae4 e5eaf0 f9fbfc ffffff ffffff ffffff"
GRAD_S36 = "417bc0 5488c6 6795cc 7aa3d3 8db0d9 a0bddf b3cae6 c6d7ec d9e5f2 ecf2f9 ffffff"
GRAD_S9 = "9ebad3 a8c1d7 b1c8dc bbcfe0 c5d6e5 cedde9 d8e3ed e2eaf2 ecf1f6 f5f8fb ffffff"


def gradient(stops, w, h, res=64):
    cols = np.array([to_rgb("#" + c) for c in stops.split()])
    jj, ii = np.meshgrid((np.arange(res) + 0.5) / res, (np.arange(res) + 0.5) / res)
    u = (jj * w + ii * h) / (w + h)
    return np.stack([np.interp(u, np.linspace(0, 1, len(cols)), cols[:, c]) for c in range(3)], axis=-1)


class Page:
    """A figure page of W x H points; matplotlib content at 1/scale of that size."""

    def __init__(self, W, H, scale, rc):
        plt.rcdefaults()
        plt.rcParams.update(rc)
        plt.rcParams["text.latex.preamble"] = LATEX
        self.W, self.H, self.s = W, H, scale
        self.fig = plt.figure(figsize=(W / scale / 72, H / scale / 72))
        self.ov = self.fig.add_axes([0, 0, 1, 1], zorder=10)
        self.ov.set_axis_off()
        self.ov.set_autoscale_on(False)
        self.ov.set_xlim(0, W)
        self.ov.set_ylim(H, 0)

    def axes(self, box, **kw):
        x0, y0, x1, y1 = box
        return self.fig.add_axes([x0 / self.W, 1 - y1 / self.H, (x1 - x0) / self.W, (y1 - y0) / self.H], **kw)

    def text(self, s, box, rotation=0):
        """LaTeX label scaled to the width of the box (its height if rotated) and centred on it."""
        x0, y0, x1, y1 = box
        t = self.ov.text((x0 + x1) / 2, (y0 + y1) / 2, s, usetex=True, fontsize=10, ha="center", va="center",
                         rotation=rotation)
        bb = t.get_window_extent(self.fig.canvas.get_renderer())
        span = bb.height if rotation else bb.width
        t.set_fontsize(10 * ((y1 - y0) if rotation else (x1 - x0)) / self.s * self.fig.dpi / 72 / span)

    def line(self, x0, x1, y, lw, color="k", dash=None):
        kw = {"dashes": (dash / lw, dash / lw)} if dash else {}
        self.ov.plot([x0, x1], [y, y], color=color, lw=lw / self.s, solid_capstyle="butt", dash_capstyle="butt", **kw)

    def glyph(self, x_first, y_top, w, h, step, line, active, stops, shadow):
        """Six-qubit chain; the sites in `active` carry the gradient `stops`, the others the pink one."""
        yc = y_top + h / 2
        self.ov.plot(line, [yc, yc], color="k", lw=0.25 / self.s, zorder=1)
        for i in range(6):
            x0 = x_first + i * step
            on = i in active
            if on and shadow:                                   # soft halo, about a tenth of the glyph width
                for frac, a in ((0.105, 0.10), (0.062, 0.22)):
                    self.ov.add_patch(Ellipse((x0 + w / 2, yc), w, h, fill=False, ec=(0, 0, 0, a), lw=frac * w / self.s,
                                              zorder=2))
            im = self.ov.imshow(gradient(stops if on else PINK, w, h), extent=(x0, x0 + w, y_top + h, y_top),
                                interpolation="bilinear", aspect="auto", zorder=3)
            im.set_clip_path(Ellipse((x0 + w / 2, yc), w, h, transform=self.ov.transData))
            self.ov.add_patch(Ellipse((x0 + w / 2, yc), w, h, fill=False, ec="k", lw=0.25 / self.s, zorder=4))

    def save(self, name):
        self.ov.set_xlim(0, self.W)
        self.ov.set_ylim(self.H, 0)
        self.fig.savefig(os.path.join(OUT, name), dpi=600)
        plt.close(self.fig)


def chain_data(S):
    d = np.load(os.path.join(DATA, f"chain_S{S}.npz"))
    return d["t"], d["theta_X"], d["theta_YY"], d["err"]


def fig4():
    t, X, Y, _ = chain_data(36)
    pg = Page(1100, 700, 1.2, CHAIN_RC)
    boxes = [(93.6, 42.7, 560.9, 331.6), (606.4, 42.7, 1073.7, 331.6), (93.6, 355.0, 560.9, 643.9), (606.4, 355.0, 1073.7, 643.9)]
    axs = [pg.axes(b) for b in boxes]
    for ax in axs[:3]:
        ax.sharex(axs[3])
    Om, J = tn.Chain(6).targets(t)
    for ax, true, rec in zip(axs, [Om[0], Om[3], J[0], J[3]], [X[0], X[3], Y[0], Y[3]]):
        ax.plot(t, true, color="black", linestyle="--", lw=2.5)
        ax.plot(t, rec, color=BLUE, alpha=0.85, lw=2)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=11)
    axs[3].set_ylim(0.2, 0.8)
    for ax in axs[:2]:
        ax.tick_params(labelbottom=False)
    pg.line(332.4, 420.1, 12.0, 4.0)
    pg.text(r"Reconstructed", (434.0, 2.2, 582.9, 20.3))
    pg.line(636.9, 733.8, 12.0, 4.0, dash=8)
    pg.text(r"Target", (744.4, 0.0, 817.0, 23.9))
    pg.text(r"Hamiltonian coefficients", (17.6, 212.9, 38.0, 495.3), rotation=90)
    pg.text(r"Time ($t$)", (541.9, 676.1, 626.0, 700.0))
    for lab, box, x_first, y_top, line, active in [
            (r"$h_1(t)X_1$", (322.4, 264.1, 380.2, 282.2), 325.7, 286.5, (316.6, 487.9), {0}),
            (r"$h_4(t)X_4$", (751.6, 253.8, 809.8, 271.9), 686.7, 273.4, (677.6, 848.9), {3}),
            (r"$J_1(t)Y_1Y_2$", (311.0, 557.0, 382.4, 575.1), 312.7, 580.1, (303.6, 474.9), {0, 1}),
            (r"$J_4(t)Y_4Y_5$", (764.2, 557.0, 835.7, 575.1), 686.7, 578.4, (677.6, 848.9), {3, 4})]:
        pg.text(lab, box)
        pg.glyph(x_first, y_top, 20.9, 24.4, 26.56, line, active, GRAD_FIG4, shadow=True)
    pg.save("new_reconstruction.pdf")


def fig6():
    t9, X9, _, e9 = chain_data(9)
    t36, X36, _, e36 = chain_data(36)
    pg = Page(600, 250, 0.5625, CHAIN_RC)
    axs = [pg.axes((x0, 54.2, x1, 211.2)) for x0, x1 in [(23.0, 190.3), (223.7, 391.0), (424.5, 591.8)]]
    target_qubit = 1
    axs[0].plot(t9, tn.true_Omega_X(target_qubit, t9), color="black", linestyle="--", lw=2.5)
    axs[0].plot(t9, X9[target_qubit], color=LIGHT, alpha=0.9, lw=2.5)
    axs[1].plot(t36, tn.true_Omega_X(target_qubit, t36), color="black", linestyle="--", lw=2.5)
    axs[1].plot(t36, X36[target_qubit], color=BLUE, alpha=0.9, lw=2.5)
    axs[2].plot(t9, e9, color=LIGHT, lw=2.5)
    axs[2].plot(t36, e36, color=BLUE, lw=2.5)
    axs[2].axvline(x=DIVERGENCE_TIME, color="black", linestyle="-.", lw=1.5, alpha=0.8)
    axs[2].axvspan(0, DIVERGENCE_TIME, color=SHADE_EARLY, alpha=0.5)
    axs[2].axvspan(DIVERGENCE_TIME, 3.0, color=SHADE_LATE, alpha=0.5)
    axs[2].set_ylim(0, max(e9.max(), e36.max()) * 1.1)
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=12)
    pg.line(100.7, 132.3, 22.2, 2.0)
    pg.text(r"Reconstructed", (137.7, 18.9, 192.5, 25.5))
    pg.line(222.5, 254.0, 22.2, 2.0, dash=4)
    pg.text(r"Target", (261.2, 17.9, 287.3, 26.5))
    pg.line(321.2, 352.7, 21.9, 2.0, color=LIGHT)
    pg.text(r"$S=9$", (361.6, 18.1, 385.2, 25.7))
    pg.line(409.7, 441.3, 21.9, 2.0, color=BLUE)
    pg.text(r"$S=36$", (448.9, 18.1, 477.4, 25.7))
    for lab, box in [(r"$(a)$", (10.3, 37.8, 22.6, 48.9)), (r"$(b)$", (206.7, 37.5, 218.5, 49.3)), (r"$(c)$", (402.7, 39.2, 414.5, 51.0)),
                     (r"Minimal graph coloring ($S=9$)", (55.2, 39.0, 170.9, 47.7)),
                     (r"Overcomplete Pauli set ($S=36$)", (248.9, 38.7, 376.4, 48.0)),
                     (r"Global reconstruction error $\|\Delta\boldsymbol{\mu}(t)\|_2$", (433.6, 38.7, 579.9, 48.0)),
                     (r"Time ($t$)", (294.3, 232.6, 331.0, 243.1)),
                     (r"$h_2(t)X_2$", (37.2, 58.4, 60.2, 65.5)), (r"$h_2(t)X_2$", (238.4, 58.4, 261.3, 65.5))]:
        pg.text(lab, box)
    pg.glyph(35.0, 66.7, 7.7, 9.1, 9.82, (31.6, 95.0), {1}, GRAD_S9, shadow=False)
    pg.glyph(236.2, 66.7, 7.7, 9.1, 9.82, (232.8, 296.1), {1}, GRAD_S36, shadow=False)
    pg.save("probe_analysis.pdf")


def fig7():
    d = json.load(open(os.path.join(DATA, "complexity.json")))
    N_total, n_array, res = d["N_total"], np.array(d["n"]), d
    means = {n: np.array(res["means"][str(n)]) for n in n_array}
    stds = {n: np.array(res["stds"][str(n)]) for n in n_array}
    pg = Page(724, 300, 0.85, COMPLEXITY_RC)
    ax1, ax2 = pg.axes((81.7, 31.0, 359.4, 251.8)), pg.axes((411.6, 31.0, 689.3, 251.8))
    for n in n_array:
        ax1.loglog(N_total, means[n], "o-", color=N_COLORS[n], lw=2.5, markersize=6,
                   label=f"$n={n}$ ($\\alpha \\approx {res['alphas'][str(n)]:.2f}$)")
        ax1.fill_between(N_total, np.maximum(means[n] - stds[n], 1e-10), means[n] + stds[n], color=N_BANDS[n], edgecolor="none")
    ax1.set_xlabel(r"Total resource budget ($N_\text{tot}$)")
    ax1.set_ylabel(r"Accumulated error $\mathsf{Err}(T_\text{sc})$")
    ax1.grid(True, which="major", ls="-", alpha=0.4, linewidth=0.8)
    ax1.grid(False, which="minor")
    ax1.legend(loc="upper right", framealpha=0.3)
    for i, Nt in enumerate(N_total):
        m = np.array([means[n][i] for n in n_array])
        sd = np.array([stds[n][i] for n in n_array])
        ax2.loglog(n_array, m, "o-", color=B_COLORS[i], lw=2.5, markersize=7,
                   label=f"$N_{{tot}}=10^{{{int(np.log10(Nt))}}}$ ($\\beta \\approx {res['betas'][str(Nt)]:.2f}$)")
        ax2.fill_between(n_array, np.maximum(m - sd, 1e-10), m + sd, color=B_BANDS[i], edgecolor="none")
    ax2.set_xlabel(r"Number of qubits ($n$)")
    ax2.set_xticks(n_array)
    ax2.set_xticklabels([str(n) for n in n_array])
    ax2.xaxis.set_minor_formatter(plt.NullFormatter())
    ax2.grid(True, which="major", ls="-", alpha=0.4, linewidth=0.8)
    ax2.grid(False, which="minor")
    ax2.legend(loc="upper left", framealpha=0.4)
    pg.text(r"$(a)$", (64.5, 7.1, 80.4, 21.5))
    pg.text(r"$(b)$", (389.0, 7.1, 403.4, 21.5))
    pg.save("scaling.pdf")


def fig8():
    t, X, Y, _ = chain_data(36)
    Om, J = tn.Chain(6).targets(t)
    pg = Page(450, 550, 0.475, CHAIN_RC)
    rows = [(20.5, 94.9), (106.0, 180.4), (191.6, 266.0), (277.2, 351.6), (362.7, 437.1), (448.3, 522.7)]
    left = [pg.axes((42.9, y0, 223.7, y1)) for y0, y1 in rows]
    right = [pg.axes((242.2, y0, 423.0, y1)) for y0, y1 in rows[:5]]
    for ax in left[:5] + right:
        ax.sharex(left[5])
        ax.tick_params(labelbottom=False)
    for q, ax in enumerate(left):
        ax.plot(t, Om[q], color="black", linestyle="--", lw=2.5)
        ax.plot(t, X[q], color=BLUE, alpha=0.85, lw=2.5)
    for c, ax in enumerate(right):
        ax.plot(t, J[c], color="black", linestyle="--", lw=2.5)
        ax.plot(t, Y[c], color=BLUE, alpha=0.85, lw=2.5)
        ax.set_ylim(*((0.4, 1.2) if c % 2 == 0 else (0.2, 0.8)))
    for ax in left + right:
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=12)
    pg.line(128.1, 166.2, 10.8, 1.0)
    pg.text(r"Reconstructed", (173.9, 7.1, 234.9, 14.5))
    pg.line(247.4, 285.5, 10.8, 1.0, dash=2)
    pg.text(r"Target", (290.1, 6.3, 316.9, 15.2))
    pg.text(r"Time ($t$)", (209.4, 534.8, 240.6, 543.7))
    h_glyphs = [((76.0, 29.2, 97.5, 35.9), 80.3, 38.8, (77.2, 136.9)), ((80.6, 112.3, 105.9, 120.2), 80.3, 123.9, (77.2, 136.9)),
                ((164.8, 197.9, 190.2, 205.7), 155.3, 207.5, (152.1, 211.9)), ((174.0, 280.7, 199.4, 288.6), 155.3, 291.2, (152.1, 211.9)),
                ((150.9, 364.7, 176.3, 372.6), 124.0, 374.8, (120.9, 180.6)), ((146.3, 449.6, 171.6, 457.5), 124.0, 458.4, (120.9, 180.6))]
    for q, (box, x_first, y_top, line) in enumerate(h_glyphs):
        pg.text(rf"$h_{q + 1}(t)X_{q + 1}$", box)
        pg.glyph(x_first, y_top, 7.3, 8.5, 9.27, line, {q}, GRAD_FIG8, shadow=True)
    j_glyphs = [((247.9, 29.1, 279.1, 37.0), 38.3), ((255.1, 110.6, 286.2, 118.5), 119.4), ((264.7, 197.9, 295.8, 205.7), 207.5),
                ((269.4, 280.7, 300.5, 288.6), 291.1), ((287.9, 364.7, 319.1, 372.6), 374.7)]
    for c, (box, y_top) in enumerate(j_glyphs):
        pg.text(rf"$J_{c + 1}(t)Y_{c + 1}Y_{c + 2}$", box)
        pg.glyph(253.5, y_top, 7.3, 8.5, 9.27, (250.4, 310.1), {c, c + 1}, GRAD_FIG8, shadow=True)
    pg.save("full.pdf")


if __name__ == "__main__":
    for f in (fig4, fig6, fig7, fig8):
        f()
        print(f.__name__, "written")
