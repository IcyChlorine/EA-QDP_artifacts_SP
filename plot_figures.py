"""Recompute every accuracy in Sec. VI from the measurement counts and draw Figs. 2, 4 and 5.

Usage:  python plot_figures.py

Reads data/{hardware,simulation}/{motivating,attack,defense}.json and writes
  figures/exp_motivating_accuracy.pdf   (Fig. 2, bottom)
  figures/exp_attack_accuracy.pdf       (Fig. 4)
  figures/exp_defense_accuracy.pdf      (Fig. 5)
  results_summary.csv                   (every plotted point: accuracy, 95% CI, counts, bound)

Decision rules (character i of a count key half is qubit pair i; see data/*.json):
  XZ_NR, *_BV:          answer every bit, guess bit i from system bit i (the readout of S).
  XZ_EA in motivating:  answer only if the reference bit is 1 (Bell outcomes Phi-, Psi-), guess the system bit.
  XZ_EA in attack:      answer bit i only if system bit i is 1 (Psi+/Psi-), guess the reference bit i.
  DP_EA, DP_GLOBAL_EA:  answer bit i only if system bit i is 0 (Phi+/Phi-), guess the reference bit i.
Accuracy = correct answered bits / answered bits, pooled over all secret strings.

Error bars (95%, sampling error only):
  motivating: exact Clopper-Pearson interval of the pooled binomial accuracy;
  attack/defense: percentile bootstrap that resamples whole shots within each secret string.

Bounds (red/green boundary): e^eps/(1+e^eps), with eps the per-bit QDP parameter (Figs. 2, 4)
or the per-bit EA-QDP parameter (Fig. 5) computed in circuits.py.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import binomtest

import circuits as C

HERE = Path(__file__).resolve().parent
DATA, FIGURES = HERE / "data", HERE / "figures"
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 41

PAPER_RC = {"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False,
            "axes.spines.right": False, "pdf.fonttype": 42, "axes.linewidth": .6,
            "xtick.major.width": .6, "ytick.major.width": .6}
RED, GREEN = "#fbe8e7", "#e8f4e9"
RED_TEXT, GREEN_TEXT = "#943d39", "#326647"
COLORS = {"XZ_NR": "#245f99", "XZ_BV": "#245f99", "XZ_EA": "#a32c69", "DP_BV": "#237a54",
          "DP_EA": "#6a4c93", "DP_GLOBAL_EA": "#b5651d"}
MARKERS = {"XZ_NR": "D", "XZ_BV": "D", "XZ_EA": "s", "DP_BV": "v", "DP_EA": "^", "DP_GLOBAL_EA": "o"}


# --------------------------------------------------------------------------
# Data and accuracy
# --------------------------------------------------------------------------
def load(platform, experiment):
    return json.loads((DATA / platform / f"{experiment}.json").read_text(encoding="utf-8"))["records"]


def decide(experiment, configuration, key):
    """(answered flags, guessed bits) of one shot with count key ``key``."""
    reference, system = key.split("|")
    if configuration.endswith(("_BV", "_NR")):
        return ([True]*len(system), list(system))
    if experiment == "motivating":            # oracle X^x flips the parity bit
        return ([reference[0] == "1"], [system[0]])
    accepted = "1" if configuration == "XZ_EA" else "0"   # oracle Z^x flips the phase bit
    return ([bit == accepted for bit in system], list(reference))


def shot_table(experiment, configuration, run):
    """Histogram of shots by (answered bits, correct bits) for one secret string."""
    table = Counter()
    for key, count in run["counts"].items():
        answered, guesses = decide(experiment, configuration, key)
        correct = sum(a and g == x for a, g, x in zip(answered, guesses, run["secret"]))
        table[sum(answered), correct] += count
    return table


def accuracy(record, seed):
    tables = [shot_table(record["experiment"], record["configuration"], run) for run in record["runs"]]
    answered = sum(a*c for t in tables for (a, _), c in t.items())
    correct = sum(k*c for t in tables for (_, k), c in t.items())
    total_bits = sum(run["shots"] for run in record["runs"]) * record["n"]
    if record["experiment"] == "motivating":
        ci = binomtest(correct, answered).proportion_ci(method="exact")
        ci95 = [ci.low, ci.high]
    else:
        rng = np.random.default_rng(seed)
        draws = np.zeros((BOOTSTRAP_DRAWS, 2), dtype=np.int64)
        for table, run in zip(tables, record["runs"]):
            values = np.array(list(table), dtype=np.int64)
            counts = np.array(list(table.values()), dtype=np.int64)
            draws += rng.multinomial(run["shots"], counts/run["shots"], size=BOOTSTRAP_DRAWS) @ values
        draws = draws[draws[:, 0] != 0]
        ci95 = np.quantile(draws[:, 1]/draws[:, 0], [.025, .975]).tolist()
    return {"accuracy": correct/answered, "ci95": ci95, "correct_bits": correct,
            "answered_bits": answered, "answer_rate": answered/total_bits}


def points(experiment):
    """All plotted points of one experiment, with recomputed accuracy and bounds."""
    out = []
    for p_index, platform in enumerate(("hardware", "simulation")):
        for record in load(platform, experiment):
            c, n = record["configuration"], record["n"]
            seed = [BOOTSTRAP_SEED, list(C.EXPERIMENTS).index(experiment),
                    C.EXPERIMENTS[experiment].index(c), n, p_index]
            eps = C.per_bit_epsilon(experiment, c, n)
            out.append({"experiment": experiment, "platform": platform, "configuration": c, "n": n,
                        **record["parameter"], "per_bit_epsilon": eps,
                        "accuracy_bound": float(C.accuracy_bound(eps)),
                        "framework": "EA-QDP" if experiment == "defense" else "QDP",
                        **accuracy(record, seed)})
    return out


def select(rows, configuration, platform):
    return sorted((r for r in rows if r["configuration"] == configuration and r["platform"] == platform),
                  key=lambda r: r["n"])


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def series(ax, rows, configuration, simulated, **style):
    x = np.array([r["n"] for r in rows])
    y = np.array([r["accuracy"] for r in rows])
    lo, hi = np.array([r["ci95"] for r in rows]).T
    c = COLORS[configuration]
    ax.errorbar(x, y, [y-lo, hi-y], fmt=MARKERS[configuration], ls=":" if simulated else "-", color=c,
                mfc="white" if simulated else c, mew=.9, ms=style.get("ms", 3.2),
                capsize=1.6, elinewidth=.7, lw=.9, zorder=4 if simulated else 5)


def shade(ax, bounds, x_min, x_max, y_min, y_max):
    """Red region above the per-bit accuracy bound, green region below it."""
    ns = sorted(bounds)
    bx = [x_min, *ns, x_max]
    by = [bounds[ns[0]], *[bounds[n] for n in ns], bounds[ns[-1]]]
    ax.fill_between(bx, by, y_max, color=RED, lw=0, zorder=0)
    ax.fill_between(bx, y_min, by, color=GREEN, lw=0, zorder=0)


def figure_motivating(rows):
    """Fig. 2 (bottom): accuracy against E_XZ without reference and with the entanglement attack."""
    bound = select(rows, "XZ_EA", "hardware")[0]["accuracy_bound"]
    fig, ax = plt.subplots(figsize=(3.25, 1.6), layout="constrained")
    y_min, y_max = .5, 1.04
    ax.axhspan(bound, y_max, color=RED, lw=0, zorder=0)
    ax.axhspan(y_min, bound, color=GREEN, lw=0, zorder=0)
    ax.text(.5, (bound+y_max)/2, "Violation of QDP", color=RED_TEXT, ha="center", va="center")
    ax.text(.5, (y_min+bound)/2, "Allowed by QDP", color=GREEN_TEXT, ha="center", va="center")
    groups = ((0, "XZ_NR", COLORS["XZ_NR"], "D"), (1, "XZ_EA", COLORS["XZ_EA"], "s"))  # markers as in Fig. 4
    for x, configuration, color, marker in groups:
        for offset, platform in ((-.055, "simulation"), (.055, "hardware")):
            p = select(rows, configuration, platform)[0]
            y, (lo, hi) = p["accuracy"], p["ci95"]
            hardware = platform == "hardware"
            ax.errorbar([x+offset], [y], [[y-lo], [hi-y]], fmt=marker, color=color,
                        mfc=color if hardware else "white", mew=1, ms=5, capsize=2.5, lw=1, zorder=4, clip_on=False)
            side = 1 if hardware else -1
            ax.annotate(f"{100*y:.1f}%", (x+offset, y), xytext=(7*side, 0), textcoords="offset points",
                        va="center", ha="left" if hardware else "right", color=color)
    ax.set(xlim=(-.6, 1.6), ylim=(y_min, y_max), xticks=[0, 1],
           xticklabels=["No reference", "Entanglement attack"], ylabel="Accuracy")
    ax.set_yticks([.5, 2/3, .8, .9, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _: "66.7%" if abs(v-2/3) < 1e-9 else f"{100*v:.0f}%")
    ax.tick_params(length=2.5, pad=2)

    def pair(filled):
        return tuple(Line2D([], [], marker=m, color=".3", mfc=".3" if filled else "white", ls="none", ms=4)
                     for m in ("D", "s"))
    fig.legend([pair(False), pair(True)], ["Simulation", "Hardware"], handler_map={tuple: HandlerTuple(ndivide=None)},
               loc="outside lower center", ncols=2, frameon=False, fontsize=7)
    return fig


def figure_attack(rows):
    """Fig. 4: E_XZ,q^{(x)n} (a) and D_p^{(x)n} (b) calibrated by QDP, with and without reference."""
    panels = ((r"(a) $\mathcal{E}_{XZ,q}^{\otimes n}$", ("XZ_BV", "XZ_EA")),
              (r"(b) $\mathcal{D}_p^{\otimes n}$", ("DP_BV", "DP_EA")))
    n_max = max(r["n"] for r in rows)
    x_min, x_max, y_min, y_max = .6, n_max+.4, .5, 1.03
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True, layout="constrained")
    for ax, (title, configs) in zip(axes, panels):
        shade(ax, {r["n"]: r["accuracy_bound"] for r in select(rows, configs[-1], "simulation")},
              x_min, x_max, y_min, y_max)
        for config in configs:
            series(ax, select(rows, config, "simulation"), config, True, ms=3 if config == "XZ_EA" else 3.2)
            series(ax, select(rows, config, "hardware"), config, False, ms=3 if config == "XZ_EA" else 3.2)
        ax.set(title=title, xlim=(x_min, x_max), ylim=(y_min, y_max), xticks=range(1, n_max+1),
               xlabel=r"Number of qubits $n$")
        ax.tick_params(length=2.5, pad=2)
        ax.title.set_size(8.5)
    a, b = axes
    a.text(7.0, .755, "Violation of QDP", color=RED_TEXT, ha="center", va="center", fontsize=7.5)
    a.text(7.0, .56, "Allowed by QDP", color=GREEN_TEXT, ha="center", va="center", fontsize=7.5)
    b.text(6.5, .86, "Violation of QDP", color=RED_TEXT, ha="center", va="center", fontsize=7.5)
    b.text(4.0, .55, "Allowed by QDP", color=GREEN_TEXT, ha="center", va="center", fontsize=7.5)
    a.set_ylabel("Accuracy")
    a.set_yticks([.5, .6, .7, .8, .9, 1.0])
    a.yaxis.set_major_formatter(lambda v, _: f"{100*v:.0f}%")
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], ms=3.2, lw=.9, label=label)
               for c, label in (("XZ_BV", r"$\mathcal{E}_{XZ,q}^{\otimes n}$, no reference (BV)"),
                                ("XZ_EA", r"$\mathcal{E}_{XZ,q}^{\otimes n}$, entanglement attack"),
                                ("DP_BV", r"$\mathcal{D}_p^{\otimes n}$, no reference (BV)"),
                                ("DP_EA", r"$\mathcal{D}_p^{\otimes n}$, entanglement attack"))]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=7,
               handlelength=1.6, columnspacing=1.0, handletextpad=.4)
    return fig


def figure_defense(rows):
    """Fig. 5: D_p^{(x)n} (a) and D_p^glob (b) calibrated by EA-QDP, entanglement attack."""
    panels = ((r"(a) $\mathcal{D}_p^{\otimes n}$", "DP_EA"), (r"(b) $\mathcal{D}_p^{\rm glob}$", "DP_GLOBAL_EA"))
    n_max = max(r["n"] for r in rows)
    x_min, x_max, y_min, y_max = .6, n_max+.4, .4, 1.0
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85), sharey=True, layout="constrained")
    for ax, (title, config) in zip(axes, panels):
        shade(ax, {r["n"]: r["accuracy_bound"] for r in select(rows, config, "simulation")},
              x_min, x_max, y_min, y_max)
        ax.axhline(.5, color=".35", ls="--", lw=.7, zorder=1)  # blind-guess baseline
        series(ax, select(rows, config, "simulation"), config, True)
        series(ax, select(rows, config, "hardware"), config, False)
        ax.set(title=title, xlim=(x_min, x_max), ylim=(y_min, y_max), xticks=range(1, n_max+1),
               xlabel=r"Number of qubits $n$")
        ax.tick_params(length=2.5, pad=2)
        ax.title.set_size(8.5)
        ax.text(6.4, .82, "Violation of EA-QDP", color=RED_TEXT, ha="center", va="center", fontsize=7.5)
        ax.text((x_min+x_max)/2, .44, "Allowed by EA-QDP", color=GREEN_TEXT, ha="center", va="center", fontsize=7.5)
    a = axes[0]
    a.set_ylabel("Accuracy")
    a.set_yticks([.4, .5, .6, .7, .8, .9, 1.0])
    a.yaxis.set_major_formatter(lambda v, _: f"{100*v:.0f}%")
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], ms=3.2, lw=.9, label=label)
               for c, label in (("DP_EA", r"$\mathcal{D}_p^{\otimes n}$, entanglement attack"),
                                ("DP_GLOBAL_EA", r"$\mathcal{D}_p^{\rm glob}$, entanglement attack"))]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=7,
               handlelength=1.6, columnspacing=1.0, handletextpad=.4)
    return fig


# --------------------------------------------------------------------------
def main():
    plt.switch_backend("Agg")
    FIGURES.mkdir(exist_ok=True)
    summary = []
    for experiment, draw in (("motivating", figure_motivating), ("attack", figure_attack),
                             ("defense", figure_defense)):
        rows = points(experiment)
        summary += rows
        with plt.rc_context(PAPER_RC):  # savefig inside: the generic 'serif' family is resolved at draw time
            fig = draw(rows)
            fig.savefig(FIGURES / f"exp_{experiment}_accuracy.pdf", bbox_inches="tight", pad_inches=.01)
        plt.close(fig)
        print(f"figures/exp_{experiment}_accuracy.pdf")
    fields = ["experiment", "platform", "configuration", "n", "q", "p", "framework", "per_bit_epsilon",
              "accuracy_bound", "accuracy", "ci95_low", "ci95_high", "correct_bits", "answered_bits", "answer_rate"]
    with (HERE / "results_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in summary:
            writer.writerow({**r, "ci95_low": r["ci95"][0], "ci95_high": r["ci95"][1]})
    print("results_summary.csv")


if __name__ == "__main__":
    main()
