"""The paper's data figures, from the result files only (CPU): preset B layout, 'signal' colour roles, Arial metrics,
direct labels, no legends, no panel letters. Needs the plotting helpers figkit, plotkit and themes: figlib/ beside this
script, else the directory named by PAPER_FIGURES.

    python <this script> [--numbers numbers.tex] [--out figures]
writes case2_readouts.pdf (Figure 2), case2_dose.pdf (Figure 3), case2_cost.pdf (Figure 4) and case2_estimator.pdf
(Figure 5, appendix), each with a .png beside it, and case2_schematic.pdf (not used in the paper). Some labels read the
paper's numbers file (written by scripts/paper_numbers.py).

Colour roles:
  agent with weight access (the hero)      vivid orange #E8590C, plus P (Claude filled, Codex hollow)
  information-matched script (strongest baseline)  deep blue #0F4D92 hexagon h; fully scripted light blue #4A90D9 X
  input-side (black-box) agent edits       teal #2A9D8F, triangle ^ (Claude filled, Codex hollow)
  other corrections (fine-tuning, steering) slate, v / s / D
  invented effects, the replayed estimators crimson #C2185B
  references (truth, zero), the unedited model black, open circle for the unedited model
"""
import dataclasses
import json
import os
import re
import sys
from pathlib import Path



def _toolkit():
    """the directory holding figkit.py: figlib/ beside this script, else $PAPER_FIGURES, else the first one under
    case1_reasoning/ next to this case"""
    here = Path(__file__).resolve().parent / "figlib"
    if (here / "figkit.py").is_file():
        return str(here)
    if os.environ.get("PAPER_FIGURES"):
        return os.environ["PAPER_FIGURES"]
    case1 = Path(__file__).resolve().parents[2] / "case1_reasoning"
    hits = sorted(case1.rglob("figkit.py")) if case1.is_dir() else []
    return str(hits[0].parent) if hits else None


if _toolkit():
    sys.path.insert(0, _toolkit())
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import figkit as fk  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotkit as pk  # noqa: E402
from themes import THEMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TW, CW = 486, 234          # the two-column template's \textwidth and \columnwidth in bp
B = THEMES["B"]
INK, MUTED, RAMP = B.ink, B.muted, B.ramp
EDIT = "#E8590C"           # agent with weight access
STRONG = "#0F4D92"         # information-matched script, Claude's estimate
STRONG2 = "#4A90D9"        # fully scripted pipeline
INPUT_SIDE = "#2A9D8F"     # input-side agent edits (teal; amber belongs to case 1)
ALERT = "#C2185B"          # invented effects
OTHER = RAMP[4]            # fine-tuning, steering
REF = "#000000"
# the figures below are laid out at the widths TW, CW and printed at ACL's \textwidth (455 bp) and
# \columnwidth (219 bp), a factor of 455/486; every font is set larger by its inverse so that it prints at its nominal size
ACL_SCALE = 486 / 455
SIZE_KEYS = ("font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")
LAB = 8 * ACL_SCALE
THEME = dataclasses.replace(B, accent=EDIT, accent_tint="#FCE4D6", risk=ALERT, risk_tint="#F9DDE8", ok=INK)
plt.rcParams.update(pk.mpl_style("B"))
plt.rcParams.update({k: plt.rcParams[k] * ACL_SCALE for k in SIZE_KEYS})
plt.rcParams["figure.constrained_layout.use"] = False


def canvas(w, h):
    return plt.figure(figsize=(w / 72, h / 72), layout="none")


def place(fig, x0, x1, y0, y1):
    W, H = fig.get_size_inches() * 72
    return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])


LETTERS = False             # lowercase bold panel letters (--letters, the standalone paper's style)


def letters(fig, axes):
    """9 pt bold a, b, c at each panel's top left: the left edge on the panel's leftmost label, 7 pt above the plot area"""
    if not LETTERS or len(axes) < 2:
        return
    fig.canvas.draw()
    r, k = fig.canvas.get_renderer(), 72 / fig.dpi
    W, H = fig.get_size_inches() * 72
    for ch, a in zip("abc", axes):
        bb = a.get_window_extent(r)
        fig.text(a.get_tightbbox(r).x0 * k / W, (bb.y1 * k + 7) / H, ch, fontsize=9, fontweight="bold", va="top")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


readout = [json.loads(x) for x in open(R / "readout.jsonl")]
agents = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_eval.jsonl")}
gen = json.loads((R / "generalize.json").read_text())
NUM = {}


def load_numbers(path: Path) -> None:
    NUM.update(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", path.read_text()))


def num(k):
    """a macro's value; the merged paper's numbers carry the prefix wm, the standalone paper's do not"""
    v = NUM.get(k, NUM.get(k[2:]) if k.startswith("wm") else None)
    return float(v.replace("$-$", "-").replace("$+$", ""))


def ctl(key, method, seed=0):
    return next(r for r in readout if r["key"] == key and r["method"] == method and r["seed"] == seed)


# one row per correction, top to bottom: label, colour, marker, filled, size, main-world source, no-effect source,
# generalize key, replayed-estimator macro
ROWS = [
    ("unedited model", REF, "o", False, 22, ("ctl", "es_register"), ("ctl", "es_register"), "e03:plain model", None),
    ("steering, mean difference", OTHER, "v", True, 22, ("ctl", "steer_a0.25"), ("ctl", "steer_a0.1"),
     "e03:steering, mean difference", None),
    ("steering, probe direction", OTHER, "s", True, 18, ("ctl", "steerp_a32"), ("ctl", "steerp_a8"),
     "e03:steering, probe direction", None),
    ("fine-tuning", OTHER, "D", True, 18, ("ctl", "ft_pnp"), ("ctl", "ft_pnp"), "e03:fine-tuning", None),
    ("fine-tuning, sized to estimate", OTHER, "D", False, 18, ("ctl", "ftC"), None,
     "e03:fine-tuning sized to Claude's estimate", None),
    ("Codex, input side", INPUT_SIDE, "^", False, 24, ("agent", "openai_c"), None, "e03:Codex, black box", None),
    ("Claude, input side", INPUT_SIDE, "^", True, 24, ("agent", "claude_c"), None, "e03:Claude, black box", None),
    ("Codex, weight access", EDIT, "P", False, 40, ("agent", "openai_a"), ("agent", "openai_b"), "e03:Codex, white box",
     "openai_a_n03"),
    ("Claude, weight access", EDIT, "P", True, 52, ("agent", "claude_a"), ("agent", "claude_b"), "e03:Claude, white box",
     "claude_a_n03"),
    ("script, Claude's estimate", STRONG, "h", True, 34, ("ctl", "scriptC"), None,
     "e03:scripted install, Claude's estimate", None),
    ("full script", STRONG2, "X", True, 30, ("ctl", "scriptS"), ("ctl", "scriptS"),
     "e03:scripted install, scripted estimate", "script_n03"),
]


#: the repeated weight-access runs (same prompt, budget and tools), drawn as small marks under run 1's row
REPEATS = {"Codex, weight access": ("openai_d", "openai_e"), "Claude, weight access": ("claude_d", "claude_e")}


def value(src, world):
    kind, name = src
    if kind == "ctl":
        r = ctl(world, name)["insulin_x2"]
    else:
        r = agents[name]["insulin_x2"]
    return r["ratio"] if world == "e03" else r["mean_shift"]


def fresh_worlds() -> dict:
    """the implied insulin x2 shift on each fresh no-effect world (results/fresh_null_seeds*.json), by estimator"""
    fs = json.loads((R / "fresh_null_seeds.json").read_text())["seeds"]
    fa = json.loads((R / "fresh_null_seeds_agents.json").read_text())["seeds"]
    return {"script_n03": [v["implied_insulin_x2"] for v in fs.values()],
            "claude_a_n03": [v["claude_a"]["implied_insulin_x2"] for v in fa.values()],
            "openai_a_n03": [v["openai_a"]["implied_insulin_x2"] for v in fa.values()]}


def pooled_bias(key: str):
    """(each world's implied shift, the mean over all 21 no-effect worlds, its 95% interval with worlds resampled), from
    results/null_seed_batch.json (scripts/null_seed_batch.py; the six earlier, five graded-seed and ten new)"""
    est = {"claude_a_n03": "claude_a", "openai_a_n03": "openai_a", "script_n03": "script"}[key]
    p = json.loads((R / "null_seed_batch.json").read_text())["estimators"][est]
    return p["six"] + p["five"] + p["ten"], p["mean_21"], p["interval_21"]


def estimator(key: str):
    """an estimator's implied insulin x2 shift on the no-effect records, with the 2.5th and 97.5th percentiles over
    resampled training stays (results/replay_bootstrap/<key>.jsonl; draw -1 is the original records)"""
    rows = [json.loads(x) for x in open(R / "replay_bootstrap" / f"{key}.jsonl")]
    point = next(r["implied_insulin_x2"] for r in rows if r["draw"] == -1)
    v = np.array([r["implied_insulin_x2"] for r in rows if r["draw"] >= 0 and r["implied_insulin_x2"] is not None])
    return point, np.percentile(v, 2.5), np.percentile(v, 97.5)


def tint(col, k=0.42):
    """the colour mixed with white, k of the colour: the light shade of a row (hours 13-16 in the middle panel)"""
    import matplotlib.colors as mc
    r, g, b_ = mc.to_rgb(col)
    return (1 - k + k * r, 1 - k + k * g, 1 - k + k * b_)


def mark(ax, x, y, col, mk, filled, s, z=4):
    face = {"color": col, "linewidths": 0} if filled else {"facecolor": "white", "edgecolor": col, "linewidths": 0.9}
    ax.scatter([x], [y], s=s, marker=mk, zorder=z, clip_on=False, **face)


#: the replayed estimators have their own table (tab:inertsets); the right panel shows each edit's own shift only
SHOW_ESTIMATORS = False


def fig_readouts():
    """Main figure, one claim: weight access and the information-matched script recover the queried effect; the
    effect's time course and the no-effect world show what that readout hides. Left (hero): insulin x2 readout over
    truth, held-out stays, 95% intervals. Middle: the same readout in hours 13-16 and 21-24. Right: the shift in the
    no-effect world: each correction's own shift, and below it the bias of the estimator behind it pooled over six
    no-effect worlds (light 95% band, black line at the mean), above it each world's value (dots). A row's glyph and fill
    name the correction in every panel; hours are told apart by shade, the pooled estimator by its band and mean line,
    which no row uses."""
    n = len(ROWS)
    fig = canvas(TW, 170)
    a = place(fig, 132, 260, 30, 156)
    b = place(fig, 278, 364, 30, 156)
    c = place(fig, 382, 476, 30, 156)
    ys = np.arange(n)[::-1]
    lims, not_run = [], []
    fresh = fresh_worlds()
    for (lab, col, mk, fill, s, main, null, gk, rep), y in zip(ROWS, ys):
        x, lo, hi = value(main, "e03")
        a.plot([lo, hi], [y, y], color=col, lw=1.1, marker="", solid_capstyle="butt", zorder=2)
        mark(a, x, y, col, mk, fill, s)
        reps = [agents[r]["insulin_x2"]["ratio"][0] for r in REPEATS.get(lab, ()) if r in agents]
        if reps:                                   # runs 2 and 3: small marks just below the row, on a dotted line
            span = [min(reps + [x]), max(reps + [x])]          # spanning all three runs' points, tied to run 1's
            a.plot(span, [y - 0.32] * 2, color=col, lw=0.6, ls=(0, (1, 1.2)), marker="", zorder=2)
            a.plot([x, x], [y - 0.12, y - 0.32], color=col, lw=0.6, marker="", zorder=2)
            for v in reps:
                mark(a, v, y - 0.32, col, mk, fill, 0.32 * s, z=3)
        h = gen[gk]["detail"]["insulin_x2"]["horizon"]
        e, l = h["hours 13-16"][0], h["hours 21-24"][0]
        b.plot([e, l], [y, y], color=col, lw=0.8, marker="", zorder=2)   # the row's own glyph: light early, dark late
        mark(b, e, y, tint(col), mk, fill, 0.6 * s, z=3)
        mark(b, l, y, col, mk, fill, 0.6 * s, z=3)
        if null is None:
            not_run.append(y)
            continue
        x, lo, hi = value(null, "n03")
        lims.extend([lo, hi])                     # the axis spans every interval drawn: nothing is cut at its edge
        c.plot([lo, hi], [y, y], color=col, lw=1.1, marker="", solid_capstyle="butt", zorder=2)
        mark(c, x, y, col, mk, fill, s)
        if rep and SHOW_ESTIMATORS:               # the estimator behind the procedure, below the edit's own readout
            pts, m, (plo, phi) = pooled_bias(rep)
            # on the row itself (the edit's own marker on top), each world's dot just above it, so nothing sits
            # between two rows
            c.plot([plo, phi], [y, y], color=tint(col, 0.5), lw=4.2, marker="", solid_capstyle="butt", zorder=1)
            c.plot([m, m], [y - 0.2, y + 0.2], color=REF, lw=1.1, marker="", zorder=3)   # the mean, as a median line
            lims.extend([plo, phi])
            c.scatter(pts, [y + 0.3] * len(pts), s=9, marker="o", color=col, linewidths=0, zorder=5)
            lims.extend(pts)
    top = n - 0.35
    ylo, yhi = -2.55, n - 0.4                 # an empty strip below the rows holds the keys, off the shading
    f0 = (-0.6 - ylo) / (yhi - ylo)           # axes fraction where the rows begin
    a.axvspan(0.9, 1.1, ymin=f0, color=RAMP[0], lw=0, zorder=0)        # the band: a tenth of the truth
    for ax in (a, b):
        ax.axvline(1.0, ymin=f0, color=REF, lw=0.8, zorder=1)
        ax.text(1.0, top, "truth", fontsize=LAB, ha="center", va="bottom")
    a.set(xlim=(-0.02, 1.14), ylim=(ylo, yhi), xticks=[0, 0.5, 1.0], yticks=ys,
          xlabel="Insulin ×2, readout / truth")
    a.set_yticklabels([r[0] for r in ROWS])
    for t, r in zip(a.get_yticklabels(), ROWS):
        if r[1] == EDIT:
            t.set_fontweight("bold")
    a.text(-0.01, -1.15, "bars: 95% over held-out stays", fontsize=LAB, color=MUTED, ha="left", va="center")
    mark(a, 0.02, -2.15, MUTED, "P", True, 0.32 * 52)
    a.text(0.06, -2.15, "runs 2, 3 (dotted: range)", fontsize=LAB, color=MUTED, ha="left", va="center")
    b.set(xlim=(0, 1.35), ylim=(ylo, yhi), xticks=[0, 0.5, 1.0], yticks=[], xlabel="Readout / truth by hour")
    b.plot([0.02, 0.10], [-1.15] * 2, color=tint(MUTED), lw=3.2, marker="", solid_capstyle="butt")
    b.text(0.13, -1.15, "h 13–16", fontsize=LAB, color=MUTED, ha="left", va="center")
    b.plot([0.71, 0.79], [-1.15] * 2, color=MUTED, lw=3.2, marker="", solid_capstyle="butt")
    b.text(0.82, -1.15, "h 21–24", fontsize=LAB, color=MUTED, ha="left", va="center")
    band = num("wmBand")                       # a tenth of the true insulin x2 change
    x0, x1 = np.floor(min(lims + [-3.0]) - 0.3), np.ceil(max(lims + [3.0]) + 0.3)
    c.axvspan(-band, band, ymin=f0, color=RAMP[0], lw=0, zorder=0)   # the band only, as in the left panel
    c.axvline(0, ymin=f0, color=REF, lw=0.8, zorder=1)
    c.text(0, top, "truth", fontsize=LAB, ha="center", va="bottom")
    xs, W = x0 + 0.03 * (x1 - x0), x1 - x0
    if SHOW_ESTIMATORS:
        c.plot([xs, xs + 0.16 * W], [-1.15] * 2, color=tint(MUTED, 0.5), lw=4.2, marker="", solid_capstyle="butt")
        c.plot([xs + 0.08 * W] * 2, [-1.15 - 0.2, -1.15 + 0.2], color=REF, lw=1.1, marker="", zorder=5)
        c.text(xs + 0.2 * W, -1.15, "estimator, pooled", fontsize=LAB, color=MUTED, ha="left", va="center")
        c.scatter([xs + 0.08 * W], [-2.15], s=11, marker="o", color=MUTED, linewidths=0, zorder=5)
        c.text(xs + 0.2 * W, -2.15, "each world", fontsize=LAB, color=MUTED, ha="left", va="center")
    else:
        c.text(xs, -1.15, "each edit's own", fontsize=LAB, color=MUTED, ha="left", va="center")
        c.text(xs, -2.15, "no-effect shift", fontsize=LAB, color=MUTED, ha="left", va="center")
    for y in not_run:                                           # off the truth line, inside the right-hand margin
        c.text(x1 - 0.15, y, "not run", fontsize=LAB, color=MUTED, ha="right", va="center")
    c.set(xlim=(x0, x1), ylim=(ylo, yhi), xticks=[t for t in range(int(x0), int(x1) + 1) if t % 2 == 0], yticks=[],
          xlabel="No-effect shift (mg/dL)")
    for ax in (a, b, c):
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    letters(fig, [a, b, c])
    save(fig, "case2_readouts")


#: the main text's first result figure, the left panel of case2_readouts at column width
SHORT = {"steering, mean difference": "steering, mean diff.", "steering, probe direction": "steering, probe",
         "fine-tuning, sized to estimate": "fine-tuning, sized", "script, Claude's estimate": "script, Claude's est."}


def fig_readouts_col():
    """Main text, result 1: the insulin x2 readout over the truth on held-out stays for every seed-0 correction, with
    95% intervals and the band; the agents' runs 2 and 3 as small marks under run 1."""
    n = len(ROWS)
    H = 196
    fig = canvas(CW, H)
    a = place(fig, 104, 226, 30, H - 16)
    ys = np.arange(n)[::-1]
    for (lab, col, mk, fill, s, main, null, gk, rep), y in zip(ROWS, ys):
        x, lo, hi = value(main, "e03")
        a.plot([lo, hi], [y, y], color=col, lw=1.1, marker="", solid_capstyle="butt", zorder=2)
        mark(a, x, y, col, mk, fill, s)
        reps = [agents[r]["insulin_x2"]["ratio"][0] for r in REPEATS.get(lab, ()) if r in agents]
        if reps:
            span = [min(reps + [x]), max(reps + [x])]
            a.plot(span, [y - 0.32] * 2, color=col, lw=0.6, ls=(0, (1, 1.2)), marker="", zorder=2)
            a.plot([x, x], [y - 0.12, y - 0.32], color=col, lw=0.6, marker="", zorder=2)
            for v in reps:
                mark(a, v, y - 0.32, col, mk, fill, 0.32 * s, z=3)
    ylo, yhi = -2.55, n - 0.4
    f0 = (-0.6 - ylo) / (yhi - ylo)
    a.axvspan(0.9, 1.1, ymin=f0, color=RAMP[0], lw=0, zorder=0)
    a.axvline(1.0, ymin=f0, color=REF, lw=0.8, zorder=1)
    a.text(1.0, n - 0.35, "truth", fontsize=LAB, ha="center", va="bottom")
    a.set(xlim=(-0.02, 1.14), ylim=(ylo, yhi), xticks=[0, 0.5, 1.0], yticks=ys, xlabel="Insulin ×2, readout / truth")
    a.set_yticklabels([SHORT.get(r[0], r[0]) for r in ROWS])
    for t, r in zip(a.get_yticklabels(), ROWS):
        if r[1] == EDIT:
            t.set_fontweight("bold")
    a.text(-0.01, -1.15, "bars: 95% over held-out stays", fontsize=LAB, color=MUTED, ha="left", va="center")
    mark(a, 0.02, -2.15, MUTED, "P", True, 0.32 * 52)
    a.text(0.06, -2.15, "runs 2, 3 (dotted: range)", fontsize=LAB, color=MUTED, ha="left", va="center")
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    save(fig, "case2_readouts_col")


def fig_schematic():
    """Block for the merged paper's two-column intro schematic: a patient world model whose drug effect the dosing
    policy cancels; an agent edits its weights from the records and a forward-pass tool, without the truth; the edit
    is judged on held-out stays with known truth and in a world where the drugs are inert."""
    T = fk.use_theme(THEME, width_pt=CW, font="arial")
    plain = num("wmplainIx")
    H = 640
    d = fk.Figure("case2_schematic", H, "Case 2: drug effects in a patient world model", out=OUT)
    d.text(0, 0, 1000, "patient world model", T.zone, True)
    d.box(0, 52, 1000, 150, T.panel, None, T.r_outer)
    fk.tile(d, "patient", 26, 74, 104)
    d.text(160, 72, 820, "simulated ICU records; the dosing\npolicy cancels the drug's effect", T.small)
    d.text(160, 150, 820, f"insulin ×2: {plain:.2f} of the true change", T.small, True, color=T.risk)
    d.line([(500, 210), (500, 252)], INK, arrow=True)
    d.box(0, 260, 1000, 132, T.accent_tint, T.accent, T.r_outer, 1.6 * T.lw_box)
    d.text(26, 276, 950, "agent edits the weights", T.body, True)
    d.text(26, 322, 950, "Claude or Codex; records + forward-pass tool, no truth", T.small, color=MUTED)
    d.line([(250, 400), (250, 442)], INK, arrow=True)
    d.line([(750, 400), (750, 442)], INK, arrow=True)
    for x, head, sub, tint in ((0, "known truth", "held-out stays,\nunqueried doses", T.panel),
                               (520, "no-effect world", "same doctors,\ninert drugs", T.risk_tint)):
        d.box(x, 450, 480, 182, tint, None, T.r_outer)
        d.text(x + 24, 466, 440, head, T.body, True)
        d.text(x + 24, 514, 440, sub, T.small, color=MUTED)
    d.save()
    fk.check_errors()
    fk.render(OUT / "case2_schematic.pdf")


GEN_ROWS = [("unedited model", "e03:plain model", REF, "o", False), ("fine-tuning", "e03:fine-tuning", OTHER, "D", True),
            ("Codex", "e03:Codex, white box", EDIT, "P", False), ("Claude", "e03:Claude, white box", EDIT, "P", True),
            ("full script", "e03:scripted install, scripted estimate", STRONG2, "X", True)]


def fig_dose():
    """Appendix: predicted and true change against the dose factor from hour 12; only x1/2 and x2 were queried."""
    fig = canvas(TW, 168)
    axes = [place(fig, 42, 176, 30, 158), place(fig, 280, 408, 30, 158)]
    for ax, drug, unit in zip(axes, ("insulin", "ne"), ("Glucose change (mg/dL)", "MAP change (mmHg)")):
        name = {0.0: f"{drug}_x0@12", 0.5: f"{drug}_half", 1.5: f"{drug}_x1.5@12", 2.0: f"{drug}_x2", 3.0: f"{drug}_x3@12"}
        for q in (0.5, 2.0):
            ax.axvspan(q - 0.06, q + 0.06, color=RAMP[1], lw=0, zorder=0)
        truth = None
        ends, style = {}, {}
        for lab, key, col, mk, fill in GEN_ROWS:
            r = gen[key]
            xs, ys, tr = [], [], []
            for f, e in sorted(name.items()):
                src = r.get("original", {}).get(e) if "@" not in e else r.get("doses", {}).get(e)
                xs.append(f); ys.append(src["model_shift"][0]); tr.append(src["true_shift"][0])
            truth = (xs, tr)
            ax.plot(xs, ys, color=col, lw=1.0, marker=mk, ms=4 if mk != "P" else 5.5, mfc=col if fill else "white",
                    mec=col, mew=0.9, zorder=3)
            ends[lab], style[lab] = ys[-1], (col, mk, fill)
        ax.plot(*truth, color=REF, lw=1.1, ls=(0, (4, 2.5)), marker="", zorder=4)
        ends["truth"], style["truth"] = truth[1][-1], (REF, None, True)
        ax.axhline(0, color=RAMP[2], lw=0.6, zorder=0)
        ax.set(xticks=[0, 0.5, 1.5, 2, 3], xticklabels=["×0", "×½", "×1.5", "×2", "×3"], xlim=(-0.1, 3.0),
               xlabel=("Insulin" if drug == "insulin" else "Norepinephrine") + " dose factor from hour 12", ylabel=unit)
        # direct labels at the right end, spread so they do not overlap, each with its series' colour and marker and a
        # leader to its line's end, so that stacked labels stay attached to their lines
        order = sorted(ends.items(), key=lambda kv: kv[1])
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 10.5 / 96               # about 10.5 pt between label centres on the 96-pt axis
        placed = []
        for lab, v in order:
            yv = v if not placed or v - placed[-1] >= gap else placed[-1] + gap
            placed.append(yv)
            col, mk, fill = style[lab]
            ax.plot([3.0, 3.1], [v, yv], color=col, lw=0.6, marker="", clip_on=False, zorder=2)
            if mk is None:
                ax.plot([3.12, 3.28], [yv, yv], color=col, lw=1.1, ls=(0, (2.5, 1.5)), marker="", clip_on=False)
            else:
                ax.plot([3.2], [yv], marker=mk, ls="", ms=4 if mk != "P" else 5, color=col, mfc=col if fill else "white",
                        mec=col, mew=0.9, clip_on=False)
            ax.text(3.34, yv, lab, fontsize=LAB, va="center", ha="left", color=col if col != REF else INK,
                    fontweight="bold" if lab == "Claude" else "normal", clip_on=False)
    axes[0].text(0.5, axes[0].get_ylim()[0], "asked", fontsize=LAB, color=MUTED, ha="center", va="bottom")
    axes[1].set_yticks([-5, 0, 5, 10])        # whole-number ticks keep the axis label clear of the left panel's labels
    letters(fig, axes)
    save(fig, "case2_dose")


HEADS = [("value_bin", "reading value"), ("code", "next event type"), ("action_code", "next drug"),
         ("time", "time to next bundle"), ("count", "readings per bundle"), ("action_count", "drugs per bundle"),
         ("cont", "record continues"), ("kl", "latent (KL)"), ("rate_bin", "infusion rate")]


def fig_cost():
    """Change in held-out loss on every output head: the agents' main-world edits beside the full script and fine-tuning
    (results/model_parts.jsonl, the same teacher-forced held-out stays for all)."""
    parts = {json.loads(x)["label"]: json.loads(x)["parts"] for x in open(R / "model_parts.jsonl")
             if json.loads(x)["world"] == "e03"}
    # four series (the input-side edits are in Table 6), so the markers do not overlap near 0
    runs = [("openai_a", "Codex, weight access", EDIT, "P", False), ("claude_a", "Claude, weight access", EDIT, "P", True),
            ("fine-tuning", "fine-tuning", OTHER, "D", True), ("scriptS", "full script", STRONG2, "X", True)]
    H = 178
    fig = canvas(CW, H)
    ax = place(fig, 92, 226, 28, 148)
    ys = np.arange(len(HEADS))[::-1]
    offs = np.linspace(-0.3, 0.3, len(runs))
    for (run, lab, col, mk, fill), dy in zip(runs, offs):
        for (h, _), y in zip(HEADS, ys):
            p = parts[run][h]
            pct = 100 * (p["edited"] - p["plain"]) / abs(p["plain"])
            ax.plot([0, pct], [y + dy] * 2, color=col, lw=0.7, marker="", zorder=2)
            mark(ax, pct, y + dy, col, mk, fill, 22 if mk == "P" else 14)
    ax.axvline(0, color=REF, lw=0.7, marker="")
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set(xlim=(-1.2, 40), xticks=[0, 1, 10], xticklabels=["0", "1%", "10%"], yticks=ys,
           xlabel="Held-out loss change (symlog)", ylim=(-0.6, len(HEADS) - 0.4))
    ax.set_yticklabels([h[1] for h in HEADS])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for i, (run, lab, col, mk, fill) in enumerate(runs):            # key above the plot, as direct labels
        x0, y0 = 6 + (i % 2) * 112, H - 9 - (i // 2) * 12
        fx = fig.add_axes([x0 / CW, (y0 - 5) / H, 10 / CW, 10 / H])
        fx.set_axis_off()
        mark(fx, 0.5, 0.5, col, mk, fill, 22 if mk == "P" else 14)
        fx.set(xlim=(0, 1), ylim=(0, 1))
        fig.text((x0 + 13) / CW, y0 / H, lab, fontsize=LAB, va="center")
    save(fig, "case2_cost")


def coef(world, run_no, col):
    """the replay's per-lag table (Claude's run printed it): coefficient and SE of column `col` (0 = finger-stick or
    arterial) at lags 1..12, before the feedback of later doses is removed"""
    out = [json.loads(x) for x in open(ROOT / "agent_runs" / "replay" / f"claude_a_{world}" / "outputs.jsonl")]
    text = next(o["output"] for o in out if o["run"] == run_no)
    est, se = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and 1 <= int(parts[0]) <= 12 and "±" in parts[1 + col]:
            v, e = parts[1 + col].split("±")
            est.append(float(v)); se.append(float(e))
    return np.array(est[:12]), np.array(se[:12])


def fig_estimator():
    """Appendix: Claude's estimator replayed unchanged on each world's records: coefficient by lag, +-2 SE, before the
    feedback of later doses is removed."""
    fig = canvas(TW, 140)
    axes = [place(fig, 44, 182, 30, 128), place(fig, 294, 420, 30, 128)]
    lags = np.arange(1, 13)
    for ax, (run_no, unit) in zip(axes, ((15, "Glucose, mg/dL per U/h"), (17, "MAP, mmHg per µg/kg/min"))):
        ax.axhline(0, color=RAMP[2], lw=0.6, zorder=0)
        for world, col, fill, lab in (("e03", REF, True, "main world"), ("n03", ALERT, False, "no-effect world")):
            e, se = coef(world, run_no, 0)
            ax.fill_between(lags, e - 2 * se, e + 2 * se, color=col, alpha=0.12, lw=0)
            ax.plot(lags, e, color=col, lw=1.0, marker="o", ms=3.4, mfc=col if fill else "white", mec=col, mew=0.9)
            ax.text(12.6, e[-1], lab, fontsize=LAB, va="center", color=ALERT if col == ALERT else INK, clip_on=False)
        ax.set(xlabel="Hours after the dose", ylabel=unit, xlim=(0.5, 12.5), xticks=[1, 4, 8, 12])
    letters(fig, axes)
    save(fig, "case2_estimator")


def fig_graded():
    """The paper's method figure: each main-world estimator replayed in worlds of graded effect strength.
    Left: its pooled error (implied minus true insulin x2 shift on the same stays, mean over six paired worlds, 95% interval
    over resampled stays where the copies exist) against the effect strength; shading: a tenth of the true shift at that
    strength (at 0, the main world's). Right: the share of stay-hours with insulin running whose latest finger-stick is at
    least 250 mg/dL, the top tier of Claude's first estimator. results/graded_summary.json, results/tier_occupancy.json."""
    with plt.rc_context({k: plt.rcParams[k] / ACL_SCALE for k in SIZE_KEYS}):
        _fig_graded()


def _fig_graded():
    LAB = 8                                     # drawn at ACL's width: nominal sizes
    g = json.loads((R / "graded_summary.json").read_text())
    W = 455                                     # ACL's \textwidth (bp), drawn at print size
    fig = canvas(W, 168)
    a = place(fig, 42, 286, 66, 160)
    b = place(fig, 350, 448, 66, 160)
    xs = [0.0, 0.25, 0.5, 1.0]
    mt = g["mean_true_shift"]["tuning"]
    band = [0.1 * mt["1.0"], 0.1 * mt["0.25"], 0.1 * mt["0.5"], 0.1 * mt["1.0"]]
    a.fill_between(xs, [-v for v in band], band, color=RAMP[0], lw=0, zorder=0)
    a.axhline(0, color=REF, lw=0.8, zorder=1)
    cx = "#7F2704"                              # Codex in dark brown
    style = {"claude_a": (EDIT, "P", True, "-", 1.6), "claude_d": (EDIT, "P", True, (0, (4, 2)), 0.9),
             "claude_e": (EDIT, "P", True, (0, (1, 1.5)), 0.9), "openai_a": (cx, "P", False, "-", 0.9),
             "openai_d": (cx, "P", False, (0, (4, 2)), 0.9), "openai_e": (cx, "P", False, (0, (1, 1.5)), 0.9),
             "script": (STRONG2, "X", True, "-", 1.2)}
    off = {k: (i - 3) * 0.012 for i, k in enumerate(style)}
    lo_all, hi_all = [], []
    for est, (col, mk, fill, ls, lw) in style.items():
        lv = g["levels"].get(est, {})
        pts = [(x + off[est], lv[str(x)]) for x in xs if lv.get(str(x), {}).get("complete")]
        if not pts:
            continue
        a.plot([p[0] for p in pts], [p[1]["mean"] for p in pts], color=col, lw=lw, ls=ls, marker="", zorder=2)
        for x, v in pts:
            if "interval" in v:
                a.plot([x, x], v["interval"], color=col, lw=0.7, marker="", zorder=2)
                lo_all.append(v["interval"][0]); hi_all.append(v["interval"][1])
            mark(a, x, v["mean"], col, mk, fill, 26 if est == "claude_a" else 14, z=3)
            lo_all.append(v["mean"]); hi_all.append(v["mean"])
    y0, y1 = min(lo_all + [-2.0]) - 0.3, max(hi_all + [2.0]) + 0.3
    a.set(xlim=(-0.08, 1.08), ylim=(y0, y1), xticks=xs, xticklabels=["0", "0.25", "0.5", "1"],
          xlabel="Effect strength (× the main world's)", ylabel="Estimator error (mg/dL)")
    fig.text(42 / W, 14 / 168, "Claude: orange, filled; Codex: brown, hollow; runs 1, 2, 3: solid, dashed, dotted",
             fontsize=LAB, color=MUTED, ha="left", va="bottom")
    fig.text(42 / W, 3 / 168, "blue: full script; shading: the band", fontsize=LAB, color=MUTED, ha="left", va="bottom")
    occ = [g["tier_top_doses"][str(x)] for x in xs]
    b.plot(xs, [100 * v for v in occ], color=MUTED, lw=1.1, marker="o", ms=3.5, zorder=2)
    b.set(xlim=(-0.08, 1.08), ylim=(0, 16), xticks=xs, xticklabels=["0", "0.25", "0.5", "1"],
          xlabel="Effect strength", ylabel="Dosed hours ≥250 (%)")
    letters(fig, [a, b])
    save(fig, "case2_graded")


def fig_graded_col():
    """The graded replay at column width for the main text (the error panel of fig_graded)."""
    with plt.rc_context({k: plt.rcParams[k] / ACL_SCALE for k in SIZE_KEYS}):
        _fig_graded_col()


def _fig_graded_col():
    LAB = 8
    g = json.loads((R / "graded_summary.json").read_text())
    W, H = 219, 168                             # ACL's \columnwidth (bp), drawn at print size
    fig = canvas(W, H)
    a = place(fig, 36, 212, 70, 163)
    xs = [0.0, 0.25, 0.5, 1.0]
    mt = g["mean_true_shift"]["tuning"]
    band = [0.1 * mt["1.0"], 0.1 * mt["0.25"], 0.1 * mt["0.5"], 0.1 * mt["1.0"]]
    a.fill_between(xs, [-v for v in band], band, color=RAMP[0], lw=0, zorder=0)
    a.axhline(0, color=REF, lw=0.8, zorder=1)
    cx = "#7F2704"
    grey = "#B8BEC6"                            # the five other estimators recede; Claude 1 and the script lead
    style = {"claude_d": (grey, "o", True, "-", 0.7), "claude_e": (grey, "o", True, "-", 0.7),
             "openai_a": (grey, "o", True, "-", 0.7), "openai_d": (grey, "o", True, "-", 0.7),
             "openai_e": (grey, "o", True, "-", 0.7), "script": (STRONG2, "X", True, "-", 1.3),
             "claude_a": (EDIT, "P", True, "-", 1.7)}
    lead = ("claude_a", "script")
    off = {k: (i - 3) * 0.012 for i, k in enumerate(style)}
    lo_all, hi_all = [], []
    for est, (col, mk, fill, ls, lw) in style.items():
        lv = g["levels"].get(est, {})
        pts = [(x + off[est], lv[str(x)]) for x in xs if lv.get(str(x), {}).get("complete")]
        if not pts:
            continue
        a.plot([p[0] for p in pts], [p[1]["mean"] for p in pts], color=col, lw=lw, ls=ls, marker="", zorder=2)
        for x, v in pts:
            if "interval" in v and est in lead:
                a.plot([x, x], v["interval"], color=col, lw=0.8, marker="", zorder=3)
                lo_all.append(v["interval"][0]); hi_all.append(v["interval"][1])
            mark(a, x, v["mean"], col, mk, fill, 24 if est in lead else 7, z=4 if est in lead else 2)
            lo_all.append(v["mean"]); hi_all.append(v["mean"])
    # Claude's first minus each of its repeats, paired on the same resampled stays (the text's main quantity)
    pc = "#8C2D04"
    for b_, ls_ in (("claude_d", (0, (4, 2))), ("claude_e", (0, (1, 1.5)))):
        pv = g["paired"][f"claude_a-{b_}"]
        pts = [(x - 0.02, pv[str(x)]) for x in xs if pv.get(str(x), {}).get("complete")]
        a.plot([p[0] for p in pts], [p[1]["mean"] for p in pts], color=pc, lw=1.0, ls=ls_, marker="", zorder=3)
        for x, v in pts:
            if "interval" in v:
                a.plot([x, x], v["interval"], color=pc, lw=0.6, marker="", zorder=3)
                lo_all.append(v["interval"][0]); hi_all.append(v["interval"][1])
            mark(a, x, v["mean"], pc, "o", False, 10, z=4)
    y0, y1 = min(lo_all + [-2.0]) - 0.3, max(hi_all + [2.0]) + 0.3
    a.set(xlim=(-0.08, 1.08), ylim=(y0, y1), xticks=xs, xticklabels=["0", "0.25", "0.5", "1"],
          xlabel="Effect strength (× the main world's)", ylabel="Estimator error (mg/dL)")
    for i, t in enumerate(("orange: Claude's first estimator; brown, hollow:", "its excess over runs 2, 3 (dashed, dotted)",
                           "blue: full script; grey: the other five", "bars: 95% over stays; shading: the band")):
        fig.text(36 / W, (31 - 10 * i) / H, t, fontsize=LAB, color=MUTED, ha="left", va="bottom")
    save(fig, "case2_graded_col")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    nf = HERE / "numbers.tex" if (HERE / "numbers.tex").is_file() else HERE / "numbers_case2.tex"
    ap.add_argument("--numbers", default=str(nf), help="the macros file the figures read")
    ap.add_argument("--out", default=str(OUT), help="where the figures go")
    ap.add_argument("--letters", action="store_true", help="lowercase bold panel letters")
    a_ = ap.parse_args()
    load_numbers(Path(a_.numbers))
    OUT = Path(a_.out)
    OUT.mkdir(parents=True, exist_ok=True)
    LETTERS = a_.letters
    fig_readouts()
    fig_readouts_col()
    fig_schematic()
    fig_dose()
    fig_cost()
    fig_estimator()
    fig_graded()
    fig_graded_col()
    print("figures written to", OUT)
