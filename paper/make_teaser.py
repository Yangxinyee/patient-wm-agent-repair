"""Figure 1 of the paper: the unedited world model, the agent's repair loop, and the two-world test.

    PAPER_FIGURES=<directory holding figkit.py> python <this script>   (figlib/ beside the script is used first)
figkit preset B with Arial metrics at the two-column template's text width (486 bp); the signal palette's roles: deep blue for the unedited
model, vivid orange for the agents' edit, crimson for the shortcut warning. The test column follows the
paper's three results (the asked question, where no one asked, every output head) and states the replay as one question.
The first two zones fill their height. The first is the chain simulator -> training stays -> world model with
the shortcut in a card at its foot; the second is agent, tools and the submitted edit, which the arrow takes to the test.
Every number is read from the paper's numbers file (scripts/paper_numbers.py).
"""
import json
import os
import re
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parent / "figlib"
sys.path.insert(0, str(_LIB) if (_LIB / "figkit.py").is_file() else os.environ["PAPER_FIGURES"])
import figkit as fk  # noqa: E402
from figkit import WHITE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
T = fk.use_theme("B", width_pt=455, font="arial")   # ACL's text width
BLUE, ORANGE, CRIMSON = "#0F4D92", "#E8590C", "#C2185B"
ev = {json.loads(x)["run"]: json.loads(x) for x in open(ROOT / "results" / "agent_eval.jsonl")}
r_claude = ev["claude_a"]["insulin_x2"]["ratio"][0]
r_codex = ev["openai_a"]["insulin_x2"]["ratio"][0]
_NUM = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (ROOT / "paper" / "numbers.tex").read_text()))
def sg_(v):
    return v.replace("$-$", "−").replace("$+$", "+")


rep_claude, rep_codex = (sg_(_NUM[k]) for k in ("repFinImplied", "codexRepNull"))
b_claude = ev["claude_c"]["insulin_x2"]["ratio"][0]
b_codex = ev["openai_c"]["insulin_x2"]["ratio"][0]


def build():
    H = 372
    B, SM = T.body, T.small
    d = fk.Figure("fig1_teaser", H, "Agent repair of a patient world model", str(OUT))
    top, bot = 30, 360

    pad = 12                                   # inner margin of zones 1 and 2
    L = SM * 1.2                               # line height of the small type

    # ---- zone 1: where the unedited world model comes from, and its shortcut --------------------
    x1, w1 = 6, 232
    d.text(x1, 2, w1, "Unedited world model", B, True, align="center")
    d.box(x1, top, w1, bot - top, T.panel, None, T.r_outer)
    S, R = 38, 2 * L + 2                       # tile size, row height (two lines of text)
    hc = pad + L + 8 + 2 * L + pad             # the shortcut card: warning line, then the reading
    yc = bot - pad - hc
    gap = (yc - (top + pad) - 3 * R) / 3       # rows and card evenly spaced down the zone
    rows1 = [("gear", "ICU simulator,\nknown drug effects", None),
             ("db", "Training stays", None),
             ("nn", "Event-stream\nworld model", BLUE)]
    ys = [top + pad + i * (R + gap) for i in range(3)]
    for (g, lab, col), y in zip(rows1, ys):
        fk.tile(d, g, x1 + pad + 4, y + (R - S) / 2, S, col)
        d.ctext(x1 + pad + 4 + S + 12, y, w1 - S - 2 * pad - 20, R, lab, SM, align="left")
    for y0, y1_ in zip(ys, ys[1:]):
        cx = x1 + pad + 4 + S / 2
        d.line([(cx, y0 + (R + S) / 2 + 3), (cx, y1_ + (R - S) / 2 - 3)], arrow=True)
    d.box(x1 + pad, yc, w1 - 2 * pad, hc, WHITE, None, T.r_inner)
    fk.warn(d, x1 + pad + 10, yc + pad + 2, 15, CRIMSON)
    d.text(x1 + pad + 32, yc + pad, w1 - 2 * pad - 40, "dose follows glucose", SM, color=CRIMSON)
    d.text(x1 + pad + 10, yc + pad + L + 8, w1 - 2 * pad - 20, f"insulin ×2 read at {_NUM['plainZIx']}\nof the true effect",
           SM, color=T.muted)
    y_wm = ys[2] + R / 2                       # the world model's row: the arrow into zone 2 leaves here

    # ---- zone 2: the agent, its tools and the edit it submits ---------------------------------------
    x2, w2 = 268, 424
    d.text(x2, 2, w2, "Agent correction, no answer given", B, True, align="center")
    d.box(x2, top, w2, bot - top, WHITE, T.ramp[2], T.r_outer)
    tg = 30                                    # gap between the tool cards, where the submit arrow runs
    tw_, th_ = (w2 - 2 * pad - tg) / 2, 74
    ty = y_wm - th_ / 2                        # tool row level with the world model's row
    aw, ah = 196, 66
    ax_, ay = x2 + (w2 - aw) / 2, top + pad
    d.box(ax_, ay, aw, ah, "#FDEBDD", ORANGE, T.r_inner, T.lw_box * 1.2)
    fk.tile(d, "chip", ax_ + 10, ay + (ah - 32) / 2, 32, ORANGE)
    d.ctext(ax_ + 52, ay, aw - 58, ah, "LLM agent\nClaude · Codex", SM, True, align="left")
    tools = [("server", "Python on\nweights, records"), ("chart", "Forward tool: fit,\ndose-edit rollouts")]
    for i, (g, lab) in enumerate(tools):
        tx = x2 + pad + i * (tw_ + tg)
        d.box(tx, ty, tw_, th_, T.panel, None, T.r_inner)
        fk.tile(d, g, tx + 8, ty + (th_ - 30) / 2, 30)
        d.ctext(tx + 46, ty, tw_ - 50, th_, lab, SM, align="left")
        cx = tx + tw_ / 2
        fk.arrow2(d, [(cx, ay + ah + 2), (cx, ty - 2)])
    oh = pad + 2 * L + 4 + pad                 # the submitted edit, and how it was made (the zone title says no answer)
    oy = bot - pad - oh
    ox, ow = x2 + pad, w2 - 2 * pad
    d.box(ox, oy, ow, oh, T.panel, None, T.r_inner)
    o1, o2 = "Submitted edit", "without gradients"
    gw = 34 + 14 + max(fk.tw(o1, SM, True), fk.tw(o2, SM))
    gx = ox + (ow - gw) / 2                    # tile and text centred in the card, under the agent
    fk.tile(d, "nn", gx, oy + (oh - 34) / 2, 34, ORANGE)
    d.text(gx + 48, oy + pad, gw - 48, o1, SM, True)
    d.text(gx + 48, oy + pad + L + 4, gw - 48, o2, SM, color=T.muted)
    xm = x2 + w2 / 2                           # the agent submits: down between the two tool cards
    d.line([(xm, ay + ah + 2), (xm, oy - 2)], arrow=True)

    # ---- zone 3: the known-truth test, in the order of the paper's three results ------------------
    x3, w3 = 724, 270
    d.text(x3, 2, w3, "Held-out test, known truth", B, True, align="center")
    rows = [("Asked: insulin ×2 vs truth", f"within a tenth in {_NUM['SubWithin']} of {_NUM['SubN']}\nsubmissions, both Claude's"),
            ("Where no one asked", f"stopped: every correction fails\ntripled: {_NUM['SubEditThreeShort']} of {_NUM['SubEditN']} fall short\n"
             f"lab glucose: agents {_NUM['GenAgentsIlabMin']}–{_NUM['GenAgentsIlabMax']}"),
            ("Every output head", f"{_NUM['OtherTotOverN']} of {_NUM['OtherTotN']} on other models raise\ntotal loss beyond seed spread")]
    hs = [SM * 1.4 + SM * 1.2 * (1 + b.count("\n")) + 14 for _, b in rows]
    by = top
    for (head, body), h in zip(rows, hs):
        d.box(x3, by, w3, h, T.panel, None, T.r_outer)
        d.text(x3 + 10, by + 7, w3 - 20, head, SM, True)
        d.text(x3 + 10, by + 7 + SM * 1.4, w3 - 20, body, SM)
        by += h + 8
    d.text(x3 + 4, by + 2, w3 - 8, "replay: was the estimate right,\nand did the edit install it?", SM, color=T.muted)

    # ---- flows between zones: the world model into the agent's tools; the submitted edit to the test --
    d.line([(x1 + w1, y_wm), (x2 - 1, y_wm)], arrow=True)
    xg = (x2 + w2 + x3) / 2                    # up the gutter to the first test, the question asked
    d.line([(ox + ow, oy + oh / 2), (xg, oy + oh / 2), (xg, top + hs[0] / 2), (x3 - 1, top + hs[0] / 2)], arrow=True)
    return d.save()


if __name__ == "__main__":
    build()
    fk.check_errors()
    fk.render(OUT / "fig1_teaser.pdf")
    print(fk.stats_table())
