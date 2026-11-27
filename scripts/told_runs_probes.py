"""The told runs (rule 3 fixed before the runs): from each run's own Python blocks (not its prose), the turns whose code rolls
the model out at a tripled or stopped dose, and the turns whose code scores heads other than the value head. Flags are
candidates for reading by hand; one example per run is printed. CPU, text only.

    python scripts/told_runs_probes.py [runs...]   -> results/told_runs_probes.json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = sys.argv[1:] or ["claude_i", "claude_j", "claude_k", "claude_l", "openai_i", "openai_j", "openai_k", "openai_l"]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
DOSE = re.compile(r"(?i)insulin|norepi|\bne\b|dose|rate|drug")
#: a factor list (bracketed, at least three numbers, containing 0.5 or 1.5) that includes 3 (x3) or 0 (x0), or an identifier
#: naming tripling / stopping, in code that also names a drug, dose, rate or action
FLIST = re.compile(r"[\[\(]\s*(?:-?\d+(?:\.\d*)?\s*,\s*){2,}-?\d+(?:\.\d*)?\s*,?\s*[\]\)]")
def _has(lst, v):
    xs = [float(x) for x in re.findall(r"-?\d+(?:\.\d*)?", lst)]
    return v in xs and (1.5 in xs or 0.5 in xs)          # a multiplier list, not lags or tiers
class _L:
    def __init__(self, v, word):
        self.v, self.w = v, re.compile(word)
    def search(self, code):
        for m in FLIST.finditer(code):
            if _has(m.group(0), self.v):
                return m
        return self.w.search(code)
X3 = _L(3.0, r"(?i)tripl\w*|\bx3\b|_x3\b|×3")
X0 = _L(0.0, r"(?i)\bx0\b|_x0\b|×0|zero_dose|stop_dose")
HEADS = re.compile(r"(?i)every (?:output|head)|all (?:heads|outputs)|other heads|total (?:log )?loss|time_head|event_head|code_head|"
                   r"attr_heads\.(?!value)\w+|(?:cls|type|time|event)\w*_(?:nll|loss)|summed over")
out = {}
for r in RUNS:
    d = ROOT / "agent_runs" / r / "turns"
    if not d.is_dir():
        continue
    res = {"x3": [], "x0": [], "heads": []}
    ex = {}
    for t in sorted(d.iterdir()):
        a = t / "answer.md"
        if not a.is_file():
            continue
        code = "\n".join(BLOCK.findall(a.read_text(errors="replace")))
        if not code:
            continue
        for k, rx in (("x3", X3), ("x0", X0), ("heads", HEADS)):
            m = rx.search(code)
            if m and (k == "heads" or re.search(r"(?i)insulin|norepi|dose|action|drug|rate", code)):
                res[k].append(t.name)
                ex.setdefault(k, (t.name, code[max(0, m.start() - 160):m.end() + 100]))
    out[r] = {k: {"n_turns": len(v), "turns": v} for k, v in res.items()}
    print(f"== {r}: x3 {len(res['x3'])}, x0 {len(res['x0'])}, heads {len(res['heads'])}")
    for k, (tn, s) in ex.items():
        print(f"  [{k} {tn}] " + s.replace("\n", " | ")[:260])
(ROOT / "results" / "told_runs_probes.json").write_text(json.dumps(out, indent=1) + "\n")
