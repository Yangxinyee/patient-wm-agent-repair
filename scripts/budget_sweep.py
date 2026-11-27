"""The submitted insulin edits scaled to the largest swept factor within a
looser value-NLL budget, read on held-out stays.

    python scripts/budget_sweep.py run       (GPU; export + agent_eval per factor, serial, skips done ones)
    python scripts/budget_sweep.py summary   -> results/budget_sweep.json
Synthetic data only.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
#: target -> {budget: factor}, from results/strength_sweep.json by the design's rule
PLAN = {"codexA_ins": {"0.005": 1.1556, "0.007": 1.3112, "0.010": 1.3112},
        "codexB_ins": {"0.005": 1.1017, "0.007": 1.1017, "0.010": 1.1017},
        "codexC_ins": {"0.005": 1.1854, "0.007": 1.3216, "0.010": 1.3707},
        "claudeC_ins": {"0.005": 1.0728, "0.007": 1.0728, "0.010": 1.0728}}
RUN = {"codexA_ins": "openai_a", "codexB_ins": "openai_d", "codexC_ins": "openai_e", "claudeC_ins": "claude_e"}


def tag(t: str, a: float) -> str:
    return f"budget_{t}_a{a}"


def evaluated() -> set:
    f = ROOT / "results" / "agent_eval.jsonl"
    return {json.loads(x)["method"] for x in open(f)} if f.is_file() else set()


def run() -> None:
    sys.path[:0] = [str(ROOT), str(ROOT / "agent"), str(ROOT / "scripts")]
    import strength_sweep as ss
    for t, by in PLAN.items():
        for a in sorted(set(by.values())):
            if tag(t, a) in evaluated():
                continue
            d = ss.export(t, a)
            st = json.loads((d / "state.json").read_text())       # agent_eval.py records these: take them from the run
            src = json.loads((ROOT / "agent_runs" / RUN[t] / "state.json").read_text())
            st.update({k: src[k] for k in ("agent", "passes", "runs")})
            (d / "state.json").write_text(json.dumps(st) + "\n")
            print(t, a, "->", d, flush=True)
            subprocess.run([PY, str(ROOT / "scripts" / "agent_eval.py"), str(d), tag(t, a)], check=True, timeout=3600)


def summary() -> None:
    ev = {json.loads(x)["method"]: json.loads(x) for x in open(ROOT / "results" / "agent_eval.jsonl")}
    def verdict(r):
        lo, hi = r[1], r[2]
        return "within" if lo >= 0.9 and hi <= 1.1 else ("beyond" if hi < 0.9 or lo > 1.1 else "straddles")
    out = {"budgets": {}}
    for b in ("0.005", "0.007", "0.010"):
        rows = {}
        for run_ in ("claude_a", "claude_d"):
            rows[run_] = {"a": 1.0, "ratio": ev[f"agent_{run_}"]["insulin_x2"]["ratio"],
                          "value_nll_change": ev[f"agent_{run_}"]["value_nll_change"]}
        for t, by in PLAN.items():
            e = ev[tag(t, by[b])]
            rows[RUN[t]] = {"a": by[b], "ratio": e["insulin_x2"]["ratio"], "value_nll_change": e["value_nll_change"]}
        vs = [verdict(r["ratio"]) for r in rows.values()]
        out["budgets"][b] = {"rows": rows, "within": vs.count("within"), "straddles": vs.count("straddles"),
                             "beyond": vs.count("beyond")}
    (ROOT / "results" / "budget_sweep.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({b: {k: v for k, v in x.items() if k != "rows"} for b, x in out["budgets"].items()}))


if __name__ == "__main__":
    {"run": run, "summary": summary}[sys.argv[1]]()
