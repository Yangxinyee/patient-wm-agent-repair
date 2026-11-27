"""Two checks offline, from existing results and logs, under the keys C2 and C3 of the output.

C3: for each submission marked for many declined replies (Claude runs with more than any seed-0 Claude run), the turn of
the first declined reply and of the first forward pass that tested an edit (tool_calls.jsonl, ok, edit hash set).
C2: whether the replayed estimators' inert-world bias orders their main-world errors (results already in the macros'
sources: the inert-world pooled bias, results/null_seed_batch.json via paper_numbers; the held-out estimate and edit,
results/decompose_same_stays.json) -- recorded here as the ranks.

    python3 scripts/appendix_checks_offline.py   -> results/appendix_checks_offline.json
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def c3():
    subs = json.loads((ROOT / "results" / "submissions.json").read_text())["submissions"]
    seed0_max = max(json.loads((ROOT / "agent_runs" / r / "state.json").read_text()).get("refusals", 0)
                    for r in ("claude_a", "claude_d", "claude_e"))
    out = {}
    for s in subs:
        if s["agent"] != "Claude" or s["declined"] <= seed0_max:
            continue
        r = s["run"]
        first_edit = next((json.loads(x)["turn"] for x in open(ROOT / "agent_runs" / r / "tool_calls.jsonl")
                           if json.loads(x).get("type") == "forward" and json.loads(x).get("ok")
                           and json.loads(x).get("edit_hash") not in (None, "none", "")), None)
        first_ref = next(int(m.group(1)) for m in (re.search(r"turn (\d+): refused", x)
                                                   for x in open(ROOT / "agent_runs" / r / "driver.log")) if m)
        out[r] = {"declined": s["declined"], "first_declined_turn": first_ref, "first_edit_tested_turn": first_edit,
                  "declines_began_before_first_tested_edit": first_edit is None or first_ref < first_edit}
    return out


def c2():
    d = json.loads((ROOT / "results" / "decompose_same_stays.json").read_text())
    return {"note": "inert-world bias from results/null_seed_batch.json; held-out estimate and edit ratios from "
                    "results/decompose_same_stays.json (the macros Nb*Mean, Ss*EstH, Ss*EditH)", "keys": sorted(d)[:12]}


if __name__ == "__main__":
    res = {"C3": c3(), "C2": c2()}
    (ROOT / "results" / "appendix_checks_offline.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["C3"], indent=1))
