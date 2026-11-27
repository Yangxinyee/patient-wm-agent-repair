"""Declined replies of the Claude runs (the provider's safeguard ended the reply; meta.json "refused"), in the further main worlds (key c1_claude) and the rerun:
per run the count and the longest run of consecutive declines; over the further worlds' five Claude runs, the share declined by what the
turn sent (a Python result, a forward result, a restarted conversation's message, the note after a declined reply), the
share declined right after a declined turn (same conversation) against the first turns of a new conversation, and the
median output tokens of declined and answered turns.

    python3 scripts/refusal_breakdown.py   -> results/refusal_breakdown.json
Reads agent_runs/<run>/turns/t*/{prompt.txt,meta.json} and tokens.jsonl only. CPU, synthetic setting.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FURTHER = ["claude_w1", "claude_w2", "claude_w3", "claude_w4", "claude_w5"]
RERUN = ["claude_w3r"]
OLD_NOTE = "[The output of your last step could not be delivered"


def kind(prompt, thread):
    if prompt.startswith(OLD_NOTE):
        return "note"
    if thread == "new":
        return "restart"
    if prompt.startswith("OUTPUT of Python run"):
        return "python"
    if prompt.startswith("Forward pass"):
        return "forward"
    return "other"


def turns(run):
    d = ROOT / "agent_runs" / run
    tok = {json.loads(x)["turn"]: json.loads(x) for x in open(d / "tokens.jsonl")}
    out = []
    for t in sorted(d.glob("turns/t*"), key=lambda p: int(p.name[1:])):
        if not (t / "meta.json").is_file():
            continue                                       # a turn sent but never answered (a paused run's last)
        n = int(t.name[1:])
        meta = json.loads((t / "meta.json").read_text())
        thread = tok.get(n, {}).get("thread", "resume")
        out.append({"turn": n, "refused": bool(meta.get("refused")), "kind": kind((t / "prompt.txt").read_text(), thread),
                    "thread": thread, "out": (tok.get(n) or {}).get("output")})
    return out


def longest(ts):
    best = cur = 0
    start = best_start = None
    for x in ts:
        if x["refused"]:
            start = x["turn"] if cur == 0 else start
            cur += 1
        else:
            cur = 0
        if cur > best:
            best, best_start = cur, start
    return best, best_start


def main():
    res = {"runs": {}, "c1_claude": {}}
    allt = {}
    for run in FURTHER + RERUN:
        ts = turns(run)
        allt[run] = ts
        lg, st = longest(ts)
        res["runs"][run] = {"turns": len(ts), "declined": sum(x["refused"] for x in ts), "longest_streak": lg,
                            "streak_start": st}
    pooled = [x for r in FURTHER for x in allt[r]]
    by = {}
    for x in pooled:
        b = by.setdefault(x["kind"], [0, 0])
        b[0] += 1
        b[1] += x["refused"]
    res["c1_claude"]["by_kind"] = {k: {"turns": n, "declined": d, "share": round(d / n, 3)} for k, (n, d) in by.items()}
    after, first = [0, 0], [0, 0]
    for r in FURTHER:
        ts = allt[r]
        for prev, x in zip(ts, ts[1:]):
            if prev["refused"] and x["thread"] == "resume":
                after[0] += 1
                after[1] += x["refused"]
        for x in ts:
            if x["thread"] == "new":
                first[0] += 1
                first[1] += x["refused"]
    res["c1_claude"]["after_declined_same_conversation"] = {"turns": after[0], "declined": after[1],
                                                            "share": round(after[1] / after[0], 3)}
    res["c1_claude"]["first_turn_new_conversation"] = {"turns": first[0], "declined": first[1],
                                                       "share": round(first[1] / first[0], 3)}
    od = [x["out"] for x in pooled if x["refused"] and x["out"]]
    oa = [x["out"] for x in pooled if not x["refused"] and x["out"]]
    res["c1_claude"]["median_output_tokens"] = {"declined": statistics.median(od), "answered": statistics.median(oa)}
    res["c1_claude"]["turns"], res["c1_claude"]["declined"] = len(pooled), sum(x["refused"] for x in pooled)
    (ROOT / "results" / "refusal_breakdown.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
