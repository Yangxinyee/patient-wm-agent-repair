"""Write paper/numbers.tex: every number the draft quotes, as LaTeX macros read from the result files.

    python3 scripts/paper_numbers.py
A run whose evaluation is missing gets "??" so the gap shows in the PDF.
"""
import math
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
readout = [json.loads(x) for x in open(R / "readout.jsonl")]
agents = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_eval.jsonl")}
parts = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_parts.jsonl")}
EDITS = {"insulin_x2": "Ix", "insulin_half": "Ih", "ne_x2": "Nx", "ne_half": "Nh"}
RUNS = {"claude_a": "ClaudeW", "openai_a": "CodexW", "claude_c": "ClaudeB", "openai_c": "CodexB",
        "claude_b": "ClaudeN", "openai_b": "CodexN"}
out = []


def m(name, value):
    out.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def half_up(x, d):
    """x rounded half-up to d decimals, from its shortest decimal form (repr), so 0.155 -> 0.16; zero is unsigned"""
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP) + 0


def fx(x, d):
    return f"{half_up(x, d):.{d}f}"


def num(x, d=2, sign=False):
    q = half_up(x, d)
    s = f"{q:+.{d}f}" if sign else f"{q:.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def pt(v, shift="mean_shift"):
    """a ratio's point: the stored ratio has four decimals, the shift and the truth four each; the quotient of the two
    shifts is the more precise one when |truth| >= 1 + |ratio| (its rounding error is at most 5e-5 (1 + |ratio|) / |truth|)"""
    r = v["ratio"][0]
    if v.get(shift) and v.get("true_shift") and v["true_shift"][0]:
        t = v["true_shift"][0]
        q = v[shift][0] / t
        assert abs(q - r) < 5e-5 * (2 + abs(r) / abs(t) + 1 / abs(t)) + 1e-9, (q, r)
        return q if abs(t) >= 1 + abs(q) else r
    return r


def ctl(key, method, seed):
    return next(r for r in readout if r["key"] == key and r["method"] == method and r["seed"] == seed)


# truth and the plain model
t = ctl("e03", "es_register", 0)
for e, k in EDITS.items():
    m(f"truth{k}", num(t[e]["true_shift"][0], 1, sign=True))
plain = [pt(ctl("e03", "es_register", s)["insulin_x2"]) for s in range(3)]
m("plainIx", num(plain[0]))
for e, k in EDITS.items():
    v = [pt(ctl("e03", "es_register", s)[e]) for s in range(3)]
    m(f"plain{k}Lo", num(min(v))); m(f"plain{k}Hi", num(max(v)))
# seed 0, the agents' model: the plain model, fine-tuning and steering
for meth, tag in (("es_register", "plainZ"), ("ft_pnp", "ftZ"), ("steer_a0.25", "steerZ"), ("steerp_a32", "probeZ")):
    r0 = ctl("e03", meth, 0)
    for e, k in EDITS.items():
        x, lo, hi = pt(r0[e]), *r0[e]["ratio"][1:]
        m(f"{tag}{k}", num(x)); m(f"{tag}{k}Lo", num(lo)); m(f"{tag}{k}Hi", num(hi))
# steering and fine-tuning, three seeds
for meth, key in (("steer_a0.25", "steer"), ("ft_pnp", "ft")):
    vals = {e: [pt(ctl("e03", meth, s)[e]) for s in range(3)] for e in EDITS}
    for e, k in EDITS.items():
        m(f"{key}{k}Lo", num(min(vals[e]))); m(f"{key}{k}Hi", num(max(vals[e])))
pr = {e: [pt(ctl("e03", "steerp_a32", s)[e]) for s in range(3)] for e in EDITS}
for e, k in EDITS.items():
    m(f"probe{k}Lo", num(min(pr[e]))); m(f"probe{k}Hi", num(max(pr[e])))
null_pr = [ctl("n03", mm, s)["insulin_x2"]["mean_shift"][0] for mm, s in (("steerp_a8", 0), ("steerp_a32", 1), ("steerp_a32", 2))]
m("probeNullLo", num(min(null_pr), 2, True)); m("probeNullHi", num(max(null_pr), 2, True))
null_ft = [ctl("n03", "ft_pnp", s)["insulin_x2"]["mean_shift"][0] for s in range(3)]
m("ftNullLo", num(min(null_ft), 2, True)); m("ftNullHi", num(max(null_ft), 2, True)); m("ftNullZ", num(null_ft[0], 2, True))
null_plain = [ctl("n03", "es_register", s)["insulin_x2"]["mean_shift"][0] for s in range(3)]
m("plainNullLo", num(min(null_plain), 2, True))
m("plainNullHi", num(max(null_plain), 2, True)); m("plainNull", num(null_plain[0], 2, True))
# the full script on all three training seeds of the plain model (scripts/scripted_tune.py with AWR_SEED)
sc = {e: [pt(ctl("e03", "scriptS", s)[e]) for s in range(3)] for e in EDITS}
for e, k in EDITS.items():
    m(f"scriptSeed{k}Lo", num(min(sc[e]))); m(f"scriptSeed{k}Hi", num(max(sc[e])))
null_sc = [ctl("n03", "scriptS", s)["insulin_x2"] for s in range(3)]
m("scriptSeedNullLo", num(min(r["mean_shift"][0] for r in null_sc), 2, True))
m("scriptSeedNullHi", num(max(r["mean_shift"][0] for r in null_sc), 2, True))
m("scriptSeedNullCiHi", num(max(r["mean_shift"][2] for r in null_sc), 2, True))   # the largest upper bound over seeds
null_st = [ctl("n03", "steer_a0.1", s)["insulin_x2"]["mean_shift"][0] for s in range(3)]
m("steerNullZ", num(null_st[0], 2, True)); m("probeNullZ", num(null_pr[0], 2, True))
m("steerNullHi", num(max(null_st), 2, True)); m("steerNullLo", num(min(null_st), 2, True))
# agents
for run, k in RUNS.items():
    a = agents.get(run)
    if a is None:
        for e, ek in EDITS.items():
            m(f"{k}{ek}", "??"); m(f"{k}{ek}Lo", "??"); m(f"{k}{ek}Hi", "??")
        m(f"{k}Nll", "??")
        continue
    for e, ek in EDITS.items():
        x, lo, hi = ((pt(a[e]), *a[e]["ratio"][1:]) if a["world"] == "e03" else a[e]["mean_shift"])
        m(f"{k}{ek}", num(x)); m(f"{k}{ek}Lo", num(lo)); m(f"{k}{ek}Hi", num(hi))
    x = a["value_nll_change"]
    m(f"{k}Nll", num(x, 5 if abs(x) < 0.0005 else 4, True))
# repeated runs. Main world, weight access: three runs per agent (A: the first, B and C: the repeats, same
# prompt, budget and tools); no-effect, weight access: the original world (A) and fresh worlds fA (B) and fB (C)
REP = {"Claude": ("claude_a", "claude_d", "claude_e"), "Codex": ("openai_a", "openai_d", "openai_e")}
NULLRUNS = {"Claude": ("claude_b", "claude_h", "claude_g"), "Codex": ("openai_b", "openai_h", "openai_g")}
BAND_IN = lambda v: Decimal("0.9") <= half_up(v["ratio"][1], 2) and half_up(v["ratio"][2], 2) <= Decimal("1.1")
for ag, runs_ in REP.items():
    for i, run in enumerate(runs_):
        L, a = "ABC"[i], agents.get(run)
        for e, ek in EDITS.items():
            if a:
                x, lo, hi = pt(a[e]), *a[e]["ratio"][1:]
                m(f"Run{ag}{L}{ek}", num(x)); m(f"Run{ag}{L}{ek}Lo", num(lo)); m(f"Run{ag}{L}{ek}Hi", num(hi))
            else:
                m(f"Run{ag}{L}{ek}", "??"); m(f"Run{ag}{L}{ek}Lo", "??"); m(f"Run{ag}{L}{ek}Hi", "??")
        m(f"Run{ag}{L}Nll", (num(a["value_nll_change"], 5 if abs(a["value_nll_change"]) < 0.0005 else 4, True)) if a else "??")
    done = [r for r in runs_ if r in agents]
    xs = [pt(agents[r]["insulin_x2"]) for r in done]
    m(f"Run{ag}N", str(len(xs)))
    m(f"Run{ag}Min", num(min(xs)) if xs else "??"); m(f"Run{ag}Max", num(max(xs)) if xs else "??")
    m(f"Run{ag}Within", str(sum(BAND_IN(agents[r]["insulin_x2"]) for r in done)))
for ag, runs_ in NULLRUNS.items():
    for i, run in enumerate(runs_):
        a = agents.get(run)
        m(f"Null{ag}{'ABC'[i]}", ("0.00" if half_up(a["insulin_x2"]["mean_shift"][0], 2) == 0 else num(a["insulin_x2"]["mean_shift"][0], 2, True)) if a else "??")
        m(f"Null{ag}{'ABC'[i]}Nll", (num(a["value_nll_change"], 5, True)) if a else "??")
    m(f"Null{ag}N", str(sum(r in agents for r in runs_)))
# every end-to-end no-effect run whose edit reads exactly zero on all four contrasts (held-out stays of its world)
nz = [r for rs in NULLRUNS.values() for r in rs if r in agents]
m("NullRunsAll", str(len(nz)))
m("NullRunsZero", str(sum(all(abs(agents[r][e]["mean_shift"][0]) < 5e-4 for e in EDITS) for r in nz)))
# the fresh worlds' unedited models (scripts/edited_rollout.py with AWR_WORK_ROOT; method es_register_fA / _fB)
for w in ("fA", "fB"):
    rr = [r for r in readout if r["key"] == "n03" and r["method"] == f"es_register_{w}" and r["seed"] == 0]
    m(f"PlainNull{w[1]}", num(rr[0]["insulin_x2"]["mean_shift"][0], 2, True) if rr else "??")
# information-matched baselines without an agent (scripts/scripted_*.py), seed 0
BASE = {"scriptC": "ScriptC", "scriptS": "ScriptS", "ftC": "FtSized"}
for meth, k in BASE.items():
    r0 = ctl("e03", meth, 0)
    for e, ek in EDITS.items():
        x, lo, hi = pt(r0[e]), *r0[e]["ratio"][1:]
        m(f"{k}{ek}", num(x)); m(f"{k}{ek}Lo", num(lo)); m(f"{k}{ek}Hi", num(hi))
x, lo, hi = ctl("n03", "scriptS", 0)["insulin_x2"]["mean_shift"]
m("ScriptSNull", num(x, 2, True)); m("ScriptSNullLo", num(lo, 2, True)); m("ScriptSNullHi", num(hi, 2, True))
for w, wk in (("e03", "Main"), ("n03", "Null")):
    est = json.loads((R / "scripted" / f"estimate_{w}.json").read_text())["pairs"]
    for pair, pk in (("I->glucose_finger", "I"), ("N->map_arterial", "N")):
        m(f"scrEst{wk}{pk}", num(est[pair]["sum8"], 1, True)); m(f"scrEst{wk}{pk}Se", num(est[pair]["sum8_se"], 1))
tc = json.loads((R / "scripted" / "tune_scriptC_e03.json").read_text())
m("scrCRounds", str(len(tc["rounds"])))
for e, ek in EDITS.items():
    m(f"scrCTune{ek}", num(tc["rounds"][-1]["readouts"][e], 1, True))
# per-head losses (percent change); the repeated main-world runs too
for run, k in dict(RUNS, claude_d="RunClaudeB", claude_e="RunClaudeC", openai_d="RunCodexB", openai_e="RunCodexC").items():
    p = parts.get(run)
    if p is not None and "monitor" in p["parts"]:
        m(f"{k}MonitorAbs", num(1000 * p["parts"]["monitor"]["change"], 2, True) if abs(p["parts"]["monitor"]["change"]) >= 5e-6 else "0.00")
    else:
        m(f"{k}MonitorAbs", "??")
    for h in ("value_bin", "code", "action_code", "action_count", "time", "count", "cont", "kl", "monitor", "rate_bin"):
        hk = re.sub(r"[^a-z]", "", h.replace("_bin", "").replace("action_code", "drug"))
        if p is None or h not in p["parts"]:
            m(f"{k}Head{hk}", "??"); continue
        q = p["parts"][h]
        pct = 100 * q["change"] / abs(q["original"])
        m(f"{k}Head{hk}", "0.0" if abs(pct) < 0.05 else num(pct, 1, True))
# cost and run statistics
tot, runs_all = 0.0, 0
handoff_rule = None
STAT_RUNS = dict(RUNS, claude_d="RunClaudeB", claude_e="RunClaudeC", openai_d="RunCodexB", openai_e="RunCodexC",
                 claude_h="NullClaudeB", claude_g="NullClaudeC", openai_h="NullCodexB", openai_g="NullCodexC",
                 void_claude_f="VoidClaude", void_openai_f="VoidCodex")
for run, k in STAT_RUNS.items():
    d = ROOT / "agent_runs" / run
    if not (d / "state.json").is_file():
        continue
    s = json.loads((d / "state.json").read_text())
    usd = s.get("usd", 0.0)
    tot += usd
    runs_all += s["runs"]
    m(f"{k}Usd", fx(usd, 2)); m(f"{k}Turns", str(s["turn"])); m(f"{k}Runs", str(s["runs"]))
    m(f"{k}Passes", str(s["passes"])); m(f"{k}Refused", str(s.get("refusals", 0)))
    m(f"{k}Handoffs", str(s["handoffs"]))                    # conversation restarts from the agent's notebook
    m(f"{k}Kstops", str(s.get("pressure_kills", 0)))         # kernel restarts by the harness's memory guard
    rule = (s["handoff"], s["handoff_tokens"])
    assert handoff_rule in (None, rule), "one restart rule for every run"
    handoff_rule = rule
m("HandoffReplies", str(handoff_rule[0])); m("HandoffTokens", f"{handoff_rule[1]:,}".replace(",", "{,}"))
m("totalUsd", fx(tot, 0)); m("totalRuns", f"{runs_all:,}".replace(",", "{,}"))
# safety-filter declines over Claude's three weight-access runs, and over every Codex run
_ref = {r: json.loads((ROOT / "agent_runs" / r / "state.json").read_text()).get("refusals", 0) for r in STAT_RUNS
        if (ROOT / "agent_runs" / r / "state.json").is_file()}
_cw = [_ref[r] for r in REP["Claude"] if r in _ref]
m("FilteredClaudeWMin", str(min(_cw))); m("FilteredClaudeWMax", str(max(_cw)))
m("FilteredCodexAll", str(sum(v for r, v in _ref.items() if "openai" in r)))
# the strength Claude's run 2 gave its norepinephrine channel, as its final answer states it
_fa = re.search(r"Its strength is set to ([0-9.]+)\.", (ROOT / "agent_runs" / "claude_d" / "final_answer.md").read_text())
m("RunClaudeBNeStrength", _fa.group(1) if _fa else "??")
_w = [json.loads((ROOT / "agent_runs" / r / "state.json").read_text()) for rs in REP.values() for r in rs]
m("MinPyRunsW", str(min(x["runs"] for x in _w)))
m("MaxPyRunsW", str(max(x["runs"] for x in _w)))
_fv = json.loads((R / "ft_value_nll.json").read_text())["value_nll_per_reading"]          # the budget's size
_pv = [v["plain"] for v in _fv.values()]
m("PlainValueNllRange", num(max(_pv) - min(_pv), 4)); m("BudgetOverSeedRange", fx(0.005 / (max(_pv) - min(_pv)), 1))                        # fewest Python runs over the six weight-access runs
m("AllDoneW", str(sum(bool(x.get("done")) for x in _w)))
import numpy as np  # noqa: E402


def psi(world, run_no, key):
    rows = [json.loads(x) for x in open(ROOT / "agent_runs" / "replay" / f"claude_a_{world}" / "outputs.jsonl")]
    text = next(o["output"] for o in rows if o["run"] == run_no)
    return np.array([float(v) for v in re.search(rf"psi {key}\s*\[([^\]]*)\]", text).group(1).split()])


fa = (ROOT / "agent_runs" / "claude_a" / "final_answer.md").read_text()
own = {k: float(re.search(rf"{pat}[^|]*\|[^|]*\|[^|]*\|\s*([−+\-0-9.]+)", fa).group(1).replace("−", "-"))
       for k, pat in (("ins", r"Insulin x2"), ("ne", r"Norepinephrine x2"))}
m("ownIx", num(own["ins"], 1, True)); m("ownNx", num(own["ne"], 1, True))
left = re.search(r"with only (\d+) Python runs left", (ROOT / "agent_runs" / "claude_a" / "turns" / "t318" / "answer.md").read_text())
m("ClaudeWRunsLeft", left.group(1) if left else "??")               # Claude's note when it stopped the halved-NE term
hl = re.search(r"([0-9.]+)-hour half-life", fa)                    # the memory slot's decay in Claude's circuit
m("ClaudeHalfLife", hl.group(1) if hl else "??")
for key, run_no, tag in (("finger", 16, "Fin"), ("lab", 16, "Lab"), ("art", 17, "Art")):
    a_, b_ = psi("e03", run_no, key), psi("n03", run_no, key)
    m(f"rep{tag}Main", num(a_.sum(), 1, True)); m(f"rep{tag}Null", num(b_.sum(), 1, True))
    if tag == "Fin":       # Eq. (2) on the replayed finger-stick profile, tuning stays (scripts/audit_eq2.py, draw -1)
        fq = ROOT / "results" / "audit_eq2" / "claude_a_n03.jsonl"
        e2 = next((json.loads(x)["eq2"] for x in open(fq) if json.loads(x)["draw"] == -1), None) if fq.is_file() else None
        m("repFinImplied", num(e2, 1, True) if e2 is not None else "??")
# generalization: interventions no agent queried (results/generalize.json, scripts/generalize_summary.py)
GEN_MODELS = {"Plain": "e03:plain model", "Ft": "e03:fine-tuning", "SteerM": "e03:steering, mean difference",
              "SteerP": "e03:steering, probe direction", "CodexB": "e03:Codex, black box", "ClaudeB": "e03:Claude, black box",
              "CodexW": "e03:Codex, white box", "ClaudeW": "e03:Claude, white box",
              "ScriptC": "e03:scripted install, Claude's estimate", "ScriptS": "e03:scripted install, scripted estimate",
              "FtSized": "e03:fine-tuning sized to Claude's estimate",
              # the repeated weight-access runs (B: run 2, C: run 3; run 1 is ClaudeW / CodexW)
              "RunClaudeB": "e03:Claude, white box, run 2", "RunClaudeC": "e03:Claude, white box, run 3",
              "RunCodexB": "e03:Codex, white box, run 2", "RunCodexC": "e03:Codex, white box, run 3"}
GEN_KEYS = {"Izero": ("doses", "insulin_x0@12"), "Ionefive": ("doses", "insulin_x1.5@12"), "Ithree": ("doses", "insulin_x3@12"),
            "Ilab": ("doses", "insulin_x2_lab@12"), "Isix": ("starts", "insulin_x2@6"), "Isixteen": ("starts", "insulin_x2@16"),
            "Ihsix": ("starts", "insulin_half@6"), "Ihsixteen": ("starts", "insulin_half@16"),
            "Nzero": ("doses", "ne_x0@12"), "Nonefive": ("doses", "ne_x1.5@12"), "Nthree": ("doses", "ne_x3@12"),
            "Ncuff": ("doses", "ne_x2_cuff@12"), "Nsix": ("starts", "ne_x2@6"), "Nsixteen": ("starts", "ne_x2@16"),
            "Nhsix": ("starts", "ne_half@6"), "Nhsixteen": ("starts", "ne_half@16")}
gpath = ROOT / "results" / "generalize.json"
gen = json.loads(gpath.read_text()) if gpath.is_file() else {}
# the held-out stays each readout is scored on (the unedited seed-0 model's rollouts; every model shares them)
_gn = gen["e03:plain model"]
m("NStaysIns", str(_gn["original"]["insulin_x2"]["n_stays"])); m("NStaysNe", str(_gn["original"]["ne_x2"]["n_stays"]))
m("NStaysLab", str(_gn["doses"]["insulin_x2_lab@12"]["n_stays"])); m("NStaysCuff", str(_gn["doses"]["ne_x2_cuff@12"]["n_stays"]))
assert all(gen[k]["original"]["insulin_x2"]["n_stays"] == _gn["original"]["insulin_x2"]["n_stays"] for k in gen if k.startswith("e03:") and "original" in gen[k])
for mk, gk in GEN_MODELS.items():
    r = gen.get(gk, {})
    for kk, (design, e) in GEN_KEYS.items():
        v = (r.get(design) or {}).get(e)
        m(f"gen{mk}{kk}", num(pt(v, "model_shift")) if v and "ratio" in v else "??")
        # the 95% interval over 1,000 stay resamples (scripts/generalize_summary.py), for the unqueried columns' verdicts
        m(f"gen{mk}{kk}Lo", num(v["ratio"][1]) if v and "ratio" in v else "??")
        m(f"gen{mk}{kk}Hi", num(v["ratio"][2]) if v and "ratio" in v else "??")
        if v and "ratio" in v:                      # the interval as a table cell, bold when it lies within the band
            lo, hi = half_up(v["ratio"][1], 2), half_up(v["ratio"][2], 2)
            cell = f"{num(v['ratio'][1])}--{num(v['ratio'][2])}"
            m(f"gen{mk}{kk}CI", cell)       # unasked cells are not bolded: the bold of Table 2 marks asked questions only
    d = (r.get("detail") or {}).get("insulin_x2")
    m(f"gen{mk}Early", num(d["horizon"]["hours 13-16"][0]) if d else "??")
    m(f"gen{mk}Late", num(d["horizon"]["hours 21-24"][0]) if d else "??")
    m(f"gen{mk}Corr", num(d["per_stay"]["corr"][0]) if d else "??")
    m(f"gen{mk}CorrLo", num(d["per_stay"]["corr"][1]) if d else "??")
    m(f"gen{mk}CorrHi", num(d["per_stay"]["corr"][2]) if d else "??")
    m(f"gen{mk}Rmse", fx(d['per_stay']['rmse'], 1) if d else "??")
    dn = (r.get("detail") or {}).get("ne_x2")
    m(f"gen{mk}NEarly", num(dn["horizon"]["hours 13-16"][0]) if dn else "??")
    m(f"gen{mk}NLate", num(dn["horizon"]["hours 21-24"][0]) if dn else "??")
# x0 against each edit's own response at x1/2 (a response linear in the dose gives twice the x1/2 shift at x0)
for mk, gk in (("ClaudeW", "e03:Claude, white box"), ("CodexW", "e03:Codex, white box"), ("Ft", "e03:fine-tuning")):
    r = gen.get(gk, {})
    z, h = (r.get("doses") or {}).get("insulin_x0@12"), (r.get("original") or {}).get("insulin_half")
    if z and h:
        m(f"gen{mk}IzeroShift", num(z["model_shift"][0], 1, True)); m(f"gen{mk}IhalfLin", num(2 * h["model_shift"][0], 1, True))
        m(f"gen{mk}IzeroFrac", fx(100 * z['model_shift'][0] / (2 * h['model_shift'][0]), 0))
        if mk == "ClaudeW":
            m("genTruthIzero", num(z["true_shift"][0], 1, True))
# how a correction's shift grows from x2 to x3 (model shift at x3 over model shift at x2, same stays), against the truth's
# growth; a response linear in the dose departure grows 2-fold, a log-rate input log 3 / log 2-fold
for mk, gk in (("ClaudeW", "e03:Claude, white box"), ("ScriptS", "e03:scripted install, scripted estimate"),
               ("ScriptC", "e03:scripted install, Claude's estimate"), ("Ft", "e03:fine-tuning")):
    r = gen.get(gk, {})
    for kk, two, three in (("I", "insulin_x2", "insulin_x3@12"), ("N", "ne_x2", "ne_x3@12")):
        a, b = (r.get("original") or {}).get(two), (r.get("doses") or {}).get(three)
        m(f"grow{mk}{kk}", num(b["model_shift"][0] / a["model_shift"][0]) if a and b else "??")
        if mk == "ClaudeW" and a and b:
            m(f"growTruth{kk}", num(b["true_shift"][0] / a["true_shift"][0]))
m("growLogRatio", num(math.log(3) / math.log(2)))
# the two scripts' readouts at x3, both drugs: their range
three = [pt(gen[g]["doses"][e], "model_shift") for g in ("e03:scripted install, scripted estimate", "e03:scripted install, Claude's estimate")
         for e in ("insulin_x3@12", "ne_x3@12") if g in gen]
m("ThreeScriptsMin", num(min(three)) if three else "??"); m("ThreeScriptsMax", num(max(three)) if three else "??")
d0 = (gen.get("e03:plain model", {}).get("detail") or {}).get("insulin_x2")
m("genTruthSd", fx(d0['per_stay']['truth_sd'], 1) if d0 else "??")
for wk, mk in (("n03:plain model", "PlainN"), ("n03:fine-tuning", "FtN"), ("n03:Codex, white box", "CodexN"), ("n03:Claude, white box", "ClaudeN"),
               ("n03:scripted install, scripted estimate", "ScriptSN")):
    r = gen.get(wk, {}).get("doses") or {}
    for kk, e in (("Izero", "insulin_x0@12"), ("Ionefive", "insulin_x1.5@12"), ("Ithree", "insulin_x3@12"), ("Ilab", "insulin_x2_lab@12"),
                  ("Nzero", "ne_x0@12"), ("Nonefive", "ne_x1.5@12"), ("Nthree", "ne_x3@12"), ("Ncuff", "ne_x2_cuff@12")):
        v = r.get(e)
        m(f"gen{mk}{kk}", num(v["model_shift"][0], 2, True) if v else "??")
# a stopped drug given to the pathway as rate 0 with the whole expected dose subtracted (scripted install, Claude's estimate)
z = gen.get("e03:scripted install, Claude's estimate, stopped dose as rate 0", {}).get("doses") or {}
for kk, e in (("Izero", "insulin_x0@12"), ("Nzero", "ne_x0@12")):
    v = z.get(e)
    m(f"genStopZero{kk}", num(pt(v, "model_shift")) if v else "??")
    m(f"genStopZero{kk}Shift", num(v["model_shift"][0], 1, True) if v else "??")
    m(f"genStopZero{kk}Truth", num(v["true_shift"][0], 1, True) if v else "??")
# per-stay agreement on the queried x2 questions, with a dose-only reference (results/per_stay.json, scripts/per_stay.py)
ps_path = ROOT / "results" / "per_stay.json"
ps = json.loads(ps_path.read_text())["methods"] if ps_path.is_file() else {}
for lab, mk in (("plain model", "Plain"), ("fine-tuning", "Ft"), ("Codex, white box", "CodexW"), ("Claude, white box", "ClaudeW"),
                ("dose only (Claude's kernel)", "Dose"), ("Codex, black box", "CodexB"), ("Claude, black box", "ClaudeB"),
                ("steering, mean difference", "SteerM"), ("steering, probe direction", "SteerP"),
                ("scripted install, Claude's estimate", "ScriptC"), ("scripted install, scripted estimate", "ScriptS"),
                ("fine-tuning sized to Claude's estimate", "FtSized"),
                ("Codex, white box, run 2", "RunCodexB"), ("Claude, white box, run 2", "RunClaudeB"),
                ("Codex, white box, run 3", "RunCodexC"), ("Claude, white box, run 3", "RunClaudeC")):
    for q, qk in (("insulin_x2", "I"), ("ne_x2", "N")):
        r = ps.get(lab, {}).get(q)
        m(f"ps{mk}{qk}Corr", num(r["corr"][0]) if r else "??")
        m(f"ps{mk}{qk}CorrLo", num(r["corr"][1]) if r else "??"); m(f"ps{mk}{qk}CorrHi", num(r["corr"][2]) if r else "??")
        m(f"ps{mk}{qk}Rmse", fx(r['rmse'], 1) if r else "??")
        m(f"ps{mk}{qk}Sf", fx(r['rmse_scale_free'], 1) if r else "??")
        m(f"ps{mk}{qk}Scale", num(r["best_scale"]) if r else "??")
# the four queried contrasts over the six weight-access runs, the decomposition of each edit into its estimate and
# its installation, the x3 dose encoding, and the per-stay correlations of every agent edit
_runs6 = [r for rs in REP.values() for r in rs]
DK8 = {"claude_a": "ClaudeA", "claude_d": "ClaudeB", "claude_e": "ClaudeC", "openai_a": "CodexA", "openai_d": "CodexB", "openai_e": "CodexC"}
_within = sum(BAND_IN(agents[r][e]) for r in _runs6 for e in EDITS)
m("QueriedWithin", str(_within)); m("QueriedCells", str(len(_runs6) * len(EDITS)))
m("QueriedWithinMain", str(sum(BAND_IN(agents[r]["insulin_x2"]) for r in _runs6)))
def _verdict(v):
    lo, hi = half_up(v["ratio"][1], 2), half_up(v["ratio"][2], 2)
    return "within" if Decimal("0.9") <= lo and hi <= Decimal("1.1") else ("beyond" if hi < Decimal("0.9") or lo > Decimal("1.1") else "straddles")
_vs = [_verdict(agents[r][e]) for r in _runs6 for e in EDITS]
m("QueriedStraddle", str(_vs.count("straddles"))); m("QueriedBeyond", str(_vs.count("beyond")))
# the same rule on the two scripts, a second band width (+-0.2) and counts by points alone
def _verdict_w(v, w):
    lo, hi = half_up(v["ratio"][1], 2), half_up(v["ratio"][2], 2)
    a, b = Decimal(1) - Decimal(w), Decimal(1) + Decimal(w)
    return "within" if a <= lo and hi <= b else ("beyond" if hi < a or lo > b else "straddles")
_pin = lambda v, w: Decimal(1) - Decimal(w) <= half_up(pt(v), 2) <= Decimal(1) + Decimal(w)
_vs2 = [_verdict_w(agents[r][e], "0.2") for r in _runs6 for e in EDITS]
m("QueriedWithinTwo", str(_vs2.count("within"))); m("QueriedStraddleTwo", str(_vs2.count("straddles")))
m("QueriedBeyondTwo", str(_vs2.count("beyond")))
m("QueriedPointWithin", str(sum(_pin(agents[r][e], "0.1") for r in _runs6 for e in EDITS)))
m("QueriedPointWithinTwo", str(sum(_pin(agents[r][e], "0.2") for r in _runs6 for e in EDITS)))
VK = {"within": "within", "straddles": "straddles", "beyond": "beyond"}
for meth, k in (("scriptC", "ScriptC"), ("scriptS", "ScriptS")):
    r0 = ctl("e03", meth, 0)
    vv = [_verdict_w(r0[e], "0.1") for e in EDITS]
    for e, ek in EDITS.items():
        m(f"{k}{ek}Verdict", _verdict_w(r0[e], "0.1"))
    m(f"{k}Within", str(vv.count("within"))); m(f"{k}Straddle", str(vv.count("straddles"))); m(f"{k}Beyond", str(vv.count("beyond")))
    m(f"{k}WithinTwo", str(sum(_verdict_w(r0[e], "0.2") == "within" for e in EDITS)))
    m(f"{k}PointWithin", str(sum(_pin(r0[e], "0.1") for e in EDITS)))
for r in _runs6:
    for e, ek in EDITS.items():
        m(f"V{DK8[r]}{ek}", {"within": "\\vIn", "straddles": "\\vEdge", "beyond": "\\vOut"}[_verdict_w(agents[r][e], "0.1")])
for meth, k in (("scriptC", "ScriptC"), ("scriptS", "ScriptS")):
    r0 = ctl("e03", meth, 0)
    for e, ek in EDITS.items():
        m(f"V{k}{ek}", {"within": "\\vIn", "straddles": "\\vEdge", "beyond": "\\vOut"}[_verdict_w(r0[e], "0.1")])
# verdicts by question (insulin x2, insulin x1/2, both norepinephrine questions) and the script's NE widths
for qk, qs in (("Ix", ("insulin_x2",)), ("Ih", ("insulin_half",)), ("N", ("ne_x2", "ne_half"))):
    vq = [_verdict(agents[r][e]) for r in _runs6 for e in qs]
    m(f"Q{qk}Within", str(vq.count("within"))); m(f"Q{qk}Straddle", str(vq.count("straddles"))); m(f"Q{qk}Beyond", str(vq.count("beyond")))
_sw = [ctl("e03", "scriptS", 0)[e]["ratio"][2] - ctl("e03", "scriptS", 0)[e]["ratio"][1] for e in ("ne_x2", "ne_half")]
m("ScriptSNeWidthMin", num(min(_sw))); m("ScriptSNeWidthMax", num(max(_sw)))
# the second measurement by interval for the six agent edits and the full script; the stopped dose
# given as rate 0 (its interval); the cells where "within" is reachable
_v2 = lambda lo, hi: "within" if Decimal("0.9") <= half_up(lo, 2) and half_up(hi, 2) <= Decimal("1.1") else (
    "beyond" if half_up(hi, 2) < Decimal("0.9") or half_up(lo, 2) > Decimal("1.1") else "straddles")
_G6 = ("e03:Claude, white box", "e03:Claude, white box, run 2", "e03:Claude, white box, run 3", "e03:Codex, white box",
       "e03:Codex, white box, run 2", "e03:Codex, white box, run 3")
_sec = {g_: [_v2(*gen[g_]["doses"][e]["ratio"][1:]) for e in ("insulin_x2_lab@12", "ne_x2_cuff@12")] for g_ in _G6}
_agent_any_edge = [g_ for g_, v in _sec.items() if "beyond" not in v or any(x != "beyond" for x in v)]
m("SecAgentBeyondN", str(sum(all(x == "beyond" for x in v) for v in _sec.values())))
m("SecAgentEdgeN", str(sum(any(x != "beyond" for x in v) for v in _sec.values())))
m("SecLabBeyondN", str(sum(v[0] == "beyond" for v in _sec.values()))); m("SecLabEdgeN", str(sum(v[0] == "straddles" for v in _sec.values())))
m("SecCuffBeyondN", str(sum(v[1] == "beyond" for v in _sec.values())))
_sc = [_v2(*gen["e03:scripted install, scripted estimate"]["doses"][e]["ratio"][1:]) for e in ("insulin_x2_lab@12", "ne_x2_cuff@12")]
m("SecScriptBeyond", "both" if all(x == "beyond" for x in _sc) else "not both")
_z0 = gen["e03:scripted install, Claude's estimate, stopped dose as rate 0"]["doses"]["insulin_x0@12"]["ratio"]
m("genStopZeroIzeroLo", num(_z0[1])); m("genStopZeroIzeroHi", num(_z0[2])); m("genStopZeroVerdict", _v2(*_z0[1:]))
m("ReachCells", str(len(_runs6))); m("ClosedCells", str(len(_runs6) * 3))
m("IxHalfWidth", num(float(np.mean([(agents[r]["insulin_x2"]["ratio"][2] - agents[r]["insulin_x2"]["ratio"][1]) / 2 for r in _runs6]))))
_r2h = [pt(agents[r]["insulin_half"]) / pt(agents[r]["insulin_x2"]) for r in _runs6]
m("HalfRatioMin", num(min(_r2h))); m("HalfRatioMax", num(max(_r2h)))
_r2s = [pt(ctl("e03", k_, 0)["insulin_half"]) / pt(ctl("e03", k_, 0)["insulin_x2"]) for k_ in ("scriptC", "scriptS")]
m("HalfRatioScriptMin", num(min(_r2s))); m("HalfRatioScriptMax", num(max(_r2s)))
_ih = [pt(agents[r]["insulin_half"]) for r in _runs6]
m("HalfMin", num(min(_ih))); m("HalfMax", num(max(_ih)))
m("LinHalfCeil", num(0.5 * abs(t["insulin_x2"]["true_shift"][0]) / abs(t["insulin_half"]["true_shift"][0])))
m("HalfTruthShare", fx(100 * abs(t["insulin_half"]["true_shift"][0]) / abs(t["insulin_x2"]["true_shift"][0]), 0))
_dec = json.loads((ROOT / "results" / "decompose_edits.json").read_text())["runs"]
DK = {"claude_a": "ClaudeA", "claude_d": "ClaudeB", "claude_e": "ClaudeC", "openai_a": "CodexA", "openai_d": "CodexB", "openai_e": "CodexC"}
for r, k in DK.items():
    m(f"Dec{k}Est", num(_dec[r]["estimate_ratio"])); m(f"Dec{k}Edit", num(_dec[r]["edit_ratio"])); m(f"Dec{k}Inst", num(_dec[r]["installation_ratio"]))
m("DecEstMin", num(min(v["estimate_ratio"] for v in _dec.values()))); m("DecEstMax", num(max(v["estimate_ratio"] for v in _dec.values())))
m("DecInstMin", num(min(v["installation_ratio"] for v in _dec.values()))); m("DecInstMax", num(max(v["installation_ratio"] for v in _dec.values())))
# estimate and installation on the same stays (results/decompose_same_stays.json, scripts/decompose_same_stays.py)
_ss = json.loads((ROOT / "results" / "decompose_same_stays.json").read_text())
SK = dict(DK, script_claude_estimate="ScriptC", script="ScriptS")
m("SsTruthT", num(_ss["truth_tuning"], 2, True)); m("SsTruthH", num(_ss["runs"]["claude_a"]["held_out"]["truth"], 2, True))
for r, k in SK.items():
    v = _ss["runs"][r]
    t, h = v["tuning"], v["held_out"]
    m(f"Ss{k}EstT", num(t["implied"], 2, True)); m(f"Ss{k}EditT", num(t["readout"], 2, True))
    m(f"Ss{k}EstTr", num(t["estimate_ratio"])); m(f"Ss{k}EditTr", num(t["edit_ratio"]))
    m(f"Ss{k}InstT", num(t["paired"]["installation"]))
    m(f"Ss{k}InstTLo", num(t["paired"]["installation_ci"][0])); m(f"Ss{k}InstTHi", num(t["paired"]["installation_ci"][1]))
    m(f"Ss{k}Res", num(t["readout"] - t["implied"], 2, True))
    m(f"Ss{k}EstH", num(h["estimate_ratio"])); m(f"Ss{k}EditH", num(h["edit_ratio"])); m(f"Ss{k}InstH", num(h["installation"]))
_ag = {r: _ss["runs"][r] for r in DK}
_inst = lambda rs: [_ag[r]["tuning"]["paired"]["installation"] for r in rs]
m("SsInstClaudeMin", num(min(_inst(REP["Claude"])))); m("SsInstClaudeMax", num(max(_inst(REP["Claude"]))))
m("SsInstCodexMin", num(min(_inst(REP["Codex"])))); m("SsInstCodexMax", num(max(_inst(REP["Codex"]))))
_sc = [_ss["runs"][r]["tuning"]["paired"]["installation"] for r in ("script_claude_estimate", "script")]
m("SsInstScriptMin", num(min(_sc))); m("SsInstScriptMax", num(max(_sc)))
for r, k in SK.items():                    # held-out estimate, edit and installation with paired intervals
    hp = _ss["runs"][r]["held_out"]["paired"]
    for q, qk in (("estimate_ratio", "Est"), ("edit_ratio", "Edit"), ("installation", "Inst")):
        m(f"Ss{k}{qk}HLo", num(hp[q][1])); m(f"Ss{k}{qk}HHi", num(hp[q][2]))
_dr = {r: _ss["runs"][r]["drop"] for r in SK}
m("SsDropLoMin", num(min(_dr[r][1] for r in DK), 2, True)); m("SsDropHiMax", num(max(_dr[r][2] for r in DK), 2, True))
m("SsDropScriptMin", num(min(_dr[r][0] for r in ("script_claude_estimate", "script"))))
m("SsDropScriptMax", num(max(_dr[r][0] for r in ("script_claude_estimate", "script"))))
_drop = [_ag[r]["tuning"]["paired"]["installation"] - _ag[r]["held_out"]["installation"] for r in DK]
m("SsDropMin", num(min(_drop))); m("SsDropMax", num(max(_drop)))
_miss = [r for r in DK if not BAND_IN(agents[r]["insulin_x2"])]        # the edits whose main-question interval leaves the band
m("SsMissN", str(len(_miss)))
m("SsMissInstHMin", num(min(_ag[r]["held_out"]["installation"] for r in _miss)))
m("SsMissInstHMax", num(max(_ag[r]["held_out"]["installation"] for r in _miss)))
_est_ok = [abs(_ag[r][s_]["estimate_ratio"] - 1) for r in DK if r != "openai_a" for s_ in ("tuning", "held_out")]
m("SsEstDevMax", num(max(_est_ok)))
m("SsEstMin", num(min(_ag[r][s_]["estimate_ratio"] for r in DK if r != "openai_a" for s_ in ("tuning", "held_out"))))
m("SsEstMax", num(max(_ag[r][s_]["estimate_ratio"] for r in DK if r != "openai_a" for s_ in ("tuning", "held_out"))))
# the insulin x3 readout split by whether the tripled doses reach the model's top dose bin (results/x3_split.json)
_xs = json.loads((ROOT / "results" / "x3_split.json").read_text())["methods"]
XK = {"claude_a": "ClaudeA", "claude_d": "ClaudeB", "claude_e": "ClaudeC", "openai_a": "CodexA", "openai_d": "CodexB",
      "openai_e": "CodexC", "scriptS": "ScriptS", "scriptC": "ScriptC"}
for t_, k in XK.items():
    for g_, gk in (("below_top_bin", "Below"), ("top_bin", "Top")):
        r_ = _xs[t_][g_]["ratio"]
        m(f"Xs{k}{gk}", num(r_[0])); m(f"Xs{k}{gk}Lo", num(r_[1])); m(f"Xs{k}{gk}Hi", num(r_[2]))
for grp, ks in (("Agents", ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e")), ("Scripts", ("scriptS", "scriptC")),
                ("AgentsNoCodexA", ("claude_a", "claude_d", "claude_e", "openai_d", "openai_e"))):
    for g_, gk in (("below_top_bin", "Below"), ("top_bin", "Top")):
        vals = [_xs[t_][g_]["ratio"][0] for t_ in ks]
        m(f"Xs{grp}{gk}Min", num(min(vals))); m(f"Xs{grp}{gk}Max", num(max(vals)))
_x0 = _xs["claude_a"]
m("XsTopShare", fx(100 * _x0["top_bin"]["n_stay_hours"] / _x0["all"]["n_stay_hours"], 0))
m("XsTruthBelow", num(_x0["below_top_bin"]["true_shift"], 1, True)); m("XsTruthTop", num(_x0["top_bin"]["true_shift"], 1, True))
m("XsFtBelow", num(_xs["ft"]["below_top_bin"]["ratio"][0])); m("XsFtTop", num(_xs["ft"]["top_bin"]["ratio"][0]))
# resampled copies per graded world (fewest over the strengths' worlds): Codex's run 2 against the others
_gs = json.loads((ROOT / "results" / "graded_summary.json").read_text())["levels"]
_gc = {e: min(v["copies_min"] for b, v in lv.items() if v.get("complete") and "copies_min" in v) for e, lv in _gs.items()}
m("GCopiesCodexB", str(_gc["openai_d"])); m("GCopiesOthers", str(min(v for e, v in _gc.items() if e != "openai_d")))
# Claude 1's total-NLL change with a paired interval, and the unedited model's seed spread (results/total_nll_interval.json)
_tn = json.loads((ROOT / "results" / "total_nll_interval.json").read_text())["pairs"]
m("TotClaudeWLo", num(_tn["claude_a"]["interval"][0], 1, True)); m("TotClaudeWHi", num(_tn["claude_a"]["interval"][1], 1, True))
m("TotSeedOne", num(_tn["seed1"]["change_per_stay"], 1, True)); m("TotSeedTwo", num(_tn["seed2"]["change_per_stay"], 1, True))
# the interval unsigned after "rises by"
m("TotClaudeWLoAbs", fx(abs(_tn["claude_a"]["interval"][0]), 1)); m("TotClaudeWHiAbs", fx(abs(_tn["claude_a"]["interval"][1]), 1))
# strength sweep (results/strength_sweep.json)
_sp = ROOT / "results" / "strength_sweep.json"
if _sp.is_file():
    exec((ROOT / "scripts" / "sweep_macros.py").read_text())
else:
    m("SweepAbstract", ""); m("SweepResults", ""); m("SweepConclusion", ""); m("SweepIntro", ""); m("SweepResultsNe", ""); m("SweepClaudeC", "")
# installs in other worlds (results/world_install/)
_wi = ROOT / "results" / "world_install"
if (ROOT / "scripts" / "world_install_macros.py").is_file():
    exec((ROOT / "scripts" / "world_install_macros.py").read_text())
else:
    m("BiResult", "")
# the coded final answers against the band verdicts (results/transcript_codes.json, the coding manual)
_tc = json.loads((ROOT / "results" / "transcript_codes.json").read_text())["runs"]
_QK = {"insulin_x2": "Ix", "insulin_half": "Ih", "ne_x2": "Nx", "ne_half": "Nh"}
_TK = {"claude_a": "ClaudeA", "claude_d": "ClaudeB", "claude_e": "ClaudeC", "openai_a": "CodexA", "openai_d": "CodexB",
       "openai_e": "CodexC", "claude_c": "ClaudeIn", "openai_c": "CodexIn", "claude_b": "ClaudeNa", "claude_h": "ClaudeNb",
       "claude_g": "ClaudeNc", "openai_b": "CodexNa", "openai_h": "CodexNb", "openai_g": "CodexNc"}
_ae10 = {json.loads(x)["run"]: json.loads(x) for x in open(R / "agent_eval.jsonl")}
def _beyond(r, e):
    a = _ae10[r]
    if a["world"] == "e03":
        return _verdict(a[e]) == "beyond"
    lo, hi = a[e]["mean_shift"][1], a[e]["mean_shift"][2]
    return hi < -1.42 or lo > 1.42
_tot = {"k": 0, "m": 0, "kmain": 0, "mmain": 0, "over": 0}
for r, k_ in _TK.items():
    c = _tc[r]
    bey = {e: _beyond(r, e) for e in _QK}
    kk = sum(bey[e] and c["named"][e] for e in _QK); mm = sum(bey.values())
    over = sum(c["named"][e] and not bey[e] for e in _QK)
    m(f"Tc{k_}K", str(kk)); m(f"Tc{k_}M", str(mm)); m(f"Tc{k_}Reason", c["reason"])
    for e, ek in _QK.items():
        m(f"Tc{k_}{ek}", ("b" if bey[e] else "") + ("$\\bullet$" if c["named"][e] else "") or "--")
    _tot["k"] += kk; _tot["m"] += mm; _tot["over"] += over
    if r in ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e"):
        _tot["kmain"] += kk; _tot["mmain"] += mm
m("TcK", str(_tot["k"])); m("TcM", str(_tot["m"])); m("TcMainK", str(_tot["kmain"])); m("TcMainM", str(_tot["mmain"]))
m("TcOver", str(_tot["over"])); m("TcUnasked", str(sum(c["unasked_checked"] for c in _tc.values())))
# split the main-world count by cause (halved insulin, whose gap a linear estimate cannot show its own
# author, apart from the rest), and the reasons the six main-world answers give
_MR = ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e")
_hk = sum(_beyond(r, "insulin_half") and _tc[r]["named"]["insulin_half"] for r in _MR)
_hm = sum(_beyond(r, "insulin_half") for r in _MR)
m("TcHalfK", str(_hk)); m("TcHalfM", str(_hm))
m("TcRestK", str(_tot["kmain"] - _hk)); m("TcRestM", str(_tot["mmain"] - _hm))
for q_, qk_ in (("insulin_x2", "Ix"), ("ne_x2", "Nx"), ("ne_half", "Nh")):
    m(f"TcMain{qk_}K", str(sum(_beyond(r, q_) and _tc[r]["named"][q_] for r in _MR)))
    m(f"TcMain{qk_}M", str(sum(_beyond(r, q_) for r in _MR)))
_rs = [_tc[r]["reason"] for r in _MR]
m("TcBudgetN", str(_rs.count("fit budget"))); m("TcNoReasonN", str(_rs.count("none")))
m("TcNamedAnyN", str(sum(any(_tc[r]["named"].values()) for r in _MR)))
m("TcNamedInsN", str(sum(_tc[r]["named"]["insulin_x2"] or _tc[r]["named"]["insulin_half"] for r in _MR)))
# the agent runs the transcript table covers (every run with a final answer, dry and discarded runs excluded)
_ar = [d.name for d in (ROOT / "agent_runs").iterdir() if d.is_dir() and (d / "final_answer.md").is_file()
       and not d.name.startswith(("dry", "void"))
       and not (d.name in ("claude_i", "claude_j", "claude_k", "claude_l", "openai_i", "openai_j", "openai_k", "openai_l")
                or "_w" in d.name)]                     # the told and further-world runs have their own appendix
m("AgentRunsN", str(len(_ar)))
# norepinephrine, each run's estimate against what its edit installs on the tuning stays (results/ne_installation.json)
_ne = json.loads((ROOT / "results" / "ne_installation.json").read_text())
m("NeTruthT", num(_ne["truth_ne_x2_tuning"], 2, True))
for e, ek in EDITS.items():                                 # the tuning-stay truths of the four contrasts
    m(f"tunTruth{ek}", num(_ne["truth_tuning"][e], 1, True))
for r, k in DK.items():
    m(f"Ne{k}Read", num(_ne["runs"][r]["ne_x2_tuning_readout"], 2, True))
m("NeCodexBEst", num(_ne["runs"]["openai_d"]["ne_x2_estimate"], 2, True))
m("NeCodexCEstLo", num(_ne["runs"]["openai_e"]["ne_x2_estimate_range"][0], 1, True))
m("NeCodexCEstHi", num(_ne["runs"]["openai_e"]["ne_x2_estimate_range"][1], 1, True))
m("NeCodexBShare", num(_ne["runs"]["openai_d"]["installed_share"])); m("NeCodexCShare", num(_ne["runs"]["openai_e"]["installed_share"]))
# ten new no-effect worlds beside the earlier eleven (results/null_seed_batch.json)
_nb = json.loads((ROOT / "results" / "null_seed_batch.json").read_text())
EK = {"claude_a": "ClaudeA", "claude_d": "ClaudeB", "claude_e": "ClaudeC", "openai_a": "CodexA", "openai_d": "CodexB",
      "openai_e": "CodexC", "script": "Script"}
m("NbWorlds", str(len(_nb["sets"]["six"]) + len(_nb["sets"]["five"]) + len(_nb["sets"]["ten"]))); m("NbNew", str(len(_nb["sets"]["ten"])))
for e, k in EK.items():
    v = _nb["estimators"][e]
    m(f"Nb{k}Mean", num(v["mean_21"], 2, True)); m(f"Nb{k}Lo", num(v["interval_21"][0], 2, True)); m(f"Nb{k}Hi", num(v["interval_21"][1], 2, True))
    m(f"Nb{k}Verdict", v["verdict_21"]); m(f"Nb{k}Six", num(v["mean_six"], 2, True)); m(f"Nb{k}Five", num(v["mean_five"], 2, True))
    m(f"Nb{k}Ten", num(v["mean_ten"], 2, True))
m("NbClaudeAPos", str(_nb["claude_a_positive_21"]))
for e, k in EK.items():                          # each of the ten new worlds, for the appendix table
    for i_, v_ in enumerate(_nb["estimators"][e]["ten"]):
        m(f"Nb{k}T{'ABCDEFGHIJ'[i_]}", num(v_, 2, True))
for b, k in (("claude_d", "B"), ("claude_e", "C")):
    x = _nb[f"excess_claude_a_minus_{b}"]
    m(f"NbExcess{k}", num(x["mean_21"], 2, True)); m(f"NbExcess{k}Lo", num(x["interval_21"][0], 2, True))
    m(f"NbExcess{k}Hi", num(x["interval_21"][1], 2, True)); m(f"NbExcess{k}Pos", str(x["positive_21"]))
    m(f"NbExcess{k}Ten", num(x["mean_ten"], 2, True)); m(f"NbExcess{k}Eleven", num(x["mean_eleven"], 2, True))
    m(f"NbExcess{k}Six", num(x["mean_six"], 2, True)); m(f"NbExcess{k}Five", num(x["mean_five"], 2, True))
# the pre-written rule against each reference set: half of Claude 1's inert-world bias
_ca = _nb["estimators"]["claude_a"]
_g9 = json.loads((ROOT / "results" / "graded_summary.json").read_text())
_z0 = _g9["levels"]["claude_a"]["0.0"]["mean"]; _ic = _g9["intercepts"]["claude_a"]["intercept"]
m("GClaudeAZeroHalf", num(_z0 / 2, 2, True)); m("GRuleMargin", num(_ic - _z0 / 2, 2, True))
# the rule's fit through the three strengths' means, its 18 inputs, and each cell's verdict against the
# band at that strength (a tenth of the mean true shift there; at 0 the main band)
_icd = _g9["intercepts"]["claude_a"]
m("GClaudeAIcptMeans", num(_icd["intercept_means"], 2, True))
_LV = {"0.25": "Quarter", "0.5": "Half", "1.0": "One"}
for i_, (w_, x_, e_) in enumerate(_icd["inputs"]):
    lv = ("0.25", "0.5", "1.0")[i_ // 6]; si = "ABCDEF"[i_ % 6]
    m(f"GInp{si}{_LV[lv]}T", num(x_, 2, True)); m(f"GInp{si}{_LV[lv]}E", num(e_, 2, True))
_EKG = {"claude_a": "ClaudeA", "claude_d": "ClaudeB", "claude_e": "ClaudeC", "openai_a": "CodexA", "openai_d": "CodexB",
        "openai_e": "CodexC", "script": "Script"}
_VM = {"within": "\\vIn", "straddles": "\\vEdge", "beyond": "\\vOut"}
for e_, k_ in _EKG.items():
    for lv, lk in {"0.0": "Zero", **_LV}.items():
        v_ = _g9["levels"][e_].get(lv, {})
        m(f"GV{k_}{lk}", _VM.get(v_.get("verdict"), "\\vEdge"))
for lv, lk in {"0.0": "Zero", **_LV}.items():
    m(f"GBand{lk}", num(_g9["levels"]["claude_a"][lv]["band"], 2))
    m(f"GWithin{lk}", str(sum(_g9["levels"][e_].get(lv, {}).get("verdict") == "within" for e_ in _EKG)))

m("NbClaudeAHalf", num(_ca["mean_21"] / 2, 2, True)); m("NbClaudeASixHalf", num(_ca["mean_six"] / 2, 2, True))
# the criterion fixed before the runs, per estimator: new points above the six's mean and below the five's mean
for e, k in (("claude_a", "ClaudeA"), ("claude_d", "ClaudeB"), ("claude_e", "ClaudeC"), ("script", "Script")):
    v = _nb["estimators"][e]
    m(f"Nb{k}AboveSix", str(v["ten_above_mean_six"])); m(f"Nb{k}BelowFive", str(v["ten_below_mean_five"]))
    m(f"Nb{k}Sits", "yes" if v["sits_with_one_set"] else "no")
_g4 = [_nb["estimators"][e]["six_minus_rest"] for e in ("claude_a", "claude_d", "claude_e", "script")]
m("SixGapMin", num(min(_g4))); m("SixGapMax", num(max(_g4)))
_pc = [_nb["estimators"][e]["perm_p_six_high"] for e in ("claude_a", "claude_d", "claude_e")]
m("PermPClaudeMin", num(min(_pc), 3)); m("PermPClaudeMax", num(max(_pc), 3)); m("PermPScript", num(_nb["estimators"]["script"]["perm_p_six_high"], 2))
m("NbSitsN", str(sum(_nb["estimators"][e]["sits_with_one_set"] for e in ("claude_a", "claude_d", "claude_e", "script"))))
_beyond = [k for e, k in EK.items() if _nb["estimators"][e]["verdict_21"] == "beyond"]
m("NbBeyondN", str(len(_beyond)))
m("NbStraddleN", str(sum(_nb["estimators"][e]["verdict_21"] == "straddles" for e in EK)))
m("NbWithinN", str(sum(_nb["estimators"][e]["verdict_21"] == "within" for e in EK)))
m("NbReading", "world-level spread" if _nb.get("reading", "").startswith("case 1") else "a seed-range effect")
# the five agent edits other than Claude's first: the largest change in any other head's loss (percent; the latent
# forecast, whose loss can be negative, is left out as in tab:heads' percentages)
_oh = []
for r in ("claude_d", "claude_e", "openai_a", "openai_d", "openai_e"):
    for h, q in parts[r]["parts"].items():
        if h not in ("value_bin", "monitor"):
            _oh.append(abs(100 * q["change"] / abs(q["original"])))
m("OtherHeadMaxPct", fx(max(_oh), 1))
_x3 = json.loads((ROOT / "results" / "x3_encoding.json").read_text())
m("XThreeTopBin", fx(100 * _x3["x3"]["top_bin"], 0)); m("XTwoTopBin", fx(100 * _x3["x2"]["top_bin"], 0))
m("XThreeClip", fx(100 * _x3["x3"]["z_clipped"], 1)); m("XThreeAboveMax", fx(100 * _x3["x3"]["above_train_max"], 1))
_psa = [ps[l]["insulin_x2"]["corr"][0] for l in ps if l.startswith(("Codex, white box", "Claude, white box"))]
m("psAgentMin", num(min(_psa))); m("psAgentMax", num(max(_psa)))
# Codex's third run's one-hour effect and persistence per world (scripts/codex_persistence.py; the bound is 1.2)
_cp6 = json.loads((ROOT / "results" / "codex_persistence.json").read_text())
_nw = [v for k, v in _cp6.items() if k.startswith("n")]
_mw = [v for k, v in _cp6.items() if k.startswith("e")]
m("CodexCNullBetaMin", num(min(v["beta"] for v in _nw), 2, True)); m("CodexCNullBetaMax", num(max(v["beta"] for v in _nw), 2, True))
m("CodexCNullRhoBound", str(sum(v["persistence"] >= 1.2 - 1e-9 for v in _nw)))
m("CodexCNullRhoMax", num(max(v["persistence"] for v in _nw)))
m("CodexCMainRhoMin", num(min(v["persistence"] for v in _mw))); m("CodexCMainRhoMax", num(max(v["persistence"] for v in _mw)))
m("CodexCMainBetaMin", num(min(v["beta"] for v in _mw), 2, True)); m("CodexCMainBetaMax", num(max(v["beta"] for v in _mw), 2, True))
# the probe's control, the dose and the latest reading alone (scripts/probe_control.py), main world
_pc6 = json.loads((ROOT / "results" / "probe_control.json").read_text())
m("ProbeLastReading", num(_pc6["e03"]["dose_and_latest_reading"][0]))
# the best per-stay correlation of any correction (every row but the dose-only reference), insulin x2
_pc = [ps[lab]["insulin_x2"]["corr"][0] for lab in ps if not lab.startswith("dose only") and "insulin_x2" in ps[lab]]
m("psMaxEditICorr", num(max(_pc)) if _pc else "??")
# glucose-tier occupancy of the training records, main vs no-effect worlds (scripts/tier_occupancy.py)
_to = json.loads((ROOT / "results" / "tier_occupancy.json").read_text())
for kind, tk in (("e", "Main"), ("n", "Null")):
    mm = _to[f"mean_{kind}"]
    m(f"Tier{tk}TopDoses", fx(100 * mm["top_tier_doses"], 1))          # % of insulin-dosed stay-hours at a reading >=250
    m(f"Tier{tk}TopReadings", fx(100 * mm["top_tier_readings"], 1))
    m(f"Tier{tk}StepDoses", fx(100 * (mm["share_doses"]["260-300"] + mm["share_doses"][">=300"]), 1))   # above the 260 step
    ks = [k for k in _to if k.startswith(kind) and not k.startswith("mean")]
    m(f"Tier{tk}TopDosesMin", fx(100 * min(_to[k]["top_tier_doses"] for k in ks), 1))
    m(f"Tier{tk}TopDosesMax", fx(100 * max(_to[k]["top_tier_doses"] for k in ks), 1))
m("TierRatio", fx(_to["mean_n"]["top_tier_doses"] / _to["mean_e"]["top_tier_doses"], 1))
for kind, tk in (("e", "Main"), ("n", "Null")):                     # the appendix table, one macro per bin
    mm = _to[f"mean_{kind}"]
    for lab, bk in zip(("<150", "150-180", "180-220", "220-250", "250-260", "260-300", ">=300"), "ABCDEFG"):
        m(f"Tier{tk}D{bk}", fx(100 * mm["share_doses"][lab], 1))
        m(f"Tier{tk}R{bk}", fx(100 * mm["share_readings"][lab], 1))
        m(f"Tier{tk}U{bk}", fx(mm["mean_rate"][lab], 1))
# norepinephrine x2 over the agents' weight-access runs other than Claude's first (it did not repeat)
_nx = [pt(agents[r]["ne_x2"]) for rs in REP.values() for r in rs if r in agents and r != "claude_a"]
m("NxOthersMin", num(min(_nx))); m("NxOthersMax", num(max(_nx)))
# budgets that bind (Codex's forward passes in every main-world weight-access run)
_cp = [json.loads((ROOT / "agent_runs" / r / "state.json").read_text())["passes"] for r in REP["Codex"]]
m("CodexMainPassesAll", str(_cp[0]) if len(set(_cp)) == 1 else "??")
# the estimator audit along effect strength (scripts/graded_summary.py)
_gs = json.loads((ROOT / "results" / "graded_summary.json").read_text())
GK = {"claude_a": "GClaudeA", "claude_d": "GClaudeB", "claude_e": "GClaudeC", "openai_a": "GCodexA", "openai_d": "GCodexB",
      "openai_e": "GCodexC", "script": "GScript"}
LK = {"0.0": "Zero", "0.25": "Quarter", "0.5": "Half", "1.0": "One"}
for est, gk in GK.items():
    for lv, lk in LK.items():
        v = _gs["levels"].get(est, {}).get(lv, {})
        m(f"{gk}{lk}", num(v["mean"], 2, True) if v.get("complete") else "??")
        m(f"{gk}{lk}Lo", num(v["interval"][0], 2, True) if "interval" in v else "??")
        m(f"{gk}{lk}Hi", num(v["interval"][1], 2, True) if "interval" in v else "??")
        m(f"{gk}{lk}CI", f'{num(v["interval"][0], 2, True)} to {num(v["interval"][1], 2, True)}' if "interval" in v else "--")
    ic = _gs["intercepts"].get(est, {})
    m(f"{gk}Icpt", num(ic["intercept"], 2, True) if ic.get("complete") else "??")
    m(f"{gk}IcptLo", num(ic["interval"][0], 2, True) if "interval" in ic else "??")
    m(f"{gk}IcptHi", num(ic["interval"][1], 2, True) if "interval" in ic else "??")
    m(f"{gk}IcptCI", f'{num(ic["interval"][0], 2, True)} to {num(ic["interval"][1], 2, True)}' if "interval" in ic else "--")
for pk, pm in (("claude_a-claude_d", "GPairB"), ("claude_a-claude_e", "GPairC"), ("claude_a-script", "GPairS")):
    for lv, lk in list(LK.items()) + [("original_null", "Orig")]:
        v = _gs["paired"][pk].get(lv, {})
        m(f"{pm}{lk}", num(v["mean"], 2, True) if v.get("complete") else "??")
        m(f"{pm}{lk}Lo", num(v["interval"][0], 2, True) if "interval" in v else "??")
        m(f"{pm}{lk}Hi", num(v["interval"][1], 2, True) if "interval" in v else "??")
        m(f"{pm}{lk}CI", f'{num(v["interval"][0], 2, True)} to {num(v["interval"][1], 2, True)}' if "interval" in v else "--")
_late = [_gs["paired"][pk][lv]["mean"] for pk in ("claude_a-claude_d", "claude_a-claude_e") for lv in ("0.5", "1.0")
         if _gs["paired"][pk].get(lv, {}).get("complete")]
m("GPairLateMax", num(max(_late), 2, True) if _late else "??")       # the largest paired gap from half strength on
for est, gk in GK.items():                                            # the five paired inert worlds (b000_5101-5105)
    v = _gs["levels"].get(est, {}).get("0.0", {})
    for i, L in enumerate("ABCDE"):
        m(f"{gk}BZero{L}", num(v["points"][i + 1], 2, True) if v.get("complete") else "--")
    e11 = _gs.get("null_eleven", {}).get(est, {})
    m(f"{gk}ElevenCI", f'{num(e11["interval"][0], 2, True)} to {num(e11["interval"][1], 2, True)}' if "interval" in e11 else "--")
    m(f"{gk}ElevenTwoCI", f'{num(e11["two_stage_interval"][0], 2, True)} to {num(e11["two_stage_interval"][1], 2, True)}' if "two_stage_interval" in e11 else "--")
_st = [abs(_gs["levels"][e][lv]["mean"]) for e in ("claude_d", "claude_e", "openai_d", "script") for lv in LK
       if _gs["levels"][e].get(lv, {}).get("complete")]
m("GSteadyMax", num(max(_st), 2) if _st else "??")
# the gap between the two inert-world sets (the six of app:fresh minus the five paired ones), Claude's estimators and the script
_po6 = json.loads((R / "no_effect_pooled.json").read_text())["estimators"]
_gap = [_po6[e]["pooled_all_six"]["mean"] - float(np.mean(_gs["levels"][e]["0.0"]["points"][1:]))
        for e in ("claude_a", "claude_d", "claude_e", "script")]
m("SetGapMin", fx(min(_gap), 1)); m("SetGapMax", fx(max(_gap), 1))
_gq = _gs["levels"]["openai_a"]                                      # Codex's first at weak effect, % of the truth there
m("GCodexAWeakShare", fx(100 * max(_gq["0.25"]["mean"] / abs(_gs["mean_true_shift"]["train"]["0.25"]),
                                   _gq["0.5"]["mean"] / abs(_gs["mean_true_shift"]["train"]["0.5"])), 0))                   # Claude's repeats and the script, any strength
for est, gk in GK.items():                                            # all eleven no-effect worlds
    v = _gs.get("null_eleven", {}).get(est, {})
    m(f"{gk}Eleven", num(v["mean"], 2, True) if v.get("complete") else "??")
    m(f"{gk}ElevenLo", num(v["interval"][0], 2, True) if "interval" in v else "??")
    m(f"{gk}ElevenHi", num(v["interval"][1], 2, True) if "interval" in v else "??")
    m(f"{gk}ElevenPos", str(v["points_positive"]) if v.get("complete") else "??")
    m(f"{gk}ElevenTwoLo", num(v["two_stage_interval"][0], 2, True) if "two_stage_interval" in v else "??")
    m(f"{gk}ElevenTwoHi", num(v["two_stage_interval"][1], 2, True) if "two_stage_interval" in v else "??")
    m(f"{gk}ElevenVerdict", {"within": "within the band", "beyond": "beyond the band", "straddles": "straddling the band's edge"}.get(v.get("verdict"), "??"))
_cf = _gs.get("confounded", {})                                      # the confounded world, points
for est, gk in GK.items():
    v = _cf.get(est, {}).get("mean")
    m(f"{gk}Conf", num(v, 1, True) if v is not None else "--")          # "--": not replayed there
m("ConfTruth", num(-_cf["mean_true_shift"], 1, True) if _cf else "??")
_cv = [_cf[e]["mean"] for e in GK if _cf.get(e, {}).get("mean") is not None]
m("ConfReplayedN", str(len(_cv))); m("ConfBeyondN", str(sum(abs(x) > 0.1 * _cf["mean_true_shift"] for x in _cv)))
m("ConfBand", fx(0.1 * _cf["mean_true_shift"], 1) if _cf else "??")
_ca = [_cf[e]["mean"] for e in ("claude_a", "claude_d", "claude_e") if _cf.get(e, {}).get("mean") is not None]
m("ConfClaudeMin", num(min(_ca), 1, True) if _ca else "??"); m("ConfClaudeMax", num(max(_ca), 1, True) if _ca else "??")
for lv, lk in LK.items():
    m(f"GTier{lk}", fx(100 * _gs["tier_top_doses"][lv], 1))
    m(f"GTruth{lk}", num(-_gs["mean_true_shift"]["tuning"][lv], 1, True) if lv != "0.0" else "0")
# Codex's main-world estimator replayed unchanged on each world's records (agent_runs/replay/openai_a_<world>, its runs 1-24)
for w, wk in (("e03", "Main"), ("n03", "Null")):
    f = ROOT / "agent_runs" / "replay" / f"openai_a_{w}" / "outputs.jsonl"
    rows = [json.loads(x) for x in open(f)] if f.is_file() else []
    o23 = next((o["output"] for o in rows if o["run"] == 23), "")
    o24 = next((o["output"] for o in rows if o["run"] == 24), "")
    fit = re.search(r"RD dynamic fit \[effect, persistence\]: \[\s*([-0-9.e]+)\s+([-0-9.e]+)\]", o23)
    tg = re.search(r"train / first 100 tuning: \(([-0-9.e]+), \d+\) \(([-0-9.e]+), \d+\)", o24)
    m(f"codexRep{wk}Beta", num(float(fit.group(1)), 2, True) if fit else "??")
    m(f"codexRep{wk}Rho", num(float(fit.group(2)), 2) if fit else "??")
    m(f"codexRep{wk}", num(float(tg.group(1)), 1, True) if tg else "??")
    if wk == "Main":
        codex_rep_main = float(tg.group(1)) if tg else None
codex_tune_ix = None
# Codex's white-box readouts on the 400 tuning stays, final weights (final_answer.md, from its tool passes)
fo = (ROOT / "agent_runs" / "openai_a" / "final_answer.md").read_text()
for pat, mk in ((r"Insulin ×2", "Ix"), (r"Insulin ×½", "Ih"), (r"Norepinephrine ×2", "Nx"), (r"Norepinephrine ×½", "Nh")):
    v = re.search(rf"{pat}\s*\|\s*([−+\-0-9.]+)", fo)
    m(f"codexTune{mk}", num(float(v.group(1).replace("−", "-")), 1, True) if v else "??")
    if mk == "Ix" and v:
        codex_tune_ix = float(v.group(1).replace("−", "-"))
# Codex's first estimate on the training stays it averaged over, as a fraction of the truth there
_tt = json.loads((ROOT / "results" / "audit_eq2" / "truth.json").read_text())["e03"]["train"]
m("codexRepMainRatio", num(codex_rep_main / _tt) if codex_rep_main else "??")
# its value-head correction on tuning stays as a fraction of its own sustained estimate (main world, training stays)
m("codexOwnFrac", num(codex_tune_ix / codex_rep_main) if codex_tune_ix is not None and codex_rep_main else "??")
# the unqueried readouts over each agent's three weight-access runs (points; None until a run is in)
AGENT_GEN = {"Claude": ("ClaudeW", "RunClaudeB", "RunClaudeC"), "Codex": ("CodexW", "RunCodexB", "RunCodexC")}
for kk in ("Ithree", "Ilab", "Izero", "Ionefive", "Isix", "Isixteen"):
    allv = []
    for ag, mks in AGENT_GEN.items():
        vs = []
        for mk in mks:
            v = ((gen.get(GEN_MODELS[mk], {}).get(GEN_KEYS[kk][0]) or {}).get(GEN_KEYS[kk][1]))
            if v and "ratio" in v:
                vs.append((pt(v, "model_shift"), v["ratio"][2]))
        allv += vs
        m(f"Gen{ag}{kk}Min", num(min(x for x, _ in vs)) if vs else "??")
        m(f"Gen{ag}{kk}Max", num(max(x for x, _ in vs)) if vs else "??")
    m(f"GenAgents{kk}Min", num(min(x for x, _ in allv)) if allv else "??")
    m(f"GenAgents{kk}Max", num(max(x for x, _ in allv)) if allv else "??")
    m(f"GenAgents{kk}N", str(len(allv)))
    m(f"GenAgents{kk}HiMax", num(max(h for _, h in allv)) if allv else "??")   # the largest upper bound over runs
# total held-out NLL per stay, every head (results/model_parts.jsonl, scripts/model_parts.py)
mp_path = ROOT / "results" / "model_parts.jsonl"
mp, mp_parts = {}, {}
if mp_path.is_file():
    for x in open(mp_path):
        j = json.loads(x)
        mp[(j["world"], j["label"])] = j["total_nll_per_stay"]
        mp_parts[(j["world"], j["label"])] = j["parts"]["value_bin"]
for (w, lab), mk in {("e03", "steering, mean difference"): "SteerM", ("e03", "steering, probe direction"): "SteerP",
                     ("e03", "fine-tuning"): "Ft", ("e03", "openai_c"): "CodexB",
                     ("e03", "claude_c"): "ClaudeB", ("e03", "openai_a"): "CodexW", ("e03", "claude_a"): "ClaudeW",
                     ("n03", "fine-tuning"): "FtN", ("n03", "openai_b"): "CodexN", ("n03", "claude_b"): "ClaudeN",
                     ("e03", "scriptC"): "ScriptC", ("e03", "scriptS"): "ScriptS", ("e03", "ftC"): "FtSized",
                     ("n03", "scriptS"): "ScriptSN",
                     # the repeated weight-access runs
                     ("e03", "claude_d"): "RunClaudeB", ("e03", "claude_e"): "RunClaudeC",
                     ("e03", "openai_d"): "RunCodexB", ("e03", "openai_e"): "RunCodexC"}.items():
    t = mp.get((w, lab))
    m(f"tot{mk}", (num(t["change"], 2, True) if abs(t["change"]) < 0.095 else num(t["change"], 1, True)) if t else "??")
    if mk == "SteerM" and t:
        m("totPlain", fx(t['plain'], 1))
    if mk == "ClaudeW" and t:                                   # unsigned, for the abstract ("raises ... by 6.3 nats")
        m("totClaudeWAbs", fx(abs(t['change']), 1))
        m("totClaudeWPct", fx(100 * t['change'] / t['plain'], 1))
    if mk == "ScriptC" and t:                                   # unsigned, "lowers it by 0.05"
        m("totScriptCAbs", num(abs(t["change"]), 2))
for (w, lab), mk in {("e03", "scriptC"): "ScriptC", ("e03", "scriptS"): "ScriptS", ("e03", "ftC"): "FtSized",
                     ("n03", "scriptS"): "ScriptSN"}.items():
    vb = mp_parts.get((w, lab))
    m(f"{mk}Nll", num(vb["edited"] - vb["plain"], 3, True) if vb else "??")
t = mp.get(("n03", "fine-tuning"))
m("totPlainN", fx(t['plain'], 1) if t else "??")
# the linear probe of the plain model's state (results/probe.json, scripts/probe.py), main world, three seeds
pw = json.loads((ROOT / "results" / "probe.json").read_text())["worlds"]["e03"]
seeds = [pw[f"synth_e03-m-T1-full-long-s{i}"] for i in range(3)]
rnd = pw["random:synth_e03-m-T1-full-long-s0"]


def f2(x):
    return fx(x, 2)


extra = {"ProbeExpLo": f2(min(v["log_expected"]["probe_state_dose"][0] for v in seeds)),
         "ProbeExpHi": f2(max(v["log_expected"]["probe_state_dose"][0] for v in seeds)),
         "ProbeSLo": f2(min(v["s"]["probe_state_dose"][0] for v in seeds)),
         "ProbeSHi": f2(max(v["s"]["probe_state_dose"][0] for v in seeds)),
         "ProbeDoseOnly": f2(seeds[0]["s"]["dose_only"][0]), "ProbeCeil": f2(seeds[0]["s"]["record_ceiling"][0]),
         "ProbeGapLo": fx(100 * min(v['s']['gap_closed_state_dose'] for v in seeds), 0),
         "ProbeGapHi": fx(100 * max(v['s']['gap_closed_state_dose'] for v in seeds), 0),
         "ProbeRandom": f2(rnd["s"]["probe_state_dose"][0]), "ProbeRandomLo": f2(rnd["s"]["probe_state_dose"][1]),
         "ProbeRandomHi": f2(rnd["s"]["probe_state_dose"][2])}
# fine-tuning's held-out value NLL per reading, three seeds (results/ft_value_nll.json)
fv = json.loads((ROOT / "results" / "ft_value_nll.json").read_text())["value_nll_per_reading"]
ft = [fv[f"s{i}"]["fine_tuned"] - fv[f"s{i}"]["plain"] for i in range(3)]
extra["FtNllLo"] = f"{min(ft):+.3f}".replace("-", "$-$").replace("+", "$+$")
extra["FtNllHi"] = f"{max(ft):+.3f}".replace("-", "$-$").replace("+", "$+$")
extra["FtNllZ"] = f"{ft[0]:+.3f}".replace("-", "$-$").replace("+", "$+$")
# Claude's own tuning-stay estimates of the four contrasts (the targets the script sizes to; results/scripted/)
tg = json.loads((ROOT / "results" / "scripted" / "tune_scriptC_e03.json").read_text())["targets"]
for e, k in (("insulin_x2", "Ix"), ("insulin_half", "Ih"), ("ne_x2", "Nx"), ("ne_half", "Nh")):
    extra[f"ClaudeEst{k}"] = f"{tg[e]:+.1f}".replace("-", "$-$").replace("+", "$+$")
# the plain model's size (results/model_size.json, scripts/model_size.py)
ms = json.loads((ROOT / "results" / "model_size.json").read_text())["e03"]
extra["ModelParams"] = fx(ms['n_params'] / 1e6, 1) + "M"
extra["ModelTensors"] = str(ms["n_tensors"])
# the no-effect band: a tenth of the true insulin x2 change on held-out stays (results/readout.jsonl)
truth = next(json.loads(x) for x in open(ROOT / "results" / "readout.jsonl")
             if json.loads(x)["key"] == "e03" and json.loads(x)["method"] == "es_register" and json.loads(x)["seed"] == 0)
extra["Band"] = fx(abs(truth['insulin_x2']['true_shift'][0]) / 10, 2)
# stay bootstrap of the three estimators' implied insulin x2 shift in the no-effect world (scripts/replay_bootstrap.py,
# scripts/scripted_bootstrap.py): the replayed agents' estimators and the script's, training stays resampled
band = abs(truth["insulin_x2"]["true_shift"][0]) / 10
for name, f in (("RepClaude", "claude_a_n03"), ("RepCodex", "openai_a_n03"), ("ScriptEst", "script_n03")):
    # Claude: Eq. (2) on its replayed lag profile (results/audit_eq2); Codex and the script: their own Eq. (2) values
    fp, key = ((ROOT / "results" / "audit_eq2" / f"{f}.jsonl", "eq2") if name == "RepClaude" else
               (ROOT / "results" / "replay_bootstrap" / f"{f}.jsonl", "implied_insulin_x2"))
    rows = list({r["draw"]: r for r in map(json.loads, open(fp))}.values()) if fp.is_file() else []
    point = next((r[key] for r in rows if r["draw"] == -1), None)
    v = np.array([r[key] for r in rows if r["draw"] >= 0 and r.get(key) is not None])
    sgn = lambda x: "0.0" if half_up(x, 1) == 0 else num(x, 1, True)  # noqa: E731
    extra[f"{name}Point"] = sgn(point) if point is not None else "??"
    extra[f"{name}Lo"] = sgn(np.percentile(v, 2.5)) if len(v) else "??"
    extra[f"{name}Hi"] = sgn(np.percentile(v, 97.5)) if len(v) else "??"
    extra[f"{name}Draws"] = str(len(v))
    # against the band on the stays the estimator averages over (scripts/no_effect_pooled.py scored_band)
    _po = json.loads((R / "no_effect_pooled.json").read_text())["estimators"]
    sband = _po[{"RepClaude": "claude_a", "RepCodex": "openai_a", "ScriptEst": "script"}[name]]["band_scored"]
    extra[f"{name}Above"] = fx(100 * np.mean(v > sband), 0) if len(v) else "??"
    extra[f"{name}Below"] = fx(100 * np.mean(v < -sband), 0) if len(v) else "??"

# paired comparisons (scripts/paired_readout.py: held-out stays, models fixed; scripts/paired_bootstrap.py: the
# estimators on the same resampled training records), every one judged against the band of a tenth of the effect
def sg(x, d=1):
    q = half_up(x, d)
    return f"{q:.{d}f}" if q == 0 else f"{q:+.{d}f}".replace("-", "$-$").replace("+", "$+$")


pr = json.loads((R / "paired_readout.json").read_text())["pairs"]
for key, mk in (("Claude, weight access - script, scripted estimate", "PairClaudeScript"),
                ("Codex, weight access - script, scripted estimate", "PairCodexScript"),
                ("Claude, weight access - script, Claude's estimate", "PairClaudeScriptC")):
    d = pr[key]
    extra[mk], extra[mk + "Lo"], extra[mk + "Hi"] = (sg(x, 2) for x in d["difference_fraction"])
    extra[mk + "Abs"] = fx(abs(d['difference_fraction'][0]), 2)
    extra[mk + "Mg"], extra[mk + "MgLo"], extra[mk + "MgHi"] = (sg(x, 1) for x in d["difference_mg_dl"])
pb = json.loads((R / "paired_bootstrap_summary.json").read_text())["worlds"]
for w, wk in (("e03", "Main"), ("n03", "Null")):
    for key, mk in (("claude - script_tuning", "EstClaudeScript"), ("codex - script_train", "EstCodexScript")):
        d = pb[w][key]
        extra[mk + wk], extra[mk + wk + "Lo"], extra[mk + wk + "Hi"] = sg(d["point"]), sg(d["interval"][0]), sg(d["interval"][1])
    extra["EstCopies" + wk] = str(pb[w]["copies"])
# the fixed scripted estimator and the agents' estimators on five fresh no-effect worlds (scripts/fresh_null_seeds.py)
fs = json.loads((R / "fresh_null_seeds.json").read_text())
fa = json.loads((R / "fresh_null_seeds_agents.json").read_text())
# full precision from the draws file (results/fresh_null_seeds.json stores three decimals; one rounding rule only)
_dr = json.loads((R / "fresh_null_seeds_draws.json").read_text())
pts = [_dr[s][0] for s in fs["seeds"]]
_iv = {s: np.percentile(np.array(_dr[s][1:]), [2.5, 97.5]) for s in fs["seeds"]}
extra["FreshSeeds"] = str(len(pts))
extra["FreshDraws"] = str(fs["draws_per_seed"])
extra["FreshScriptPoints"] = ", ".join(sg(x) for x in pts[:-1]) + " and " + sg(pts[-1])
extra["FreshScriptIn"] = str(sum(abs(x) <= fs["band_mg_dl"] for x in pts))
extra["FreshScriptIntervalsIn"] = str(sum(v["verdict"] == "within the band" for v in fs["seeds"].values()))
extra["FreshScriptLoMin"] = sg(min(v[0] for v in _iv.values()))
extra["FreshScriptHiMax"] = sg(max(v[1] for v in _iv.values()))
cl = [v["claude_a"]["implied_insulin_x2"] for v in fa["seeds"].values()]
cx = [v["openai_a"]["implied_insulin_x2"] for v in fa["seeds"].values()]
extra["FreshClaudeMin"], extra["FreshClaudeMax"] = sg(min(cl)), sg(max(cl))
extra["FreshClaudeBeyond"] = str(sum(abs(x) > fs["band_mg_dl"] for x in cl))
extra["FreshCodexIn"] = str(sum(abs(x) <= fs["band_mg_dl"] for x in cx))
extra["FreshCodexZero"] = str(sum(half_up(x, 2) == 0 for x in cx))
extra["FreshCodexMin"] = sg(min(cx))
# pooled no-effect bias of each estimator over the six no-effect worlds, and each fresh world's stay bootstrap
# (scripts/fresh_null_bootstrap.py, scripts/no_effect_pooled.py)
pl_path = R / "no_effect_pooled.json"
if pl_path.is_file():
    po = json.loads(pl_path.read_text())
    word = {"within": "within the band", "beyond": "beyond the band", "straddles": "straddling the band's edge"}
    for est, mk in (("claude_a", "Claude"), ("openai_a", "Codex"), ("script", "Script"), ("claude_d", "RunClaudeB"),
                    ("claude_e", "RunClaudeC"), ("openai_d", "RunCodexB"), ("openai_e", "RunCodexC")):
        if est not in po["estimators"]:              # the repeats' estimators, once their audit is complete
            continue
        e = po["estimators"][est]
        for scope, sk in (("pooled_all_six", "Six"), ("pooled_fresh_five", "Five")):
            d = e[scope]
            extra[f"Pool{mk}{sk}"], extra[f"Pool{mk}{sk}Lo"], extra[f"Pool{mk}{sk}Hi"] = sg(d["mean"], 2), *(sg(x, 2) for x in d["interval"])
            extra[f"Pool{mk}{sk}Verdict"] = word[d["verdict"]]
            extra[f"Pool{mk}{sk}TwoLo"], extra[f"Pool{mk}{sk}TwoHi"] = (sg(x, 2) for x in d["two_stage_interval"])
            extra[f"Pool{mk}{sk}TwoVerdict"] = word[d["two_stage_verdict"]]
            extra[f"Pool{mk}{sk}Between"], extra[f"Pool{mk}{sk}Within"] = (fx(d[k], 2) for k in ("between_world_sd_of_points", "within_world_bootstrap_sd"))
            extra[f"Pool{mk}{sk}Positive"] = str(d["points_positive"])
        pw = e["per_world"]
        extra[f"Pool{mk}MinPoint"], extra[f"Pool{mk}MaxPoint"] = sg(min(v["point"] for v in pw.values())), sg(max(v["point"] for v in pw.values()))
        extra[f"Pool{mk}Copies"] = str(min(v["copies"] for k, v in pw.items() if k != "original"))
        for k, v in pw.items():
            sk = "O" if k == "original" else "ABCDE"[int(k) - 4101]      # seeds 4101..4105 (macro names take no digits)
            extra[f"Fresh{mk}Lo{sk}"], extra[f"Fresh{mk}Hi{sk}"] = (sg(x, 2) for x in v["interval"])
            extra[f"Fresh{mk}Pt{sk}"] = sg(v["point"], 2)
        extra[f"Pool{mk}WorldsBeyond"] = str(sum(v["verdict"] == "beyond" for v in pw.values()))
        extra[f"Pool{mk}WorldsStraddle"] = str(sum(v["verdict"] == "straddles" for v in pw.values()))
        extra[f"Pool{mk}WorldsWithin"] = str(sum(v["verdict"] == "within" for v in pw.values()))
        extra[f"Pool{mk}ExclZero"] = str(e["excluding_zero"])
        # % of the true main-world effect on the stays the estimator averages over
        extra[f"Pool{mk}Share"] = fx(100 * abs(e['pooled_all_six']['mean']) / (10 * e["band_scored"]), 0)
        extra[f"Band{mk}Scored"] = fx(e["band_scored"], 2)
        extra[f"Main{mk}PtsBeyond"] = str(e["main_points_beyond_band"])          # of the six main worlds' points
        extra[f"Main{mk}PtsWithin"] = str(6 - e["main_points_beyond_band"])
        if "main_pooled" in e:          # error in the main worlds: value minus the true shift on the same stays
            d = e["main_pooled"]
            extra[f"Main{mk}"], extra[f"Main{mk}Lo"], extra[f"Main{mk}Hi"] = sg(d["mean"], 2), *(sg(x, 2) for x in d["interval"])
            extra[f"Main{mk}Verdict"] = word[d["verdict"]]
            extra[f"Main{mk}TwoLo"], extra[f"Main{mk}TwoHi"] = (sg(x, 2) for x in d["two_stage_interval"])
            extra[f"Main{mk}Share"] = fx(100 * abs(d['mean']) / (10 * e["band_scored"]), 0)
            extra[f"Main{mk}MinPoint"], extra[f"Main{mk}MaxPoint"] = sg(min(d["points"]), 2), sg(max(d["points"]), 2)
            for w, v in e["main_per_world"].items():
                L = "O" if w == "e03" else "ABCDE"[int(w[1:]) - 5101]          # e03, e5101..e5105
                extra[f"Main{mk}Pt{L}"] = sg(v["error"], 2)
                extra[f"Main{mk}Lo{L}"], extra[f"Main{mk}Hi{L}"] = (sg(x, 2) for x in v["interval"])
            if est in ("claude_a", "openai_a", "script"):          # the first runs' audit (the repeats: Pool*Copies)
                extra["MainCopies"] = str(min(v["copies"] for v in e["main_per_world"].values()))
        extra[f"Fresh{mk}WithinIntervals"] = str(sum(v["verdict"] == "within" for k, v in pw.items() if k != "original"))
        extra[f"Fresh{mk}BeyondIntervals"] = str(sum(v["verdict"] == "beyond" for k, v in pw.items() if k != "original"))
        if "main_pooled" in e:
            extra[f"Main{mk}TwoVerdict"] = word[e["main_pooled"]["two_stage_verdict"]]
    # over every main-world estimator audited (the agents' runs, and the full script)
    AG = ("claude_a", "claude_d", "claude_e", "openai_a", "openai_d", "openai_e")
    _ag = [po["estimators"][k] for k in AG if k in po["estimators"]]
    _all = _ag + [po["estimators"]["script"]]
    extra["EstAudited"] = str(len(_ag))
    extra["EstNullExclZero"] = str(sum((e["pooled_all_six"]["interval"][0] > 0 or e["pooled_all_six"]["interval"][1] < 0) for e in _ag))
    extra["EstNullPointBeyond"] = str(sum(abs(e["pooled_all_six"]["mean"]) > e["band_scored"] for e in _ag))
    extra["EstNullBeyond"] = str(sum(e["pooled_all_six"]["verdict"] == "beyond" for e in _ag))
    extra["EstNullBeyondTwo"] = str(sum(e["pooled_all_six"]["two_stage_verdict"] == "beyond" for e in _ag))
    extra["EstMainBeyond"] = str(sum(e["main_pooled"]["verdict"] == "beyond" for e in _all))
    extra["EstMainErrMin"] = sg(min(e["main_pooled"]["mean"] for e in _all), 2)
    extra["EstMainErrMax"] = sg(max(e["main_pooled"]["mean"] for e in _all), 2)
    extra["EstMainPtsBeyondMax"] = str(max(e["main_points_beyond_band"] for e in _all))
    # Claude's estimator minus the full script's, paired draw by draw on the same resampled stays
    for kind, pk in (("no_effect", "Null"), ("main", "Main")):
        d = po.get("paired_claude_minus_script", {}).get(kind)
        if d:
            extra[f"Pair{pk}"], extra[f"Pair{pk}Lo"], extra[f"Pair{pk}Hi"] = sg(d["mean"], 2), *(sg(x, 2) for x in d["interval"])
            extra[f"Pair{pk}TwoLo"], extra[f"Pair{pk}TwoHi"] = (sg(x, 2) for x in d["two_stage_interval"])
            extra[f"Pair{pk}Positive"] = str(d["points_positive"])
            extra[f"Pair{pk}MinPoint"], extra[f"Pair{pk}MaxPoint"] = sg(min(d["points"]), 2), sg(max(d["points"]), 2)
# where Claude's estimator gets its no-effect bias (scripts/claude_bias_ablation.py): its runs replayed with its
# tiers cut where the sliding scale steps, or with cut-point readings dropped, on the six no-effect and six main worlds
ab_path = R / "claude_bias_ablation.json"
if ab_path.is_file():
    ab = json.loads(ab_path.read_text())
    assert ab["as_run_reproduces_audit"]
    for v, vk in (("as_run", "AsRun"), ("scale_tiers", "Tiers"), ("drop_cuts", "Cuts"), ("both", "Both")):
        s = ab["summary"][v]
        extra[f"Abl{vk}"] = sg(s["no_effect_mean"], 2)
        extra[f"Abl{vk}Min"], extra[f"Abl{vk}Max"] = sg(s["no_effect_min"], 2), sg(s["no_effect_max"], 2)
        extra[f"Abl{vk}Main"] = sg(s["main_error_mean"], 2)
        pts = [r["eq2"] for r in ab["rows"] if r["variant"] == v and r["world"].startswith("n")]
        extra[f"Abl{vk}In"] = str(sum(abs(x) <= po["estimators"]["claude_a"]["band_scored"] for x in pts))   # tuning-stay band
    assert extra["AblAsRun"] == extra["PoolClaudeSix"], (extra["AblAsRun"], extra["PoolClaudeSix"])
# Claude's own no-effect run: its own estimate of the insulin x2 shift (final_answer.md)
own_n = re.search(r"x2 implies \+([0-9.]+) mg/dL", (ROOT / "agent_runs" / "claude_b" / "final_answer.md").read_text())
extra["ClaudeNOwn"] = "$+$" + own_n.group(1)
# ... and its first sustained-effect regression there (Python run 40: the dose's residual on 10 mg/dL reading bins)
first = re.search(r"implied x2 shift = \+([0-9.]+) \(se", next(
    t.read_text() for t in sorted((ROOT / "agent_runs" / "claude_b" / "turns").glob("t*/prompt.txt"))
    if "OUTPUT of Python run 40 of" in t.read_text()))
extra["ClaudeNFirst"] = sg(float(first.group(1)), 1)
last = re.search(r'LAST = \{"claude_a": (\d+), "openai_a": (\d+)\}', (ROOT / "scripts" / "replay_bootstrap.py").read_text())
extra["ReplayRunsClaude"], extra["ReplayRunsCodex"] = last.groups()     # the estimation runs replayed (scripts/replay_*.py)
# the repeats' estimation runs replayed (scripts/replay_reps_claude.py, scripts/replay_reps_codex.py)
for mod, runs_ in (("replay_reps_claude.py", (("claude_d", "ClaudeB"), ("claude_e", "ClaudeC"))),
                   ("replay_reps_codex.py", (("openai_d", "CodexB"), ("openai_e", "CodexC")))):
    _txt = (ROOT / "scripts" / mod).read_text()
    for r, k in runs_:
        extra[f"ReplayRuns{k}"] = re.search(rf'"{r}": (\d+)', re.search(r"^LAST = \{[^}]*\}", _txt, re.M).group(0)).group(1)
extra["EstRunsMax"] = str(max(int(extra[k]) for k in ("ReplayRunsClaude", "ReplayRunsCodex", "ReplayRunsClaudeB", "ReplayRunsClaudeC",
                                                      "ReplayRunsCodexB", "ReplayRunsCodexC")))
du = json.loads((R / "dose_units.json").read_text())                  # the pathway's dose unit (scripts/dose_units.py)
extra["DoseUnitI"], extra["DoseUnitN"] = fx(du["insulin"]["typical_rate"], 2), fx(du["norepinephrine"]["typical_rate"], 2)
tq = R / "audit_eq2" / "truth.json"
if tq.is_file():                   # the true insulin x2 shift on each main world's tuning and training stay-hours
    for w, v in json.loads(tq.read_text()).items():
        L = "O" if w == "e03" else "ABCDE"[int(w[1:]) - 5101]
        extra[f"MainTruthTune{L}"], extra[f"MainTruthTrain{L}"] = sg(v["tuning"], 2), sg(v["train"], 2)
# world-to-world spread beside the counts; how far the record's doses reach; x3 estimate vs truth
_wi = ROOT / "results" / "world_install"
if (_wi / "m5105_script.json").is_file():
    _ds11 = json.loads((R / "decompose_same_stays.json").read_text())["runs"]
    def _vrd(lo, hi):
        lo, hi = half_up(lo, 2), half_up(hi, 2)
        return "within" if Decimal("0.9") <= lo and hi <= Decimal("1.1") else ("beyond" if hi < Decimal("0.9") or lo > Decimal("1.1") else "straddles")
    for est_, key_, name_ in (("script", "script", "Script"), ("claude_a", "script_claude_estimate", "Claude")):
        rr_ = [_ds11[key_]["held_out"]["paired"]["edit_ratio"]] + [json.loads((_wi / f"m{5100 + i}_{est_}.json").read_text())["held_out"]["readout"] for i in range(1, 6)]
        vv_ = [_vrd(r_[1], r_[2]) for r_ in rr_]
        m(f"Wm{name_}WithinN", str(vv_.count("within"))); m(f"Wm{name_}StraddleN", str(vv_.count("straddles")))
        m(f"Wm{name_}BeyondN", str(vv_.count("beyond")))
    _srs = [_ds11["script"]["held_out"]["edit_ratio"]] + [json.loads((_wi / f"m{5100 + i}_script.json").read_text())["held_out"]["readout"][0] for i in range(1, 6)]
    _sd11 = float(np.std(_srs, ddof=1))
    _cb = [agents[r]["insulin_x2"]["ratio"][0] for r in _runs6 if _verdict(agents[r]["insulin_x2"]) == "beyond"]
    m("IxBeyondSds", num(min((0.9 - x) / _sd11 for x in _cb), 1))
_nb = [agents[r][e]["ratio"][0] for r in _runs6 for e in ("ne_x2", "ne_half") if _verdict(agents[r][e]) == "beyond"]
m("QNBeyondMin", num(min(_nb), 2, min(_nb) < 0)); m("QNBeyondMax", num(max(_nb)))
_dsu = json.loads((R / "dose_support.json").read_text())["all_dosing_tiers"]
m("SupOneFive", num(100 * _dsu["ge_1.5x"], 0)); m("SupTwo", num(100 * _dsu["ge_2x"], 1)); m("SupThree", num(100 * _dsu["ge_3x"], 2))
m("SupZero", num(100 * _dsu["zero_share"], 1))
_dst = json.loads((R / "dose_support.json").read_text())["tiers"]
m("SupRows", " ".join(f"{k.replace('>=', '$\\ge$').replace('<', '$<$').replace('-', '--')} & {v['scale_dose']:g} & {v['dosed']:,} & "
                      + (" & ".join(num(100 * v[c], 2 if c == 'ge_3x' else 1) for c in ('ge_1.5x', 'ge_2x', 'ge_3x', 'zero_share'))
                         if v['scale_dose'] > 0 else "-- & -- & -- & --") + r" \\" for k, v in _dst.items()))
_t3 = json.loads((R / "x3_identification.json").read_text())["runs"]
_t3a = [_t3[r]["implied_over_truth_x3"] for r in ("claude_a", "claude_d", "claude_e", "openai_d", "openai_e")]
m("XiImpMin", num(min(_t3a))); m("XiImpMax", num(max(_t3a))); m("XiImpCodexA", num(_t3["openai_a"]["implied_over_truth_x3"]))
m("XiImpScript", num(_t3["script"]["implied_over_truth_x3"]))
m("XiImpScriptC", num(_t3["script_claude_estimate"]["implied_over_truth_x3"]))
# the same with 95% intervals over stays (results/x3_identification_ci.json), as table rows
_xc = R / "x3_identification_ci.json"
if _xc.is_file():
    _xr = json.loads(_xc.read_text())["runs"]
    _XL = (("claude_a", "Claude 1", "genClaudeWIthree"), ("claude_d", "Claude 2", "genRunClaudeBIthree"),
           ("claude_e", "Claude 3", "genRunClaudeCIthree"), ("openai_a", "Codex 1", "genCodexWIthree"),
           ("openai_d", "Codex 2", "genRunCodexBIthree"), ("openai_e", "Codex 3", "genRunCodexCIthree"),
           ("script", "Full script", "genScriptSIthree"), ("script_claude_estimate", "Script, Claude's estimate", "genScriptCIthree"))
    m("XiRows", " ".join(f"{lab} & {num(_xr[k]['x2'][0])} ({num(_xr[k]['x2'][1])}--{num(_xr[k]['x2'][2])}) & "
                         f"{num(_xr[k]['x3'][0])} ({num(_xr[k]['x3'][1])}--{num(_xr[k]['x3'][2])}) & \\{g} \\\\"
                         for k, lab, g in _XL))
# the submitted insulin edits at the largest swept factor each looser budget admits, held out
# (results/budget_sweep.json)
_bs = R / "budget_sweep.json"
if _bs.is_file():
    _bsb = json.loads(_bs.read_text())["budgets"]
    _sub = sum(_verdict(agents[r]["insulin_x2"]) == "within" for r in _runs6)
    m("BsWithinSub", str(_sub)); m("BsWithinFive", str(_bsb["0.005"]["within"]))
    m("BsWithinSeven", str(_bsb["0.007"]["within"])); m("BsWithinTen", str(_bsb["0.010"]["within"]))
    for b_, bk_ in (("0.005", "Five"), ("0.010", "Ten")):
        for r_, rk_ in (("openai_a", "CodexA"), ("openai_d", "CodexB"), ("openai_e", "CodexC"), ("claude_e", "ClaudeC")):
            x_ = _bsb[b_]["rows"][r_]
            m(f"Bs{rk_}{bk_}", f"{num(x_['ratio'][0])} ({num(x_['ratio'][1])}--{num(x_['ratio'][2])})")
            m(f"Bs{rk_}{bk_}A", num(x_["a"], 2)); m(f"Bs{rk_}{bk_}Nll", num(x_["value_nll_change"], 4))
else:
    for k_ in ("BsWithinSub", "BsWithinFive", "BsWithinSeven", "BsWithinTen"):
        m(k_, "--")
# the told runs: eight runs told that unqueried doses and every head are scored, model seeds 1, 2
_pa = R / "told_runs_summary.json"
if _pa.is_file():
    _ps = json.loads(_pa.read_text())
    _hc = json.loads((R / "told_runs_handcheck.json").read_text())["runs"]
    _PA = {"claude_i": "PaClaudeI", "claude_j": "PaClaudeJ", "claude_k": "PaClaudeK", "claude_l": "PaClaudeL",
           "openai_i": "PaCodexI", "openai_j": "PaCodexJ", "openai_k": "PaCodexK", "openai_l": "PaCodexL"}
    _usd, _ref_pa = {}, {}
    for r_, k_ in _PA.items():
        s_ = json.loads((ROOT / "agent_runs" / r_ / "state.json").read_text())
        v_ = _ps["runs"][r_]
        u_ = v_["unqueried"]
        _usd[r_], _ref_pa[r_] = s_["usd"], s_.get("refusals", 0)
        m(f"{k_}Seed", str(v_["seed"])); m(f"{k_}Usd", fx(s_["usd"], 2)); m(f"{k_}Turns", str(s_["turn"]))
        m(f"{k_}Runs", str(s_["runs"])); m(f"{k_}Passes", str(s_["passes"])); m(f"{k_}Refused", str(s_.get("refusals", 0)))
        ix_ = v_["asked"]["insulin_x2"]
        m(f"{k_}Ix", num(ix_[0])); m(f"{k_}IxLo", num(ix_[1])); m(f"{k_}IxHi", num(ix_[2]))
        for e_, ek_ in (("insulin_x3@12", "Ithree"), ("insulin_x0@12", "Izero"), ("insulin_x1.5@12", "Ionefive"),
                        ("ne_x3@12", "Nthree"), ("ne_x0@12", "Nzero")):
            m(f"{k_}{ek_}", num(u_[e_][0]))
        m(f"{k_}Vnll", num(v_["value_nll_change"], 4, sign=True) if v_["value_nll_change"] else "0")
        t_ = v_["total_nll_change"]
        m(f"{k_}Tot", (f"$+${t_ / 10 ** math.floor(math.log10(t_)):.1f}$\\times10^{{{math.floor(math.log10(t_))}}}$"
                       if t_ > 1000 else num(t_, 2, sign=True) if t_ else "0"))
    _ag = _ps["agents"]
    for a_, ak_ in (("claude", "Claude"), ("openai", "Codex")):
        for e_, ek_ in (("insulin_x3", "Ithree"), ("insulin_x0", "Izero"), ("insulin_x2", "Ix")):
            mu_, lo_, hi_ = _ag[a_][e_]
            m(f"Pa{ak_}{ek_}Mean", num(mu_)); m(f"Pa{ak_}{ek_}Lo", num(lo_)); m(f"Pa{ak_}{ek_}Hi", num(hi_))
        o3_, o0_ = _ag[a_]["seed0_current_brief"]["insulin_x3@12"], _ag[a_]["seed0_current_brief"]["insulin_x0@12"]
        m(f"Pa{ak_}OldIthreeMin", num(min(o3_))); m(f"Pa{ak_}OldIthreeMax", num(max(o3_)))
        m(f"Pa{ak_}OldIzeroMin", num(min(o0_))); m(f"Pa{ak_}OldIzeroMax", num(max(o0_)))
        mv_ = _ag[a_]["moves"]
        m(f"Pa{ak_}MoveThree", num(abs(mv_["x3_toward_1"]))); m(f"Pa{ak_}MoveZero", num(abs(mv_["x0_toward_1"])))
        r3_ = [_ps["runs"][r]["unqueried"]["insulin_x3@12"][0] for r in _PA if r.startswith(a_)]
        m(f"Pa{ak_}IthreeMin", num(min(r3_))); m(f"Pa{ak_}IthreeMax", num(max(r3_)))
        m(f"Pa{ak_}RuleOne", "met" if _ag[a_]["rule1_carries"] else "not met")
        m(f"Pa{ak_}RuleTwo", "met" if _ag[a_]["rule2_not"] else "not met")
    _ix = [_ps["runs"][r]["asked"]["insulin_x2"] for r in _PA]
    _pa_within = sum(lo >= 0.9 and hi <= 1.1 for _, lo, hi in _ix)
    m("PaIxWithin", str(_pa_within))
    m("PaIxBeyond", str(sum(hi < 0.9 or lo > 1.1 for _, lo, hi in _ix)))
    m("PaIxMin", num(min(x[0] for x in _ix))); m("PaIxMax", num(max(x[0] for x in _ix)))
    m("PaIxHiMax", num(max(x[2] for x in _ix)))
    m("PaProbedN", str(sum(v["probed_x0_x3"] for v in _hc.values())))
    m("PaHeadsMeasuredN", str(sum(v["measured_other_heads"] for v in _hc.values())))
    _tot = {r: _ps["runs"][r]["total_nll_change"] for r in _PA}
    _sp = sorted(abs(_tn[k]["change_per_stay"]) for k in ("seed1", "seed2"))   # unedited models of other seeds
    m("PaSeedSpreadLo", num(_sp[0], 1)); m("PaSeedSpreadHi", num(_sp[1], 1))
    m("PaTotOverN", str(sum(t > _sp[1] for t in _tot.values())))
    m("PaTotRestMax", num(max(t for t in _tot.values() if t <= _sp[1]), 2, sign=True))
    m("PaUsdTotal", fx(sum(_usd.values()), 2)); m("PaUsdMin", fx(min(_usd.values()), 2)); m("PaUsdMax", fx(max(_usd.values()), 2))
    m("PaRefusedClaude", str(sum(v for r, v in _ref_pa.items() if r.startswith("claude"))))
    m("PaRefusedClaudeMin", str(min(v for r, v in _ref_pa.items() if r.startswith("claude"))))
    m("PaRefusedClaudeMax", str(max(v for r, v in _ref_pa.items() if r.startswith("claude"))))
    m("PaRefusedCodex", str(sum(v for r, v in _ref_pa.items() if r.startswith("openai"))))
    _g0 = json.loads((R / "generalize.json").read_text())["e03:plain model"]["doses"]
    m("PaPlainZeroIthree", num(_g0["insulin_x3@12"]["ratio"][0])); m("PaPlainZeroIzero", num(_g0["insulin_x0@12"]["ratio"][0]))
    _u1 = _ps["runs"]["claude_i"]["unqueried"]           # claude_i submitted the unedited seed-1 model
    assert _ps["runs"]["claude_i"]["total_nll_change"] == 0 and _ps["runs"]["claude_i"]["value_nll_change"] == 0
    m("PaPlainOneIthree", num(_u1["insulin_x3@12"][0])); m("PaPlainOneIzero", num(_u1["insulin_x0@12"][0]))
    _pss = json.loads((R / "told_runs_plain_seeds.json").read_text())["seeds"]     # unedited models, seeds 0-2
    m("PaPlainTwoIthree", num(_pss["2"]["insulin_x3@12"][0])); m("PaPlainTwoIzero", num(_pss["2"]["insulin_x0@12"][0]))
    for sd_, sk_ in (("0", "Zero"), ("1", "One"), ("2", "Two")):
        for e_, ek_ in (("insulin_x1.5@12", "Ionefive"), ("ne_x3@12", "Nthree"), ("ne_x0@12", "Nzero")):
            m(f"PaPlain{sk_}{ek_}", num(_pss[sd_][e_][0]))
        _px = ctl("e03", "es_register", int(sd_))["insulin_x2"]["ratio"]     # the controls' unedited readout, seed sd_
        m(f"PaPlain{sk_}Ix", num(_px[0])); m(f"PaPlain{sk_}IxLo", num(_px[1])); m(f"PaPlain{sk_}IxHi", num(_px[2]))
    _gain = [_ps["runs"][r]["unqueried"]["insulin_x3@12"][0] - _pss[str(_ps["runs"][r]["seed"])]["insulin_x3@12"][0]
             for r in _PA]                              # each told edit over its own seed's unedited model, at x3
    _gold = [x - _pss["0"]["insulin_x3@12"][0] for a_ in ("claude", "openai")
             for x in _ag[a_]["seed0_current_brief"]["insulin_x3@12"]]
    m("PaToldGainMin", num(min(_gain))); m("PaToldGainMax", num(max(_gain)))
    m("PaOldGainMin", num(min(_gold))); m("PaOldGainMax", num(max(_gold)))
    _k = _ps["runs"]["claude_k"]                         # claude_k against the unedited seed-1 model (claude_i)
    _kd = [abs(_ps["runs"]["claude_i"]["asked"][q][0] - _k["asked"][q][0]) for q in ("insulin_x2", "insulin_half")]
    _kd += [abs(_u1[e][0] - _k["unqueried"][e][0]) for e in _u1 if e.startswith("insulin")]
    m("PaClaudeKInsDiff", num(max(_kd), 3))
# the further main worlds: one Claude and one Codex run in each further main
# world m5101-m5105, the instruction alternating by world (results/more_worlds_summary.json, scripts/more_worlds_summary.py); each world's
# unedited model and the script installed in it beside its runs. A run not yet scored reads "??".
_mws = R / "more_worlds_summary.json"
if _mws.is_file():
    _cs = json.loads(_mws.read_text())
    _W = {"m5101": "One", "m5102": "Two", "m5103": "Three", "m5104": "Four", "m5105": "Five"}
    _AG = {"claude": "Claude", "openai": "Codex"}
    _MSG = {"message.txt": "main", "message_unqueried.txt": "told"}
    _asg = json.loads((ROOT / "logs" / "more_worlds" / "assignment.json").read_text())["runs"]
    _UQ = (("insulin_x3@12", "Ithree"), ("insulin_x0@12", "Izero"), ("insulin_x1.5@12", "Ionefive"),
           ("ne_x3@12", "Nthree"), ("ne_x0@12", "Nzero"), ("insulin_x2_lab@12", "Ilab"), ("ne_x2_cuff@12", "Ncuff"))
    _AQ = (("insulin_x2", "Ix"), ("insulin_half", "Ih"), ("ne_x2", "Nx"), ("ne_half", "Nh"))
    _seed0_ref_max = max(json.loads((ROOT / "agent_runs" / r / "state.json").read_text()).get("refusals", 0)
                         for r in REP["Claude"])           # Claude's three seed-0 main-world weight-access runs: 9, 15, 2
    _sp1 = max(abs(_tn[k]["change_per_stay"]) for k in ("seed1", "seed2"))         # the unedited models' seed spread
    for w_, wk_ in _W.items():
        ww_ = _cs["worlds"][w_]
        m(f"CoW{wk_}ScriptIx", num(ww_["script_readout"][0]) if ww_["script_readout"] else "??")
        u_ = ww_.get("unedited")
        for q_, qk_ in _AQ:
            m(f"CoW{wk_}Un{qk_}", num(u_["asked"][q_][0]) if u_ else "??")
        m(f"CoW{wk_}UnIxLo", num(u_["asked"]["insulin_x2"][1]) if u_ else "??")
        m(f"CoW{wk_}UnIxHi", num(u_["asked"]["insulin_x2"][2]) if u_ else "??")
        for e_, ek_ in _UQ:
            m(f"CoW{wk_}Un{ek_}", num(u_["unqueried"][e_][0]) if u_ and e_ in u_["unqueried"] else "??")
    _done = _cs["runs"]
    for a_ in _asg:
        r_, k_ = a_["run"], f"Co{_AG[a_['agent']]}{_W[a_['world_root']]}"
        m(f"{k_}Msg", _MSG[a_["message"]])
        v_ = _done.get(r_)
        if v_ is None:
            pz_ = _cs.get("paused", {}).get(r_)               # paused before it submitted (claude_w3)
            for s_ in ("Ix", "IxLo", "IxHi", "Ih", "Nx", "Nh", "Verdict", "Vnll", "Tot", "Turns", "IxGain", "IthreeGain") \
                    + tuple(ek for _, ek in _UQ):
                m(f"{k_}{s_}", "--" if pz_ else "??")
            m(f"{k_}Usd", fx(pz_["usd"], 2) if pz_ else "??"); m(f"{k_}Runs", str(pz_["runs"]) if pz_ else "??")
            m(f"{k_}Passes", str(pz_["passes"]) if pz_ else "??"); m(f"{k_}Refused", str(pz_["refusals"]) if pz_ else "??")
            m(f"{k_}PausedTurn", str(pz_["after_turn"]) if pz_ else "??")
            m(f"{k_}Heavy", "$^\\ddagger$" if pz_ and pz_["refusals"] > _seed0_ref_max else "")
            continue
        for q_, qk_ in _AQ:
            m(f"{k_}{qk_}", num(v_["asked"][q_][0]))
        m(f"{k_}IxLo", num(v_["asked"]["insulin_x2"][1])); m(f"{k_}IxHi", num(v_["asked"]["insulin_x2"][2]))
        m(f"{k_}Verdict", _verdict({"ratio": v_["asked"]["insulin_x2"]}))
        for e_, ek_ in _UQ:
            m(f"{k_}{ek_}", num(v_["unqueried"][e_][0]) if e_ in v_["unqueried"] else "??")
        m(f"{k_}Vnll", num(v_["value_nll_change"], 4, sign=True) if v_["value_nll_change"] else "0")
        t_ = v_["total_nll_change"]
        m(f"{k_}Tot", (f"$+${t_ / 10 ** math.floor(math.log10(t_)):.1f}$\\times10^{{{math.floor(math.log10(t_))}}}$"
                       if t_ > 1000 else num(t_, 2, sign=True) if t_ else "0"))
        m(f"{k_}Usd", fx(v_["usd"], 2)); m(f"{k_}Runs", str(v_["runs"])); m(f"{k_}Passes", str(v_["passes"]))
        m(f"{k_}Refused", str(v_["refusals"])); m(f"{k_}Turns", str(v_["turns"]))
        m(f"{k_}Heavy", "$^\\ddagger$" if v_["refusals"] > _seed0_ref_max else "")
        u_ = _cs["worlds"][v_["world"]]["unedited"]
        m(f"{k_}IxGain", num(v_["asked"]["insulin_x2"][0] - u_["asked"]["insulin_x2"][0], 2, sign=True))
        m(f"{k_}IthreeGain", num(v_["unqueried"]["insulin_x3@12"][0] - u_["unqueried"]["insulin_x3@12"][0], 2, sign=True))
    m("CoSeedZeroRefMax", str(_seed0_ref_max))
    _sc = [v for v in _done.values()]
    m("CoScored", str(len(_sc)))
    _cv = [_verdict({"ratio": v["asked"]["insulin_x2"]}) for v in _sc]
    m("CoIxWithin", str(_cv.count("within"))); m("CoIxStraddles", str(_cv.count("straddles")))
    m("CoIxBeyondBelow", str(sum(c == "beyond" and v["asked"]["insulin_x2"][0] < 1 for c, v in zip(_cv, _sc))))
    m("CoIxBeyondAbove", str(sum(c == "beyond" and v["asked"]["insulin_x2"][0] > 1 for c, v in zip(_cv, _sc))))
    m("CoNoEditN", str(sum(v["no_edit"] for v in _sc)))
    # total NLL beyond the unedited models' seed spread, over the edits outside the main world's six (told and further worlds)
    m("OtherTotOverN", str(sum(t > _sp[1] for t in _tot.values()) + sum(v["total_nll_change"] > _sp1 for v in _sc)))
    m("OtherTotN", str(len(_tot) + len(_sc)))
    _ov = ([(t, _ps["runs"][r]["value_nll_change"]) for r, t in _tot.items() if t > _sp[1]]
           + [(v["total_nll_change"], v["value_nll_change"]) for v in _sc if v["total_nll_change"] > _sp1])
    assert all(vn < 0.005 for _, vn in _ov)          # every one of them within the value-NLL budget (held-out)
    _om = max(t for t, _ in _ov)
    _ot = sorted(t for t, _ in _ov)
    m("OtherTotFourMin", num(_ot[0], 1)); m("OtherTotFourMax", num(_ot[-3], 1)); m("OtherTotSmallN", str(len(_ot) - 2))
    m("OtherTotMax", f"{_om / 10 ** math.floor(math.log10(_om)):.1f}$\\times10^{{{math.floor(math.log10(_om))}}}$")
    # the unasked doses' verdicts over every scored agent run (main world six, told eight, further worlds): x3 and x0
    _six_g = ("e03:Claude, white box", "e03:Claude, white box, run 2", "e03:Claude, white box, run 3",
              "e03:Codex, white box", "e03:Codex, white box, run 2", "e03:Codex, white box, run 3")
    for e_, ek_ in (("insulin_x3@12", "Ithree"), ("insulin_x0@12", "Izero")):
        _all = ([gen[g]["doses"][e_] for g in _six_g] + [{"ratio": _ps["runs"][r]["unqueried"][e_]} for r in _PA]
                + [{"ratio": v["unqueried"][e_]} for v in _sc])
        _va = [_verdict(x) for x in _all]
        m(f"All{ek_}N", str(len(_va))); m(f"All{ek_}WithinN", str(_va.count("within")))
        m(f"All{ek_}OutN", str(len(_va) - _va.count("within")))
        _vc = _va[-len(_sc):]
        m(f"Co{ek_}OutN", str(len(_vc) - _vc.count("within")))
    _pz = _cs.get("paused", {})
    m("CoPausedN", str(len(_pz)))
    # every agent run scored on the main question, over all settings: the six main-world runs, the eight told runs, the further-world runs
    m("AllRunsN", str(len(_runs6) + 8 + len(_sc)))
    m("AllIxWithinN", str([_verdict(agents[r]["insulin_x2"]) for r in _runs6].count("within") + _pa_within
                          + _cv.count("within")))
    if _sc:
        m("CoIxMin", num(min(v["asked"]["insulin_x2"][0] for v in _sc)))
        m("CoIxMax", num(max(v["asked"]["insulin_x2"][0] for v in _sc)))
        m("CoIthreeMax", num(max(v["unqueried"]["insulin_x3@12"][0] for v in _sc)))
        m("CoUsdTotal", fx(sum(v["usd"] for v in _sc), 2))
        m("CoUsdMin", fx(min(v["usd"] for v in _sc), 2)); m("CoUsdMax", fx(max(v["usd"] for v in _sc), 2))
        m("CoTotOverN", str(sum(v["total_nll_change"] > _sp1 for v in _sc)))
        _tm = max(v["total_nll_change"] for v in _sc)
        m("CoTotMax", f"{_tm / 10 ** math.floor(math.log10(_tm)):.1f}$\\times10^{{{math.floor(math.log10(_tm))}}}$"
          if _tm > 1000 else num(_tm, 1))
        m("CoUsdFrozenTotal", fx(sum(v["usd"] for v in _sc) + sum(z["usd"] for z in _pz.values()), 2))
        _ug = [v["asked"]["insulin_x2"][0] - _cs["worlds"][v["world"]]["unedited"]["asked"]["insulin_x2"][0] for v in _sc]
        m("CoIxGainMin", num(min(_ug), 2)); m("CoIxGainMax", num(max(_ug), 2))
    _un = [w_["unedited"]["asked"]["insulin_x2"][0] for w_ in _cs["worlds"].values() if w_.get("unedited")]
    m("CoUnIxMin", num(min(_un))); m("CoUnIxMax", num(max(_un)))
    _scr = [w_["script_readout"][0] for w_ in _cs["worlds"].values() if w_["script_readout"]]
    m("CoScriptIxMin", num(min(_scr))); m("CoScriptIxMax", num(max(_scr)))
    for a_, ak_ in _AG.items():
        g_ = _cs["agents"][a_]
        m(f"Co{ak_}N", str(g_["n"]))
        for w_ in ("within", "straddles", "beyond"):
            m(f"Co{ak_}{w_.capitalize()}", str(g_["verdicts_x2"][w_]))
        for e_, ek_ in (("insulin_x2", "Ix"), ("insulin_x3", "Ithree"), ("insulin_x0", "Izero"), ("total_nll_change", "Tot")):
            mu_, lo_, hi_ = g_[e_]
            d_ = 1 if e_ == "total_nll_change" else 2
            m(f"Co{ak_}{ek_}Mean", num(mu_, d_) if mu_ is not None else "??")
            m(f"Co{ak_}{ek_}Lo", num(lo_, d_) if lo_ is not None else "--")
            m(f"Co{ak_}{ek_}Hi", num(hi_, d_) if hi_ is not None else "--")
        o_ = [x[0] for x in g_["seed0_main_world_x2"]]
        m(f"Co{ak_}OldIxMin", num(min(o_))); m(f"Co{ak_}OldIxMax", num(max(o_)))
        for msg_, mk_ in (("message.txt", "Cur"), ("message_unqueried.txt", "Told")):
            b_ = g_["by_instruction"][msg_]
            m(f"Co{ak_}{mk_}N", str(b_["n"]))
            for e_, ek_ in (("insulin_x2", "Ix"), ("insulin_x3", "Ithree"), ("insulin_x0", "Izero"), ("total_nll_change", "Tot")):
                mu_ = b_[e_][0]
                m(f"Co{ak_}{mk_}{ek_}", (num(mu_, 1 if ek_ == "Tot" else 2) if mu_ is not None else "??"))
            rf_ = [(x["run"], (_done.get(x["run"]) or _pz.get(x["run"]) or {}).get("refusals")) for x in _asg
                   if x["agent"] == a_ and x["message"] == msg_]
            m(f"Co{ak_}{mk_}Refused", ", ".join(str(c) for _, c in rf_ if c is not None) or "??")
    _lg = [v["asked"]["insulin_x2"][0] - v["unqueried"]["insulin_x2_lab@12"][0] for v in _sc]   # finger minus lab, x2
    m("CoLabGapMin", num(min(_lg), 2)); m("CoLabGapMax", num(max(_lg), 2))
    # declined replies (results/refusal_breakdown.json, scripts/refusal_breakdown.py): the further worlds' five Claude runs pooled
    _rb = json.loads((R / "refusal_breakdown.json").read_text())
    _rc = _rb["c1_claude"]
    m("RfTurns", f"{_rc['turns']:,}"); m("RfDeclined", str(_rc["declined"]))
    m("RfNoteShare", str(round(100 * _rc["by_kind"]["note"]["share"]))); m("RfRestartShare", str(round(100 * _rc["by_kind"]["restart"]["share"])))
    m("RfPythonShare", str(round(100 * _rc["by_kind"]["python"]["share"])))
    m("RfOutDeclined", f"{int(half_up(_rc['median_output_tokens']['declined'], 0)):,}")
    m("RfOutAnswered", f"{int(half_up(_rc['median_output_tokens']['answered'], 0)):,}")
    m("RfStreakWThree", str(_rb["runs"]["claude_w3"]["longest_streak"]))
    m("RfStreakStartWThree", str(_rb["runs"]["claude_w3"]["streak_start"]))
    m("RfStreakOtherMax", str(max(v["longest_streak"] for r, v in _rb["runs"].items() if r != "claude_w3")))
    # the rerun of claude_w3 (instruction and restart rule changed), reported apart from the ten
    _rr = _cs.get("rerun", {}).get("claude_w3r")
    k_ = "CoRerun"
    if _rr:
        for q_, qk_ in _AQ:
            m(f"{k_}{qk_}", num(_rr["asked"][q_][0]))
        m(f"{k_}IxLo", num(_rr["asked"]["insulin_x2"][1])); m(f"{k_}IxHi", num(_rr["asked"]["insulin_x2"][2]))
        for e_, ek_ in _UQ:
            m(f"{k_}{ek_}", num(_rr["unqueried"][e_][0]) if e_ in _rr["unqueried"] else "??")
        m(f"{k_}Vnll", num(_rr["value_nll_change"], 4, sign=True) if _rr["value_nll_change"] else "0")
        m(f"{k_}Tot", num(_rr["total_nll_change"], 2, sign=True) if _rr["total_nll_change"] else "0")
        m(f"{k_}Usd", fx(_rr["usd"], 2)); m(f"{k_}Runs", str(_rr["runs"])); m(f"{k_}Passes", str(_rr["passes"]))
        m(f"{k_}Refused", str(_rr["refusals"])); m(f"{k_}Restarts", str(_rr["refusal_restarts"]))
        m(f"{k_}NoEdit", "yes" if _rr["no_edit"] else "no")
        m("RfStreakRerun", str(_rb["runs"]["claude_w3r"]["longest_streak"]))
        _f7 = json.loads((ROOT / "agent_runs" / "claude_w3r" / "requests" / "fwd_7.done.json").read_text())["results"]
        _tv = _f7["edited"]["value_nll_per_value"] - _f7["original"]["value_nll_per_value"]   # its last factual pass
        m("CoRerunTriedVnll", num(_tv, 3, sign=True)); m("CoRerunTriedTimes", str(int(half_up(_tv / 0.005, 0))))
    else:
        for s_ in ("Ix", "IxLo", "IxHi", "Ih", "Nx", "Nh", "Vnll", "Tot", "Usd", "Runs", "Passes", "Refused", "Restarts",
                   "NoEdit") + tuple(ek for _, ek in _UQ):
            m(f"{k_}{s_}", "??")
# every scored submission (results/submissions.json, scripts/submissions.py): the main-text table's
# rows (paper/generated/submissions_rows.tex) and the counts by stratum
_sj = R / "submissions.json"
if _sj.is_file():
    _sb = json.loads(_sj.read_text())["submissions"]
    _lab = {"claude_a": "Claude 1", "claude_d": "Claude 2", "claude_e": "Claude 3", "openai_a": "Codex 1", "openai_d": "Codex 2",
            "openai_e": "Codex 3", "claude_i": "Claude T1", "claude_j": "Claude T2", "claude_k": "Claude T3", "claude_l": "Claude T4",
            "openai_i": "Codex T1", "openai_j": "Codex T2", "openai_k": "Codex T3", "openai_l": "Codex T4"}
    _V = {"within": "\\vIn", "straddles": "\\vEdge", "beyond": "\\vOut", "outside": "\\vOut"}
    def _set(r):
        return f"M{r['seed']}" if r["world"] == "main" else f"W{r['world'][-1]}"
    def _tot(t):
        if t == 0:
            return "0"
        if abs(t) >= 1000:
            e = math.floor(math.log10(abs(t)))
            return f"$+${t / 10 ** e:.1f}$\\times10^{{{e}}}$"
        return num(t, 2, sign=True)
    _rows, _prev = [], None
    _short = []          # the main text's compact table (x2 with interval, x3, x0, total NLL)
    _shortg = {}         # the same rows by group, for the table's two halves (seed-0 runs | told runs and further worlds)
    for r in _sb:
        grp = "main" if r["world"] == "main" and r["instruction"] == "main" else ("told" if r["world"] == "main" else "further")
        if _prev is not None and grp != _prev:
            _rows.append("\\midrule")
        _prev = grp
        lab = _lab.get(r["run"]) or f"{r['agent']}, world {r['world'][-1]}"
        d = r["paired"]["difference"]
        heavy = "$^\\ddagger$" if r["agent"] == "Claude" and r["declined"] > _seed0_ref_max else ""
        null = "$^\\dagger$" if r["insulin_unedited"] else ""
        _W = {"within": "in", "straddles": "edge", "beyond": "out", "outside": "out"}
        if _short and grp != _prevs:
            _short.append("\\midrule")
        _prevs = grp
        _short.append(f"{lab}{null}{heavy} & {_set(r)} & {r['instruction']} & "
                      f"{_V[r['verdict_interval']]}{{{num(r['x2'][0])} ({num(r['x2'][1])}--{num(r['x2'][2])})}} & "
                      f"{num(r['x3'])} & {num(r['x0'])} & {_tot(r['total_nll_change'])} \\\\")
        _shortg.setdefault(grp, []).append(f"{_lab.get(r['run']) or r['agent']}{null}{heavy} & {_set(r)} & {r['instruction']} & "
                      f"{_V[r['verdict_interval']]}{{{num(r['x2'][0])} ({num(r['x2'][1])}--{num(r['x2'][2])})}} & "
                      f"{num(r['x3'])} & {num(r['x0'])} & {_tot(r['total_nll_change'])} \\\\")
        _rows.append(f"{lab}{null} & {_set(r)} & {r['instruction']} & "
                     f"{_V[r['verdict_interval']]}{{{num(r['x2'][0])} ({num(r['x2'][1])}--{num(r['x2'][2])})}} & "
                     f"{_W[r['verdict_point']]} & {_W[r['verdict_interval']]} & {num(r['script_x2'][0])} & "
                     f"{num(d[0], 2, sign=True)} ({num(d[1], 2, sign=True)} to {num(d[2], 2, sign=True)}) & {num(r['x3'])} & "
                     f"{num(r['x0'])} & {_tot(r['total_nll_change'])} & {r['declined']}{heavy} \\\\")
    (ROOT / "paper" / "generated").mkdir(exist_ok=True)
    (ROOT / "paper" / "generated" / "submissions_rows.tex").write_text(
        "% generated by scripts/paper_numbers.py from results/submissions.json; do not edit\n" + "\n".join(_rows)
        + "\n\\bottomrule\n")
    _hd = "% generated by scripts/paper_numbers.py from results/submissions.json; do not edit\n"
    (ROOT / "paper" / "generated" / "submissions_rows_m0.tex").write_text(_hd + "\n".join(_shortg["main"]) + "\n\\bottomrule\n")
    (ROOT / "paper" / "generated" / "submissions_rows_rest.tex").write_text(
        _hd + "\n".join(_shortg["told"]) + "\n\\midrule\n" + "\n".join(_shortg["further"]) + "\n\\bottomrule\n")
    (ROOT / "paper" / "generated" / "submissions_rows_short.tex").write_text(
        "% generated by scripts/paper_numbers.py from results/submissions.json; do not edit\n" + "\n".join(_short)
        + "\n\\bottomrule\n")
    def _cnt(rows, key="verdict_interval", val="within"):
        return sum(r[key] == val for r in rows)
    _ed = [r for r in _sb if not r["insulin_unedited"]]
    _mi = [r for r in _sb if r["instruction"] == "main"]
    _to = [r for r in _sb if r["instruction"] == "told"]
    m("SubN", str(len(_sb))); m("SubWithin", str(_cnt(_sb))); m("SubStraddle", str(_cnt(_sb, val="straddles")))
    m("SubPointWithin", str(_cnt(_sb, "verdict_point")))
    m("SubNullN", str(len(_sb) - len(_ed))); m("SubEditN", str(len(_ed))); m("SubEditWithin", str(_cnt(_ed)))
    m("SubMainN", str(len(_mi))); m("SubMainWithin", str(_cnt(_mi))); m("SubToldN", str(len(_to))); m("SubToldWithin", str(_cnt(_to)))
    for k_, f_ in (("MZero", lambda r: r["world"] == "main" and r["seed"] == 0),
                   ("MOneTwo", lambda r: r["world"] == "main" and r["seed"] in (1, 2)), ("Further", lambda r: r["world"] != "main")):
        g_ = [r for r in _sb if f_(r)]
        m(f"Sub{k_}N", str(len(g_))); m(f"Sub{k_}Within", str(_cnt(g_)))
    _hv = [r for r in _sb if r["agent"] == "Claude" and r["declined"] > _seed0_ref_max]
    m("SubHeavyN", str(len(_hv))); m("SubHeavyWithin", str(_cnt(_hv)))
    m("SubEditThreeShort", str(sum(r["x3"] < 0.9 for r in _ed)))
    _bl = sum(r["paired"]["difference"][2] < 0 for r in _sb); _ab = sum(r["paired"]["difference"][1] > 0 for r in _sb)
    m("SubBelowScript", str(_bl)); m("SubAboveScript", str(_ab)); m("SubLevelScript", str(len(_sb) - _bl - _ab))
    _mz = [r for r in _sb if r["world"] == "main" and r["seed"] == 0]
    m("SubMZeroNotBelow", str(sum(r["paired"]["difference"][2] >= 0 for r in _mz)))
    m("SubNotBelow", str(len(_sb) - _bl))
# the reference confined to same-shape replacement (results/same_shape/, scripts/same_shape.py)
_ssd = R / "same_shape"
if (_ssd / "tune_bdir_script.json").is_file():
    _st = {t: json.loads((_ssd / f"tune_{t}.json").read_text()) for t in ("bdir_script", "bdir_claude", "ncol_script", "ncol_claude")}
    _sh = [r["readouts"]["insulin_x2"] for t in _st.values() for r in t["rounds"]]
    m("SsMaxShift", num(min(_sh), 2)); m("SsRoundsN", str(len(_sh)))
    m("SsTargetScript", num(_st["bdir_script"]["targets"]["insulin_x2"], 1)); m("SsTargetClaude", num(_st["bdir_claude"]["targets"]["insulin_x2"], 1))
    m("SsPlainTuning", num(-1.80, 2))
    _bd = [b["value_nll_change"] for t in ("bdir_script", "bdir_claude") for b in _st[t]["budget_steps"]]
    m("SsBdirCostMin", num(min(_bd), 2)); m("SsBdirFlip", num(max(r["readouts"]["insulin_x2"] for t in ("bdir_script", "bdir_claude")
                                                                 for r in _st[t]["rounds"]), 2, sign=True))
    m("SsNcolGain", num(_st["ncol_script"]["final"]["gains"]["I"], 0))
    _ne = agents.get("ss_ncol_script") or next((json.loads(x) for x in open(R / "agent_eval.jsonl")
                                                  if json.loads(x)["method"] == "agent_ss_ncol_script"), None)
    if _ne:
        m("SsNcolIx", num(_ne["insulin_x2"]["ratio"][0])); m("SsNcolIxLo", num(_ne["insulin_x2"]["ratio"][1]))
        m("SsNcolIxHi", num(_ne["insulin_x2"]["ratio"][2])); m("SsNcolVnll", num(_ne["value_nll_change"], 4, sign=True))
    _np = next((json.loads(x) for x in open(R / "model_parts.jsonl") if json.loads(x)["label"] == "ss_ncol_script"), None)
    if _np:
        m("SsNcolTot", num(_np["total_nll_per_stay"]["change"], 2, sign=True))
    sys.path[:0] = [str(ROOT / "scripts")]
    _sg = json.loads((_ssd / "ncol_unasked.json").read_text()) if (_ssd / "ncol_unasked.json").is_file() else {}
    for e_, k_ in (("insulin_x3", "Ithree"), ("insulin_x0", "Izero"), ("insulin_x1.5", "Ionefive")):
        m(f"SsNcol{k_}", num(_sg[e_][0]) if e_ in _sg else "??")
# by-hour verdicts, the heavy-decline timing, and the counts without the marked runs
_bh = R / "byhour_submissions.json"
if _bh.is_file() and _sj.is_file():
    _bhd = json.loads(_bh.read_text())
    _bv = list(_bhd["submissions"].values()) + list(_bhd["scripts"].values())
    m("BhBothWithinN", str(sum(v["both_within"] for v in _bv))); m("BhModelsN", str(len(_bv)))
    m("BhEarlyBelowLateN", str(sum(v["early"][0] < v["late"][0] for v in _bv)))
    _bsc = list(_bhd["scripts"].values())
    m("BhScriptEarlyMin", num(min(v["early"][0] for v in _bsc))); m("BhScriptEarlyMax", num(max(v["early"][0] for v in _bsc)))
    m("BhScriptLateMin", num(min(v["late"][0] for v in _bsc))); m("BhScriptLateMax", num(max(v["late"][0] for v in _bsc)))
    _apx = json.loads((R / "appendix_checks_offline.json").read_text())["C3"]
    m("HvBeforeEditN", str(sum(v["declines_began_before_first_tested_edit"] for v in _apx.values()))); m("HvN", str(len(_apx)))
    # without the marked runs (Claude's runs outside seed 0 that left insulin unedited or carry many declines)
    _mk = [r for r in _sb if r["agent"] == "Claude" and not (r["world"] == "main" and r["seed"] == 0)
           and (r["insulin_unedited"] or r["declined"] > _seed0_ref_max)]
    _um = [r for r in _sb if r not in _mk]
    m("SubUnmarkedN", str(len(_um))); m("SubUnmarkedWithin", str(_cnt(_um)))
    m("SubUnmarkedBelow", str(sum(r["paired"]["difference"][2] < 0 for r in _um))); m("SubMarkedN", str(len(_mk)))
    _uh = [r for r in _sb if not (r["agent"] == "Claude" and r["declined"] > _seed0_ref_max)]
    m("SubUnheavyN", str(len(_uh))); m("SubUnheavyBelow", str(sum(r["paired"]["difference"][2] < 0 for r in _uh)))
    # the six beyond the seed spread are submissions on the 17 other trained models (seeds 1-2 and further worlds)
    m("OtherModelsN", str(len([r for r in _sb if not (r["world"] == "main" and r["seed"] == 0)])))
# Table 14's facts for the main text: below the top dose bin, the scripts against the six agent edits
_xsv = {}
for _l in out:
    _mm = re.match(r"\\newcommand\{\\Xs(ClaudeA|ClaudeB|ClaudeC|CodexA|CodexB|CodexC|ScriptS|ScriptC)Below\}\{([-0-9.]+)\}", _l)
    if _mm:
        _xsv[_mm.group(1)] = float(_mm.group(2))
if len(_xsv) == 8:
    _ag = sorted(_xsv[k] for k in ("ClaudeA", "ClaudeB", "ClaudeC", "CodexA", "CodexB", "CodexC"))
    _sc = sorted(_xsv[k] for k in ("ScriptS", "ScriptC"))
    m("XsAgentsAboveScriptsN", str(len([x for x in _ag if x > _sc[1]])))
# the in-band runs' own writes rescaled by one gain to a given estimate; the full script beyond seed 0
# (results/appendix_checks_readouts.json, scripts/appendix_checks_readouts.py)
_apxr = R / "appendix_checks_readouts.json"
if _apxr.is_file():
    _b = json.loads(_apxr.read_text())
    _MW = {"ss_mclaude_script": "MwClaudeS", "ss_mclaude_claude": "MwClaudeC", "ss_mcodex_script": "MwCodexS", "ss_mcodex_claude": "MwCodexC"}
    for t_, k_ in _MW.items():
        x_ = _b["b1"][t_]
        m(f"{k_}Gain", num(x_["gain"], 3)); m(f"{k_}FitGain", num(x_["fitted_gain"], 3))
        for q_, r_ in (("Ix", x_["asked"]["insulin_x2"]), ("Ithree", x_["unasked"]["insulin_x3"])):
            m(f"{k_}{q_}", num(r_[0])); m(f"{k_}{q_}Lo", num(r_[1])); m(f"{k_}{q_}Hi", num(r_[2]))
            m(f"{k_}{q_}V", _verdict({"ratio": r_}))
        m(f"{k_}Izero", num(x_["unasked"]["insulin_x0"][0])); m(f"{k_}Vnll", num(x_["value_nll_change"], 4, sign=True))
    _same = [t_ for t_, x_ in _b["b1"].items() if abs(x_["gain"] - 1) <= 0.004]
    m("MwSameN", str(len(_same))); m("MwN", str(len(_b["b1"])))
    m("MwSameIthreeMin", num(min(_b["b1"][t]["unasked"]["insulin_x3"][0] for t in _same)))
    m("MwSameIthreeMax", num(max(_b["b1"][t]["unasked"]["insulin_x3"][0] for t in _same)))
    m("MwBothWithinN", str(sum(_verdict({"ratio": x["asked"]["insulin_x2"]}) == "within" and
                               _verdict({"ratio": x["unasked"]["insulin_x3"]}) == "within" for x in _b["b1"].values())))
    _nr = _b["b1"]["ss_mclaude_script"]["tuning_rounds"]          # Claude's write: the tuning readout against the gain
    _nr = sorted(set((round(g, 3), r) for g, r in _nr))
    m("MwNonlin", ", ".join(f"{num(g, 3)} to {num(r, 2)}" for g, r in sorted(_nr, reverse=True)))
    _s3 = [gen["e03:scripted install, scripted estimate"]["doses"]["insulin_x3@12"]["ratio"]] + \
          [v_["insulin_x3"] for v_ in _b["b2"].values()]
    _s3v = [_verdict({"ratio": r}) for r in _s3]
    m("ScThreeN", str(len(_s3))); m("ScThreeWithin", str(_s3v.count("within"))); m("ScThreeStraddle", str(_s3v.count("straddles")))
    m("ScThreeShort", str(_s3v.count("beyond"))); m("ScThreeMin", num(min(r[0] for r in _s3))); m("ScThreeMax", num(max(r[0] for r in _s3)))
    _s0 = [gen["e03:scripted install, scripted estimate"]["doses"]["insulin_x0@12"]["ratio"]] + [v_["insulin_x0"] for v_ in _b["b2"].values()]
    m("ScZeroMax", num(max(r[0] for r in _s0))); m("ScZeroWithin", str([_verdict({"ratio": r}) for r in _s0].count("within")))
for k, v in extra.items():
    m(k, v)
# unsigned copies of the two largest total-NLL rises, for "rises by"
_mv = dict(re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", x).groups() for x in out)
for k_ in ("CoCodexFourTot", "PaClaudeLTot"):
    assert _mv[k_].startswith("$+$"); m(f"{k_}Abs", _mv[k_][3:])
dest =Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "numbers.tex"     # default: the standalone paper's
dest.write_text("% generated by scripts/paper_numbers.py; do not edit\n" + "\n".join(out) + "\n")
print(len(out), "macros")
