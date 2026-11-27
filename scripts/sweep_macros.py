# Macros from results/strength_sweep.json, run inside scripts/paper_numbers.py (exec; uses m, num, json, ROOT).
# The sentences follow the reading rule fixed before the sweep: full-strength value-NLL cost above the
# 0.005 budget -> the budget binds this form of edit; at or below -> the shortfall is not explained by the budget.
_sw = json.loads((ROOT / "results" / "strength_sweep.json").read_text())
_B = _sw["budget"]
SWK = {"codexA_ins": "CodexAIns", "codexB_ins": "CodexBIns", "codexC_ins": "CodexCIns",
       "claudeB_ne": "ClaudeBNe", "codexB_ne": "CodexBNe", "codexC_ne": "CodexCNe", "claudeC_ins": "ClaudeCIns"}
_full = {}
for t_, k in SWK.items():
    v = _sw["targets"].get(t_)
    if not v or "full_strength" not in v:
        for q in ("Cost", "A", "Est", "Read", "Within", "CostOne"):
            m(f"Sw{k}{q}", "??")
        continue
    fs = v["full_strength"]
    _full[t_] = fs
    m(f"Sw{k}Cost", num(fs["cost"], 4)); m(f"Sw{k}A", num(fs["a"], 1)); m(f"Sw{k}Est", num(v["estimate"], 2, True))
    m(f"Sw{k}Read", num(fs["readout"], 2, True)); m(f"Sw{k}Within", "yes" if fs["within_tol"] else "no")
    p1 = next(p for p in v["points"] if p["a"] == 1.0)
    m(f"Sw{k}CostOne", num(p1["cost"], 4))
_ins = [t_ for t_ in ("codexA_ins", "codexB_ins", "codexC_ins") if t_ in _full]
_ne = [t_ for t_ in ("claudeB_ne", "codexB_ne", "codexC_ne") if t_ in _full]
_ci = [_full[t_]["cost"] for t_ in _ins]
_cn = [_full[t_]["cost"] for t_ in _ne]
if _ci:
    m("SwInsCostMin", num(min(_ci), 4)); m("SwInsCostMax", num(max(_ci), 4))
    m("SwInsAbove", str(sum(c > _B for c in _ci))); m("SwInsN", str(len(_ci)))
if _cn:
    m("SwNeCostMin", num(min(_cn), 3)); m("SwNeCostMax", num(max(_cn), 2))
    m("SwNeAbove", str(sum(c > _B for c in _cn))); m("SwNeN", str(len(_cn)))
_ia = sum(c > _B for c in _ci) if _ci else None
if _ia is not None and len(_ci) == 3 and _ia == 3:
    m("SweepAbstract", ", which their form of edit reaches only at a fit cost above the budget the agents were given")
    m("SweepIntro", " Scaled to their own estimates, Codex's insulin edits would cost \\SwInsCostMin--\\SwInsCostMax{} nats of "
      "value NLL per reading, above the 0.005 budget (\\cref{sec:null}).")
    m("SweepResults", " Scaled up until their tuning readout equals their own estimate, Codex's three insulin edits cost "
      "\\SwInsCostMin--\\SwInsCostMax{} nats of value NLL per reading, above the budget of 0.005; Claude's first edit and the "
      "scripts deliver about 1.0 at no cost, so this form of edit, not the budget alone, held Codex back (\\cref{app:sweep}).")
    m("SweepConclusion", " Codex's edits could not be installed in full within the fit budget in the form it chose.")
elif _ia is not None and len(_ci) == 3 and _ia == 0:
    m("SweepAbstract", ", a shortfall the fit budget does not explain")
    m("SweepIntro", " Scaled to their own estimates, Codex's insulin edits would cost only \\SwInsCostMin--\\SwInsCostMax{} nats "
      "of value NLL per reading, within the 0.005 budget (\\cref{sec:null}).")
    m("SweepResults", " Scaled up until their tuning readout equals their own estimate, Codex's three insulin edits cost "
      "\\SwInsCostMin--\\SwInsCostMax{} nats of value NLL per reading, within the budget of 0.005: the budget does not explain "
      "the shortfall (\\cref{app:sweep}).")
    m("SweepConclusion", " Codex's edits stopped short of their own estimates although full strength fitted the budget.")
elif _ia is not None:
    _ord = {"codexA_ins": "first", "codexB_ins": "second", "codexC_ins": "third"}
    _over = [t_ for t_ in _ins if _full[t_]["cost"] > _B]
    _under = [t_ for t_ in _ins if _full[t_]["cost"] <= _B]
    _w = {1: "one", 2: "two"}
    _lst = lambda ts: " and ".join(_ord[t_] for t_ in ts)
    _cst = lambda ts: " and ".join(f"\\Sw{SWK[t_]}Cost{{}}" for t_ in ts)
    m("SweepAbstract", f", {_w[len(_over)]} of which would exceed the fit budget at full strength")
    m("SweepIntro", f" At full strength Codex's {_lst(_over)} insulin edits would exceed the 0.005-nat fit budget, its "
      f"{_lst(_under)} would not (\\cref{{sec:null}}).")
    m("SweepResults", f" Scaled up until their tuning readout equals their own estimate, Codex's {_lst(_over)} insulin edits "
      f"cost {_cst(_over)} nats of value NLL per reading, above the 0.005 budget, and its {_lst(_under)} {_cst(_under)}, "
      f"within it (\\cref{{app:sweep}}); Claude's first edit and the scripts deliver about 1.0 at no cost.")
    m("SweepConclusion", f" At full strength {_w[len(_over)]} of Codex's three edits would have exceeded the fit budget in "
      f"the form it chose.")
else:
    m("SweepAbstract", ""); m("SweepIntro", ""); m("SweepResults", ""); m("SweepConclusion", "")
# norepinephrine: the weakened edits at full strength (or at the closest point the search reached)
if _cn:
    _nr = [t_ for t_ in _ne if not _full[t_]["within_tol"]]
    _ne_txt = (" The \\SwNeN{} weakened norepinephrine edits would cost \\SwNeCostMin--\\SwNeCostMax{} nats per reading "
               "at the readout closest to their estimate" + (" (\\SwNeNotReached{} never reach it)" if _nr else "") +
               (", all above the budget." if all(c > _B for c in _cn) else ", not all above the budget."))
    m("SweepResultsNe", _ne_txt)
    m("SwNeNotReached", str(len(_nr)))
else:
    m("SweepResultsNe", "")
# side measure: Codex's insulin edits at full strength, held-out x3 and x0 (results/sweep_general.json)
_sgp = ROOT / "results" / "sweep_general.json"
if _sgp.is_file():
    _sg = json.loads(_sgp.read_text())
    for t_, k in (("codexA_ins", "CodexA"), ("codexB_ins", "CodexB"), ("codexC_ins", "CodexC")):
        for e, ek in (("insulin_x3", "Three"), ("insulin_x0", "Zero")):
            m(f"Sg{k}{ek}", num(_sg[t_][e][0])); m(f"Sg{k}{ek}Lo", num(_sg[t_][e][1])); m(f"Sg{k}{ek}Hi", num(_sg[t_][e][2]))
    m("SgThreeMin", num(min(v["insulin_x3"][0] for v in _sg.values()))); m("SgThreeMax", num(max(v["insulin_x3"][0] for v in _sg.values())))
    m("SgThreeHiMax", num(max(v["insulin_x3"][2] for v in _sg.values())))
    m("SgZeroMin", num(min(v["insulin_x0"][0] for v in _sg.values()))); m("SgZeroMax", num(max(v["insulin_x0"][0] for v in _sg.values())))

# Claude 3's insulin edit at full strength
_c3 = _sw["targets"].get("claudeC_ins")
if _c3 and "full_strength" in _c3:
    m("SweepClaudeC", " Claude's third, held below its estimate for the budget by its answer, would cost "
      "\\SwClaudeCInsCost" + (", within it." if _c3["full_strength"]["cost"] <= _B else ", above it."))
else:
    m("SweepClaudeC", "")
