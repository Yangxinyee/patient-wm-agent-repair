# Macros from results/world_install/<world>_<estimate>.json, run inside scripts/paper_numbers.py
# (exec; uses m, num, json, np, ROOT). Reading rules fixed before the runs.
_WD = ROOT / "results" / "world_install"
_wr = {f.stem: json.loads(f.read_text()) for f in sorted(_WD.glob("*.json"))} if _WD.is_dir() else {}
RULER = 1.4                                          # the ruler fixed before the runs (mg/dL)
_NULLW = ("n03", "fA", "fB")
_WK = {"n03": "Orig", "fA": "FA", "fB": "FB", "e03": "Main"}
_WK.update({f"m{5100 + i}": f"M{'ABCDE'[i - 1]}" for i in range(1, 6)})
_EK = {"claude_a": "Claude", "script": "Script"}
for key_, r_ in _wr.items():
    w_, e_ = key_.rsplit("_", 1) if not key_.endswith("claude_a") else (key_[: -len("_claude_a")], "claude_a")
    k_ = f"Wi{_WK[w_]}{_EK[e_]}"
    h_ = r_["held_out"]
    m(f"{k_}Shift", num(h_["shift"][0], 2, True)); m(f"{k_}Lo", num(h_["shift"][1], 2, True)); m(f"{k_}Hi", num(h_["shift"][2], 2, True))
    m(f"{k_}Imp", num(h_["implied"], 2, True))
    if "readout" in h_:
        m(f"{k_}R", num(h_["readout"][0])); m(f"{k_}RLo", num(h_["readout"][1])); m(f"{k_}RHi", num(h_["readout"][2]))
        m(f"{k_}E", num(h_["estimate_ratio"][0])); m(f"{k_}D", num(h_["residual"][0], 2, True))
# installs not yet finished: a dash in the appendix tables
_done = {(_WK[k.rsplit("_", 1)[0] if not k.endswith("claude_a") else k[: -len("_claude_a")]],
          _EK["claude_a" if k.endswith("claude_a") else "script"]) for k in _wr}
for w_, wk_ in _WK.items():
    for ek_ in ("Claude", "Script"):
        if (wk_, ek_) not in _done:
            for f_ in ("Shift", "Lo", "Hi", "Imp", "R", "RLo", "RHi", "E", "D"):
                m(f"Wi{wk_}{ek_}{f_}", "--")
# the original main world: the full script's edit and the script with Claude's estimate (results/decompose_same_stays.json)
_ds = _ds0 = json.loads((ROOT / "results" / "decompose_same_stays.json").read_text())["runs"]
for r_, ek_ in (("script", "Script"), ("script_claude_estimate", "Claude")):
    pr_ = _ds[r_]["held_out"]["paired"]
    m(f"WiRef{ek_}R", num(pr_["edit_ratio"][0])); m(f"WiRef{ek_}RLo", num(pr_["edit_ratio"][1]))
    m(f"WiRef{ek_}RHi", num(pr_["edit_ratio"][2])); m(f"WiRef{ek_}E", num(pr_["estimate_ratio"][0]))
    m(f"WiRef{ek_}D", num(pr_["edit_ratio"][0] - pr_["estimate_ratio"][0], 2, True))
# Claude run 1's estimate installed where the drugs are inert
_ci = [_wr[f"{w}_claude_a"]["held_out"]["shift"][0] for w in _NULLW if f"{w}_claude_a" in _wr]
_si = [_wr[f"{w}_script"]["held_out"]["shift"][0] for w in _NULLW if f"{w}_script" in _wr]
if _ci:
    m("WiClaudeNullMin", num(min(_ci), 2, True)); m("WiClaudeNullMax", num(max(_ci), 2, True)); m("WiClaudeNullN", str(len(_ci)))
    m("WiClaudeNullAbove", str(sum(abs(x) > RULER for x in _ci)))
if _si:
    m("WiScriptNullMin", num(min(_si), 2, True)); m("WiScriptNullMax", num(max(_si), 2, True))
if len(_ci) == 3:
    _ab = sum(abs(x) > RULER for x in _ci)
    _nbq = json.loads((ROOT / "results" / "null_seed_batch.json").read_text())["estimators"]["claude_a"]
    _oth = _nbq["five"] + _nbq["ten"]                       # the inert worlds outside the first six (no trained models)
    m("WiOtherN", str(len(_oth))); m("WiOtherBelow", str(sum(abs(x) <= RULER for x in _oth)))
    m("WiOtherMin", num(min(_oth), 2, True)); m("WiOtherMax", num(max(_oth), 2, True))
    _txt = (" Installed by the script's procedure in the three inert worlds with trained models, all from the first six, "
            "it gives the inert drug \\WiClaudeNullMin{} to \\WiClaudeNullMax{} mg/dL, ")
    _txt += ("beyond our 1.4 mg/dL ruler in all three" if _ab == 3 else
             "within our 1.4 mg/dL ruler in all three" if _ab == 0 else f"beyond our 1.4 mg/dL ruler in {_ab} of the three")
    _txt += ("; replayed in the other \\WiOtherN{} worlds it reads \\WiOtherMin{} to \\WiOtherMax, within that ruler in "
             "\\WiOtherBelow, and the script's estimate installed gives \\WiScriptNullMin{} to \\WiScriptNullMax{} (\\cref{app:install}).")
    m("BiResult", _txt)
else:
    m("BiResult", "")
# the five further main worlds and the original
_MW = [f"m{5100 + i}" for i in range(1, 6)]
_sr = {w: _wr[f"{w}_script"]["held_out"] for w in _MW if f"{w}_script" in _wr}
if len(_sr) == 5:
    _orig = json.loads((ROOT / "results" / "decompose_same_stays.json").read_text())["runs"]["script"]["held_out"]
    rs = [_orig["edit_ratio"]] + [v["readout"][0] for v in _sr.values()]
    m("WmScriptMean", num(float(np.mean(rs)))); m("WmScriptSd", num(float(np.std(rs, ddof=1))))
    m("WmScriptMin", num(min(rs))); m("WmScriptMax", num(max(rs))); m("WmWorlds", str(len(rs)))
    m("WmOrigRank", "the lowest" if rs[0] == min(rs) else ("the highest" if rs[0] == max(rs) else "within the range"))
    ds = [v["residual"][0] for v in _sr.values()] + [_wr[f"{w}_claude_a"]["held_out"]["residual"][0] for w in _MW if f"{w}_claude_a" in _wr]
    m("WmResidMin", num(min(ds), 2, True)); m("WmResidMax", num(max(ds), 2, True)); m("WmResidAbsMax", num(max(abs(d) for d in ds)))
    m("WmRuleHolds", "yes" if all(abs(d) <= 0.05 for d in ds) and len(ds) == 10 else "no")
    m("WmResidOver", str(sum(abs(d) > 0.05 for d in ds)))
# the sentence for section 5.1, by the rule fixed before the runs
if len(_sr) == 5 and len(ds) == 10:
    if all(abs(d) <= 0.05 for d in ds):
        m("WmSentence", " In five further main worlds every installation delivers its estimate to within \\WmResidAbsMax, and the script reads \\WmScriptMin--\\WmScriptMax{} over the six "
          "(mean \\WmScriptMean; \\cref{app:install}).")
        # step 3: each agent estimator's replay in the five worlds as a predicted readout (estimate +- 0.05), with a verdict
        _rp = ROOT / "results" / "replay_profiles.json"
        if _rp.is_file():
            rows = json.loads(_rp.read_text())["rows"]
            def _vp(e):
                lo, hi = e - 0.05, e + 0.05
                return "within" if lo >= 0.9 and hi <= 1.1 else ("beyond" if hi < 0.9 or lo > 1.1 else "straddles")
            vs = [_vp(r["ratio_held_out"]) for r in rows]
            m("RpN", str(len(rows))); m("RpWithin", str(vs.count("within"))); m("RpBeyond", str(vs.count("beyond")))
            m("RpStraddle", str(vs.count("straddles")))
            _lab = {"claude_a": "Claude, run 1", "claude_d": "Claude, run 2", "claude_e": "Claude, run 3",
                    "openai_a": "Codex, run 1", "openai_d": "Codex, run 2", "openai_e": "Codex, run 3"}
            _byw = {(r["run"], r["world"]): r["ratio_held_out"] for r in rows}
            def _cell(x):
                t_ = num(x, 3) if num(x, 3) != "1.000" else num(x, 4)   # a value at 1.000 shows on which side of 1 it lies
                return "\\textbf{" + t_ + "}" if _vp(x) == "within" else t_
            m("RpTableRows", " ".join(
                _lab[r] + " & " + " & ".join(_cell(_byw[(r, f"e{5100 + i}")]) if (r, f"e{5100 + i}") in _byw else "--"
                                            for i in range(1, 6)) + r" \\" for r in _lab))
            _all = [r_["ratio_held_out"] for r_ in rows]
            m("RpAboveOne", str(sum(x > 1 for x in _all))); m("RpMean", num(float(np.mean(_all))))
            _eo = [_ds0[r_]["held_out"]["estimate_ratio"] for r_ in _lab]
            m("RpOrigMin", num(min(_eo))); m("RpOrigMax", num(max(_eo)))
            m("RpMin", num(min(r["ratio_held_out"] for r in rows))); m("RpMax", num(max(r["ratio_held_out"] for r in rows)))
            for run_, k_ in (("claude_a", "ClaudeA"), ("claude_d", "ClaudeB"), ("claude_e", "ClaudeC"), ("openai_a", "CodexA"),
                             ("openai_d", "CodexB"), ("openai_e", "CodexC")):
                xs_ = [r["ratio_held_out"] for r in rows if r["run"] == run_]
                m(f"Rp{k_}Min", num(min(xs_))); m(f"Rp{k_}Max", num(max(xs_)))
            m("RpSentence", " Installed by the script in the five further main worlds, the replayed estimators would read "
              "\\RpMin--\\RpMax, \\RpAboveOne{} of \\RpN{} above 1, against \\RpOrigMin--\\RpOrigMax{} in the agents' world.")
        else:
            m("RpSentence", "")
    else:
        m("WmSentence", " Installed in five further main worlds, the script's own estimate reads \\WmScriptMin--\\WmScriptMax{} over "
          "the six (mean \\WmScriptMean, between-world SD \\WmScriptSd); \\WmResidOver{} of the ten installations there miss "
          "their estimate by more than 0.05 (\\cref{app:install}), so the readout is shown to follow the estimate in the "
          "original world only.")
        m("RpSentence", "")
else:
    m("WmSentence", ""); m("RpSentence", "")
