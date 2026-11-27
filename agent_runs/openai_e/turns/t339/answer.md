Candidate A predicts insulin shifts of **−10.59 / +5.15** and norepinephrine shifts of **+0.33 / −0.85**. All directions are now correct, but magnitudes remain below the record-derived estimates. I’ll test a stronger insulin correction with modest norepinephrine strength, judging factual fit across all 400 tuning stays rather than one cohort alone.

```python
_r284_o = pd.read_csv("/work/fwd_41/factual_original.csv")
_r284_e = pd.read_csv("/work/fwd_41/factual_edited.csv")
_r284_a_cost = float(_r284_e["nll"].mean() - _r284_o["nll"].mean())
_r284_d = pd.read_csv("/work/fwd_42/dose_edit_edited.csv")
_r284_reached = _r284_d[
    _r284_d["reached"].astype(str).str.lower().isin(["true", "1"])
    & _r284_d["hour"].between(13, 24)]
_r284_a_effects = _r284_reached.groupby("edit")["shift"].mean().to_dict()
print("Candidate A exact cost:", _r284_a_cost)
print("Candidate A reached-hour effects:", _r284_a_effects)

with np.load("/work/joint268.npz") as _r284_z:
    _r284_source = {k: _r284_z[k].copy() for k in _r284_z.files}
_r284_i_scale, _r284_ne_scale = 0.80, 0.25
_r284_i = (_r284_i_scale / float(_r284_source["insulin_scale"])) * _r284_source["insulin_delta"]
_r284_ne = (_r284_ne_scale / float(_r284_source["ne_scale"])) * _r284_source["ne_delta"]
_r284_residual = (_r284_source["delta"] - _r284_source["insulin_delta"]
                  - _r284_source["ne_delta"])
_r284_delta = (_r284_residual + _r284_i + _r284_ne).astype(np.float32)
save_edit("core.b_in.weight", _r280_original + _r284_delta)
np.savez("/work/joint284.npz",
         delta=_r284_delta, insulin_delta=_r284_i, ne_delta=_r284_ne,
         insulin_scale=_r284_i_scale, insulin_blend=0.60, ne_scale=_r284_ne_scale)
print("Installed unaccepted B: insulin=.80, blend=.60, NE=.25")
print("Edits:", edited())

with open("/work/notes.md", "a") as _r284_f:
    _r284_f.write(
        "\n## Run 284: candidate A measured; candidate B installed\n"
        "Aggregation resolved: evaluator averages reached stay-hours, not "
        "equal-stay means. Forward41 A factual delta=" + str(_r284_a_cost) +
        "; forward42 reached-hour effects=" + json.dumps(_r284_a_effects) + ". "
        "A improves all four original responses but still underestimates effects. "
        "Installed unaccepted B: I=.80, blend=.60, NE=.25, saved joint284.npz. "
        "Closed-form recombination of archived directions, no gradients/optimizer. "
        "Plan: paired factual across all400 tuning stays and exact rollout. "
        "Judge aggregate fit with uncertainty; one-cohort failure alone does not "
        "establish population failure. Joint274 remains validated fallback.\n"
        "Python runs: 284; forwards: 42.\n"
    )
```