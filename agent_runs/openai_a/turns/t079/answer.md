The decoder’s extreme outputs occur outside the valid dose-history rows; its outputs on valid glucose rows are much smaller. The current gate selects glucose but does not exclude those invalid histories. I’ll fit a stricter gate and check sensitivity to latent draws. Suppressing both problems would make a factual-safe probe plausible.

```python
from sklearn.linear_model import Ridge

xmean0 = np.asarray(raw_input_sets["tuning"])
print("MEAN-INPUT CONSISTENCY")
for label0, arr0 in [("xx", xx), ("qq", qq)]:
    arr0 = arr0.detach().cpu().numpy() if hasattr(arr0, "detach") else np.asarray(arr0)
    if arr0.shape == xmean0.shape:
        pp0 = arr0 @ graft_decoder_w + float(graft_decoder_b)
        vm0 = np.asarray(valid_glucose_masks["tuning"], bool)
        print(label0, "input RMSE", float(np.sqrt(np.mean((arr0-xmean0)**2))),
              "decoder valid", short_stats(pp0[vm0]),
              "decoder invalid glucose",
              short_stats(pp0[np.asarray(glucose_masks["tuning"], bool) & ~vm0]))

gate_train0 = np.asarray(raw_input_sets["first"])
gate_extra0 = np.asarray(raw_input_sets["extra"])
gate_y0 = np.where(np.asarray(valid_glucose_masks["first"], bool), 1.0, -1.0)
gate_loc1 = gate_train0.mean(0)
gate_scale1 = np.maximum(gate_train0.std(0), 0.05)
gate_models1 = {}
print("\nVALID-HISTORY GATE")
for alpha1 in [0.1, 1.0, 10.0, 100.0]:
    reg1 = Ridge(alpha=alpha1).fit((gate_train0-gate_loc1)/gate_scale1, gate_y0)
    w1 = reg1.coef_/gate_scale1
    b1 = float(reg1.intercept_ - gate_loc1 @ w1)
    gate_models1[alpha1] = (w1, b1)
    print("alpha", alpha1)
    for s1 in ["extra", "tuning"]:
        margin1 = np.asarray(raw_input_sets[s1]) @ w1 + b1
        mask1 = np.asarray(valid_glucose_masks[s1], bool)
        print(s1, "valid", short_stats(margin1[mask1]),
              "invalid", short_stats(margin1[~mask1]),
              "classification error", float(np.mean((margin1>0) != mask1)))

print("\nCPU SAMPLING FORMULAS")
for fn_name1 in ["cpu_head_parts", "cpu_head_probs"]:
    print(fn_name1)
    fn1 = globals()[fn_name1]
    try:
        text1 = inspect.getsource(fn1)
    except Exception:
        text1 = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(fn1))
    print(text1[:2500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Graft failure localization; Python 70, forward 7/50\n"
            "- Mean-prior decoder is small on valid glucose histories but extreme on initial/invalid rows.\n"
            "- Existing gate selects glucose only; fitting a valid-history gate and checking latent noise.\n"
            "- No weight edits applied.\n")
```