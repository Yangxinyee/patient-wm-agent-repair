The record regressions show little cross-drug response, while both glucose sensors support the insulin effect. I’ll now rebuild factual features from exported states and verify the corrected action transition. If those checks pass, I can use altered-dose features to fit the correction without relying on an unvalidated simulator reconstruction.

```python
import pandas as pd

print("=== AVAILABLE FORWARD FILES ===")
factual_tables80 = {}
for path in sorted(Path("/work").glob("fwd_*/*")):
    print(str(path), path.stat().st_size)
    if "factual_original" in path.name and path.suffix == ".csv":
        tab = pd.read_csv(path)
        factual_tables80[str(path)] = tab
        print(" columns:", list(tab.columns), "rows:", len(tab))
        print(tab.head(2).to_string(index=False))

print("\n=== FACTUAL FEATURES FROM EXPORTED STATES ===")
ctx_fact80, _ = context64(
    sx63[pi73, :192],
    sx63[pi73, 192:960].reshape(-1, 4, 192),
    dtread73
)
means_fact80, probs_fact80, hidden_fact80, logits_fact80 = decode67(
    ctx_fact80, const_train67, trans_train67, centres_train67,
    epsread73, return_features=True
)
for ev in np.unique(events74):
    mask = events74 == ev
    print(ev, "n", int(mask.sum()),
          "mean", float(means_fact80[:, mask].mean()))
print("first reading keys:",
      list(zip(sid63[pi73[:6]],
               (hour63[pi73] + dtread73.numpy()/3600)[:6],
               events74[:6])))

print("\n=== ACTION TRANSITION VALIDATION ===")
action_idx80 = np.asarray(drug_indices49, dtype=np.int64)
check80 = action_idx80[:128]
ca80, hf80, mem80, age80, pad80 = bundle_inputs74(check80)
post80 = mlp63(torch.cat(
    [ca80, e_valid60[check80], u_valid60[check80]], dim=-1), "z_post")
out_check80, h_check80 = transition64(
    hf80, e_valid60[check80], u_valid60[check80], post80[:, :32],
    mem80, age80, ~pad80
)
print("out RMS", float((out_check80-sx63[check80, :192]).square().mean().sqrt()))
print("h RMS", float((h_check80-sx63[check80, 192:960].reshape(-1,4,192))
                     .square().mean().sqrt()))
print("re-encoded action RMS",
      float((u72-u_valid60).square().mean().sqrt()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 77–80 / head-only plan\n"
            "Recovered accepted DML/RD insulin anchor ~-1.95, NE ~+3.1 and "
            "cross-sensor dynamic kernels. Provisional stay-weighted x2 targets "
            "I=-10.675, NE=+4.326; row-weighted I=-12.769, NE=+4.504. "
            "Simple SSM targets remain rejected. Plan: analytically fit a final "
            "value-head correction using dose-altered hidden features and factual "
            "preservation constraints. No edits yet.\n")
```