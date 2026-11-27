The trace uses four very low-variance action directions, but the expected-dose fit is much less reliable for norepinephrine. I’ll check how those traces translate through the posterior and identify a minimally disruptive way to carry dose innovations. Small reconstruction error would support a trial; large error would rule it out before submission.

```python
import torch.nn.functional as F
print("=== CACHED ACTION FEATURES ===")
print(compact_structure(rr50_cache[("base", 1.0)]))
print("=== ALTERNATIVE POSTERIOR FEATURES ===")
print(compact_structure(rr49_alts[(0, 1.5)]))
print("=== EXAMPLE OBSERVATION BUNDLES ===")
for kk in list(r36_world)[:5]:
    print(kk, r36_world[kk][:5])
print("=== SET ENCODER HELPER ===")
print(inspect.signature(r36_set_encode),
      [c for c in r36_set_encode.__code__.co_consts if isinstance(c, str)])
print("=== POSTERIOR NEURON IMPORTANCE ===")
rr57_Wpost1 = np.asarray(load_original("z_post.0.weight"))
rr57_bpost1 = np.asarray(load_original("z_post.0.bias"))
rr57_Wpost2 = np.asarray(load_original("z_post.2.weight"))
rr57_bpost2 = np.asarray(load_original("z_post.2.bias"))
rr57_hidden = np.asarray(rr48_hidden0)
rr57_importance = np.mean(rr57_hidden**2, axis=0) * np.sum(rr57_Wpost2**2, axis=0)
rr57_reserved = np.argsort(rr57_importance)[:4]
print("candidate neurons", rr57_reserved.tolist(),
      "importance", rr57_importance[rr57_reserved].tolist())
print("original hidden SD", np.std(rr57_hidden[:, rr57_reserved], axis=0))
print("=== OBSERVED RATES AND RESIDUALS ===")
for dj, drugname in enumerate(("insulin", "norepinephrine")):
    active = rr48_present[:, dj] > 0
    rr = rr48_rates[active, dj]
    tt = repair_post_targets[active, dj]
    ccoef, cint, ca = rr51_assignment[drugname]
    ep = rr51_assign_X[active] @ ccoef + cint
    print(drugname, "n", int(active.sum()),
          "rate quantiles", np.quantile(rr, [0, .1, .5, .9, 1]).round(4),
          "innovation SD", float(np.std(tt)),
          "expected-rate quantiles", np.quantile(ep, [0, .1, .5, .9, 1]).round(4))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 57\n"
            "Inspecting exact action cache layouts and low-impact posterior neurons. "
            "Expected-dose offsets require gating; NE offset accuracy remains weak. No saved edit.\n"
            "Counts: 57 Python runs, 6 forward passes.\n")
```