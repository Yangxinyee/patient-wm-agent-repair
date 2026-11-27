The contrast’s overall factual spread overlaps the synthetic targets, so a general off-manifold gate is not yet justified. That calculation mixed event types, however. I’ll check finger-glucose inputs separately and audit the regression matrices. Separation within that event would support a targeted gate; otherwise I’ll use modest scaling or calibration rather than a hard threshold.

```python
def as_np_recovery(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("CACHE METADATA:")
for label in ("first", "extra", "tuning"):
    m = sc_fact[label]["meta"]
    print(label, m.columns.tolist(),
          "stay range:", (m.stay_id.min(), m.stay_id.max()))

print("\nFINGER-ONLY SCORE DISTRIBUTIONS:")
for label, c in list(sc_fact.items()) + [
    ("synthetic_" + str(k), v) for k, v in sc_syn.items()
]:
    x = as_np_recovery(c["x"])
    mask = c["meta"].event.eq("glucose_finger").to_numpy()
    f = x @ w_fallback + b_fallback
    g = as_np_recovery(selective_cache(c, 30.0)["gate"])
    print(label, "n=", int(mask.sum()))
    for desc, a in (("decoder all prior draws", f[:, mask].ravel()),
                    ("decoder averaged per reading", f[:, mask].mean(axis=0)),
                    ("selective gate", g[:, mask].ravel())):
        print(" ", desc, np.round(
            np.quantile(a, [.001, .01, .1, .5, .9, .99, .999]), 5))

print("\nREGRESSION RECONSTRUCTION AUDIT:")
g_expected = d3.T @ d3 / len(d3)
h_expected = d3.T @ sc_y / len(d3)
c_expected = r3_Fq.T @ r3_Fq / len(r3_Fq)
print("g3 discrepancy:", np.max(np.abs(g3 - g_expected)))
print("h3 discrepancy:", np.max(np.abs(h3 - h_expected)))
print("r3_C discrepancy:", np.max(np.abs(r3_C - c_expected)))
for lam in (0.1, 0.3, 1.0):
    coef_recovery = np.linalg.solve(g3 + lam * r3_C, h3)
    w_recovery = r3_Q @ coef_recovery
    ref = r3_candidates[(lam, 1.0)]
    print(lam, "weight discrepancy:",
          np.max(np.abs(w_recovery - ref["w"])),
          "bias candidates:", -sc_mean @ w_recovery, ref["b"])

print("\nFINGER-SELECTIVE SCALE SCREEN:")
for scale in (6.0, 6.5, 7.0, 8.0):
    losses = {}
    for label in ("first", "extra", "tuning"):
        c = selective_cache(sc_fact[label], 30.0)
        losses[label] = float(np.mean(cache_losses(
            c, w_fallback, b_fallback, scale
        )) - np.mean(c["original_nll"]))
    print(scale, losses)
```
