The stored gate separates readings almost perfectly in its fitted feature space, but it fails on the sampled head inputs. That points to a reconstruction or feature-basis mismatch, not a usable gate. I’ll compare those inputs explicitly and recover the record-based rollout targets. Agreement between fitted and actual input spaces is required before another edit.

```python
print("=== CONTROLLED-RESPONSE SUMMARY ===")
print(notes_text[9300:12987])
print("\n=== RESIDUAL SLOPE DIAGNOSTICS ===")
for drug, stats in repair_slope_diagnostics.items():
    print(drug, stats)

print("\n=== GATE / QUERY GLOBAL NAMES ===")
print([n for n in globals() if any(
    s in n.lower() for s in ("query", "gate", "project", "latent", "wc", "draw", "code_type")
)])

print("\n=== SAMPLE-ARRAY SHAPES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        if obj.ndim == 3 and (4510 in obj.shape or 1000 in obj.shape):
            print(name, tuple(obj.shape))
        elif name.startswith(("mc_", "rec_", "rh_")):
            print(name, tuple(obj.shape))

print("\n=== STORED FEATURE COMPARISONS ===")
xgate = np.asarray(rec_head_X)
for name, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)) and tuple(obj.shape) == (4510, 192):
        a = obj.detach().numpy() if isinstance(obj, torch.Tensor) else obj
        print(name, "RMS difference", round(float(np.sqrt(np.mean((xgate-a)**2))), 5))
print("rec_gate_v norm", float(np.linalg.norm(rec_gate_v)), "intercept", rec_gate_b)
print("wr_dt range", float(wr_dt.min()), float(wr_dt.max()))
print("wr_nf first two rows", np.asarray(wr_nf)[:2])
print("wr_codes first ten", np.asarray(wr_codes)[:10])

print("\n=== ORIGINAL WEIGHT-CACHE NAMES ===")
print([n for n, o in list(globals().items())
       if isinstance(o, dict) and "type_emb.weight" in o])
```
