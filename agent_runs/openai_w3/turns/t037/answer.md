The original rollout’s insulin response is much smaller than the record-derived estimate, and its norepinephrine-doubling response has the opposite sign. Drug-start posteriors already match locally, so that action pathway is usable. I’ll make one focused check of observation construction before editing it; systematic residuals would point to an implementation error, while noise-like residuals would argue against further reconstruction work.

```python
print("=== ATOM CONSTRUCTION ===")
print(compact_bytecode(make_atom))
print("=== CONTEXT CONSTRUCTION ===")
print(compact_bytecode(state_context))
print("=== SET ENCODER ===")
print(compact_bytecode(encode_one_set))
print("=== CORE HELPERS ===")
print(compact_bytecode(layer_norm))
print(compact_bytecode(mlp))
print("=== POSTERIOR RESIDUAL STRUCTURE ===")
for ev in ["map_cuff", "glucose_finger", "map_arterial", "glucose_lab", "insulin", "norepinephrine"]:
    ids0 = np.asarray(isolated_by_event.get(ev, []), dtype=int)[:40]
    if not len(ids0):
        continue
    ec = torch.stack([make_atom(isolated_record(int(i), ev)) for i in ids0])
    isdrug = ev in ("insulin", "norepinephrine")
    empty = encode_one_set(torch.empty((0,192)), "encode").expand(len(ids0), -1)
    enc = torch.cat([encode_one_set(a[None], "encode_action" if isdrug else "encode")
                     for a in ec], dim=0)
    calc = posterior_mean(state_context(ids0),
                          empty if isdrug else enc,
                          enc if isdrug else torch.zeros_like(enc))
    residual = as_numpy(calc - sx[ids0,960:])
    print(ev, "n", len(ids0), "MSE", float(np.mean(residual**2)),
          "mean residual energy", float(np.mean(residual.mean(0)**2)),
          "singular values", np.linalg.svd(residual, compute_uv=False)[:4])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 32; forwards 3. Original rollout misses record-derived drug effects. Exact isolated-action posterior remains a usable intervention pathway. Checking observation construction once more before choosing action edit.\n")
```