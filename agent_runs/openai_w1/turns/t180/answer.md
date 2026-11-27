The earlier record analyses suggest insulin lowers glucose and norepinephrine raises MAP after adjustment for pre-dose history. Those one-step slopes are not yet rollout targets. I’ll inspect their persistence and the exact head caches; consistent lag responses and adequate dose overlap would justify a dynamic correction.

```python
print("=== SIX-LAG RECORD ESTIMATES ===")
for key, obj in he_kernels.items():
    print("pair:", key, "fields:", list(obj))
    for field in ["kernel", "y_response", "dose_response", "propensity"]:
        if field not in obj: continue
        a = np.asarray(obj[field])
        if a.size <= 30:
            print(field, a.round(6).tolist())
        else:
            print(field, "shape", a.shape,
                  "mean/sd", float(np.nanmean(a)), float(np.nanstd(a)))
print("=== RELEVANT NOTE EXCERPTS ===")
r163_lines = [s for s in r162_notes.splitlines()
              if any(t in s.lower() for t in
                     ["impulse", "kernel=", "lag response", "six-lag",
                      "six-hour", "posterior decoder", "propensity", "causal target"])]
for s in r163_lines[-18:]:
    print(s[:350])
print("=== HEAD HELPER ===")
print("signature:", inspect.signature(ce_make_heads))
print("dependencies:", ce_make_heads.__code__.co_names)
print("string constants:", [s[:160] for s in ce_make_heads.__code__.co_consts
                            if isinstance(s,str)][:20])
for r163_k in ["ce_head", "hp_original_head", "hp_original_states", "he_kernels"]:
    r163_v = globals().get(r163_k)
    if isinstance(r163_v, dict):
        print(r163_k, [(k, tuple(v.shape) if hasattr(v,"shape") else type(v).__name__)
                       for k,v in r163_v.items()][:16])
print("=== POSTERIOR/LATENT TENSORS ===")
print([(n, SHAPES[n]) for n in NAMES
       if any(s in n for s in ["z_post", "z_in", "z_head", "z_prior"])])
print("=== POSTERIOR HELPER ===")
print(inspect.signature(r72_posterior), r72_posterior.__code__.co_names)
print("=== VOCAB CONFIG ===")
print(VOCAB["config"])
print("vocab keys:", list(VOCAB))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 163\nEarly cross-fitted pre-dose residual slopes: insulin -2.21 mg/dL per unit/hour; NE +3.27 mmHg per mcg/kg/min, with alternate channels agreeing in sign. These are observational estimates, not ground truth or rollout targets. Recovering lag kernels and exact-state head helper. Counts: 163 Python runs, 11 forwards.\n")
```