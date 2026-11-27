The aligned transition logits and numeric registers match, but the fixed decoder inputs still differ. I’ll isolate whether that residual is a constant offset or a history-dependent term. An exact match would validate reconstruction for calibration; otherwise I’ll keep investigating before changing weights.

```python
import numpy as np, torch

d169 = (static168 - head_static).numpy()
print("STATIC RESIDUAL RMS", np.sqrt(np.mean(d169**2)))
print("AFTER GLOBAL CONSTANT",
      np.sqrt(np.mean((d169 - d169.mean(0))**2)))
for ev in aligned_meta168.event.unique():
    ix = aligned_meta168.event.eq(ev).to_numpy()
    dd = d169[ix]
    print(ev, "mean norm", np.linalg.norm(dd.mean(0)),
          "within rms", np.sqrt(np.mean((dd - dd.mean(0))**2)))

design169 = np.column_stack([
    np.ones(len(aligned_meta168)),
    aligned_meta168.has_last.to_numpy(),
    np.log1p(aligned_meta168.age.to_numpy()) / 10,
    aligned_meta168.last_z.to_numpy(),
])
coef169 = np.linalg.lstsq(design169, d169, rcond=None)[0]
resid169 = d169 - design169 @ coef169
print("REGISTER-LINEAR RESIDUAL",
      np.sqrt(np.mean(resid169**2)), np.max(np.abs(resid169)))
print("COEFFICIENT NORMS", np.linalg.norm(coef169, axis=1))

candidates169 = {}
for name in ["last_numeric.bias", "embed.numeric.bias", "query_norm.bias"]:
    if name in W0:
        candidates169[name] = W0[name].numpy()
for name in ["last_status.weight", "embed.bins.rate.weight",
             "embed.bins.value.weight", "type_emb.weight"]:
    if name in W0:
        for i in range(min(2, len(W0[name]))):
            candidates169[f"{name}[{i}]"] = W0[name][i].numpy()
print("CONSTANT CANDIDATE DISTANCES")
for name, vec in candidates169.items():
    print(name,
          "plus", np.sqrt(np.mean((d169.mean(0) - vec)**2)),
          "minus", np.sqrt(np.mean((d169.mean(0) + vec)**2)))

print("\nTRUSTED QUERY IMPLEMENTATION")
print(compact_bytecode166(decoder_query, 3100))
print("\nLINEAR IMPLEMENTATION")
print(compact_bytecode166(linear0, 600))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 169: isolate remaining decoder offset\n"
        "Aligned transitions match exactly and numeric registers match to 1.2e-7, "
        "but static input RMS mismatch remains .3407. Tested constant and "
        "register-linear residual explanations and inspected trusted query code. "
        "No weights changed. Python runs: 169; forwards: 22.\n"
    )
```