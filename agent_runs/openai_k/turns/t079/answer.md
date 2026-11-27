The output layout is now clear: means are first and probabilities are fourth. The retained data also contains the correct reading mask; the attempted reconstruction failed because its keys are `(stay, hour)` tuples. I’ll use that existing mask and rerun the assay without changing the candidate.

```python
import time
import numpy as np

t73 = time.time()
mask72 = np.asarray(observed59, dtype=bool)[fp71]
mask72 &= (bins71 > 0) & (bins71 < 17)
print("Factual readings:", int(mask72.sum()))

def factual_local72():
    o, h = r59_core(fi71)
    _, p = local_outputs72(o, h, fi71)
    chosen = p[np.arange(len(fi71))[:, None], np.arange(4)[None, :], bins71]
    return float(-np.log(np.maximum(chosen[mask72], 1e-12)).mean())

diagnostics73 = []
try:
    W0.update(backup71)
    nll_original73 = factual_local72()
    print("Original reconstructed NLL:", nll_original73)
    for lam, strength in [(0.001, 0.0), (0.001, 0.3), (0.001, 1.0),
                          (0.001, 3.0), (0.0001, 0.3), (0.0001, 1.0)]:
        set_post71(D56_candidates[lam], strength)
        row = {"lambda": lam, "strength": strength,
               "dNLL": factual_local72() - nll_original73, "slopes": {}}
        for d in [0, 1]:
            ii = drug_idx71[d]
            o, h = r59_core(ii)
            baseline, _ = local_outputs72(o, h, ii)
            for multiplier in [0.0, 0.5, 2.0, 3.0]:
                o, h = r59_core(ii, u=cf_u71[d, multiplier])
                means, _ = local_outputs72(o, h, ii)
                dose_change = rate66[ii, d] * (multiplier - 1.0)
                slope = (dose_change[:, None] * (means-baseline)).sum(axis=0) / (
                    np.square(dose_change).sum() + 1e-12)
                row["slopes"][d, multiplier] = np.round(slope, 3).tolist()
        diagnostics73.append(row)
        print(row)
finally:
    W0.update(backup71)

print("Seconds:", round(time.time()-t73, 2), "Submitted edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 72–73: corrected posterior-candidate frontier\n"
            "- Run 72 identified the tuple layout but stopped because future_index59 contains (stay, hour) keys.\n"
            "- Run 73 uses the retained boolean observed59 mask; candidate diagnostics are in diagnostics73.\n"
            "- Scratch weights restored after all tests; no submitted edits.\n"
            "- Python runs: 73; forward passes: 4.\n")
```