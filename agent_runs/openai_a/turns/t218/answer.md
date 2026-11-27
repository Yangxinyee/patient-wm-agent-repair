The graft adds a glucose-bin tilt through two gated GELU units. Its current gate activates for both glucose sensors, so suppressing it for lab readings could free factual-fit margin. I’ll screen selective gates and stronger tilts on the cached inputs. Lower NLL cost with unchanged synthetic finger contrasts would justify a GPU trial, but would not yet establish rollout accuracy.

```python
import numpy as np, torch, pandas as pd

print("EXISTING SEPARATOR STRUCTURES")
for lam, entry in gate_sep_candidates.items():
    if isinstance(entry, dict):
        print(lam, {k: (tuple(v.shape) if hasattr(v, "shape") else repr(v)[:80])
                    for k, v in entry.items()})
    else:
        print(lam, type(entry).__name__,
              [(tuple(v.shape) if hasattr(v, "shape") else repr(v)[:80])
               for v in entry] if isinstance(entry, (tuple, list)) else "")

entry = gate_sep_candidates[0.01]
if isinstance(entry, dict) and "w" in entry and "b" in entry:
    selective_sep_w = np.asarray(entry["w"], dtype=np.float64)
    selective_sep_b = float(entry["b"])
elif isinstance(entry, (tuple, list)) and len(entry) == 2:
    selective_sep_w = np.asarray(entry[0], dtype=np.float64)
    selective_sep_b = float(entry[1])
else:
    # Small record-state regression only; no model gradients.
    XX = np.concatenate([sc_fact[s]["x"].mean(axis=0) for s in ["first", "extra"]])
    ee = np.concatenate([sc_fact[s]["meta"].event.to_numpy() for s in ["first", "extra"]])
    keep = np.isin(ee, ["glucose_finger", "glucose_lab"])
    XX = XX[keep].astype(np.float64)
    yy = -(ee[keep] == "glucose_lab").astype(np.float64)
    xm, ym = XX.mean(axis=0), yy.mean()
    xc, yc = XX - xm, yy - ym
    selective_sep_w = np.linalg.solve(xc.T @ xc + 0.01*np.eye(192), xc.T @ yc)
    selective_sep_b = float(ym - xm @ selective_sep_w)

def selective_cache(cache, gate_strength):
    out = dict(cache)
    separator = cache["x"] @ selective_sep_w.astype(np.float32) + np.float32(selective_sep_b)
    out["gate"] = cache["gate"] + np.float32(gate_strength) * separator
    return out

print("\nSEPARATOR SYNTHETIC EXTRAPOLATION")
for factor, cache in sc_syn.items():
    score = cache["x"] @ selective_sep_w + selective_sep_b
    print(factor, np.round(np.quantile(score, [.001, .01, .5, .99, .999]), 4))

selective_grid = []
for strength in [0., 20., 30., 40.]:
    fact_c = {s: selective_cache(c, strength) for s, c in sc_fact.items()}
    syn_c = {f: selective_cache(c, strength) for f, c in sc_syn.items()}
    for lam in [0.03, 0.1, 0.3]:
        candidate = r3_candidates[(lam, 1.0)]
        for scale in [2., 3., 4., 6.]:
            fit = {s: sc_evaluate(c, candidate["w"], candidate["b"], scale)["nll"]
                      - c["original_nll"] for s, c in fact_c.items()}
            pred = {f: sc_evaluate(c, candidate["w"], candidate["b"], scale)["mean"]
                    for f, c in syn_c.items()}
            shifts = {f: float((pred[f] - pred[1.0]).mean()) for f in [0.5, 2.0]}
            selective_grid.append(dict(strength=strength, lam=lam, scale=scale,
                                       **fit, half=shifts[0.5], double=shifts[2.0]))

selective_grid_df = pd.DataFrame(selective_grid)
print("\nSELECTIVE-GATE SCREEN")
print(selective_grid_df[
    (selective_grid_df.strength == 30.) |
    ((selective_grid_df.lam == .1) & (selective_grid_df.scale == 2.))
].round(6).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Finger-selective gate screen; Python 193, forwards 16/50\n")
    f.write("- Recovered two-GELU contrast algebra and insulin RD target; NE remains less securely identified.\n")
    f.write("- Testing code-selective gates in cached inputs only. Current scale-2 GPU checkpoint remains unchanged.\n")
    f.write(selective_grid_df.round(6).to_string(index=False) + "\n")
```
