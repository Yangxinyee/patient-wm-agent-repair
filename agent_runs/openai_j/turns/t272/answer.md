The insulin correction uses two identifiable neurons, so its strength can be reduced without changing its learned direction. I’ll screen independent insulin/MAP strengths using the exact saved-head computation. A promising candidate should reduce insulin’s overshoot and increase MAP sensitivity while retaining a factual-fit margin on held-out stays.

```python
import numpy as np
import pandas as pd
import torch

w0n242, b0n242, w2n242, b2n242 = head177_names
live242 = {n: torch.as_tensor(load(n), dtype=torch.float32).clone()
           for n in head177_names}
j0242, j1242 = map(int, head177_free)
mapcoef242 = torch.cat([
    live242[w0n242][j0242] - live242[w0n242][j1242],
    (live242[b0n242][j0242] - live242[b0n242][j1242]).reshape(1)
]).numpy()
cal242 = np.asarray(map192_cal, dtype=np.float32)
mapdir242 = (mapcoef242 - cal242) / 8.0

trainmc242 = curve236["train"]["mc"]
tunemc242 = curve236["tune"]["mc"]
orig242 = {n: torch.as_tensor(load_original(n), dtype=torch.float32)
           for n in head177_names}
base_train242 = head177_eval(trainmc242, orig242)["nll"]
base_tune242 = head177_eval(tunemc242, orig242)["nll"]

def make242(ins_strength, map_gain):
    ww = head177_make(cal242 + float(map_gain) * mapdir242)
    ww[w2n242][:, [89, 170]] *= float(ins_strength)
    return ww

candidates242 = {}
rows242 = []
ne_cache242 = {}

for gain in [6.0, 8.0, 9.0, 10.0, 11.0, 12.0]:
    for strength in [0.80, 0.85, 0.90, 0.95, 1.0]:
        ww = make242(strength, gain)
        tr = head177_eval(trainmc242, ww)
        tu = head177_eval(tunemc242, ww)
        dt = tr["nll"] - base_train242
        du = tu["nll"] - base_tune242
        row = {
            "ins_strength": strength, "map_gain": gain,
            "fit_dNLL": float(dt[fit236].mean()),
            "cv_dNLL": float(dt[~fit236].mean()),
            "tune_dNLL": float(du.mean())
        }
        if row["cv_dNLL"] < 0.0045 and row["tune_dNLL"] < 0.0047:
            candidates242[(strength, gain)] = ww
            if gain not in ne_cache242:
                refww = make242(1.0, gain)
                reftr = head177_eval(trainmc242, refww)
                metrics = {}
                for factor in [0.5, 2.0]:
                    cf = head177_eval(
                        curve236[("norepinephrine", factor)]["mc"], refww)
                    for event in ["map_arterial", "map_cuff"]:
                        metrics[event + "/" + str(factor)] = float(slope237(
                            cf["mean"], reftr["mean"], "norepinephrine",
                            factor, event, ~fit236))
                ne_cache242[gain] = metrics
            row.update(ne_cache242[gain])
        rows242.append(row)

frontier242 = pd.DataFrame(rows242)
frontier242.to_csv("/work/independent_strength_frontier242.csv", index=False)
print(frontier242.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\nCandidate count:", len(candidates242))
print("Reconstructed gain-8 coefficient error:",
      float(np.max(np.abs(mapcoef242 - np.asarray(coef234)))))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 242: independent insulin/MAP strength frontier\n"
        "- Screened independent scaling of insulin output neurons and the calibrated MAP direction.\n"
        "- Exact cached-head factual losses checked on training-stay holdout and tuning.\n"
        "- No weights installed; GPU validation remains necessary.\n"
    )
```