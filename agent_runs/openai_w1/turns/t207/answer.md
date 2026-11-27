The value head already has accessible input features, but a previous factual-only innovation decoder failed validation. I’ll now measure where the posterior signal disappears and inspect the paired-state caches. That will tell us whether to strengthen transmission or work directly at the readout.

```python
import numpy as np
import torch

def np185(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("PATHWAY WEIGHT SCALES")
for name185 in ["core.z_in.weight", "core.z_in.bias", "core.c_out.weight",
                "core.b_in.weight", "core.delta_in", "core.a_in",
                "core.out_norm.weight", "core.out_norm.bias"]:
    if name185 in NAMES:
        a185 = np185(load_original(name185))
        print(name185, a185.shape,
              "rms", round(float(np.sqrt(np.mean(a185*a185))), 6),
              "range", np.round([a185.min(), a185.max()], 6).tolist())

print("\nPAIRED STATE DIFFERENCES")
for folder185 in ["/work/fwd_11", "/work/fwd_14"]:
    try:
        o185 = np.load(folder185 + "/states_original.npz")
        e185 = np.load(folder185 + "/states_edited.npz")
        print(folder185, "keys", o185.files)
        xo185, xe185 = o185["x"], e185["x"]
        if xo185.shape == xe185.shape:
            for label185, sl185 in [("out", slice(0,192)),
                                    ("h", slice(192,960)),
                                    ("z", slice(960,992))]:
                aa185 = xo185[:, sl185]
                dd185 = xe185[:, sl185] - aa185
                print(label185, "original RMS", float(np.sqrt(np.mean(aa185**2))),
                      "difference RMS", float(np.sqrt(np.mean(dd185**2))),
                      "max difference", float(np.max(np.abs(dd185))))
    except Exception as ex185:
        print(type(ex185).__name__, str(ex185)[:160])

print("\nREPLAY CHECKS")
for key185 in ["rx_replay_best127", "rx_coda_err131", "rx_fit131",
               "rx_tests133", "rx_tests127"]:
    if key185 in globals():
        print(key185, str(globals()[key185])[:650])

print("\nHEAD CACHES")
lines185 = []
for k185, v185 in list(globals().items()):
    if k185.startswith(("he_", "hp_", "r165_")):
        if hasattr(v185, "shape"):
            detail185 = str(tuple(v185.shape))
        elif isinstance(v185, dict):
            detail185 = "keys=" + str(list(v185)[:6])
        else:
            detail185 = type(v185).__name__
        lines185.append(k185 + ": " + detail185)
print("; ".join(lines185)[:2600])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 185\nMeasuring original/edited state transmission and direct readout "
            "cache availability before choosing an algebraic pathway edit. No edits.\n")
```