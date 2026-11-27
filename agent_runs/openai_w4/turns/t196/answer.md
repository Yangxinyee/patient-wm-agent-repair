The retained construction shows that the explicit innovation channel is **insulin-only**, which explains why this edit barely affects norepinephrine. The records also suggest modest dose nonlinearity, so exact doubling/halving symmetry is not assured. I’ll inspect how the channel occupies the latent and attention dimensions before adding a separate norepinephrine channel. A dose-sensitive channel with little factual leakage would support that approach.

```python
import inspect, re, numpy as np, torch

def array171(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("CURRENT LATENT/ATTENTION EDIT STRUCTURE")
for name171 in [
    "z_post.0.weight", "z_post.0.bias", "z_post.2.weight", "z_post.2.bias",
    "z_in.weight", "z_in.bias",
    "core.prelude.0.attn.q.weight",
    "core.prelude.0.attn.kv.weight",
    "core.prelude.0.attn.out.weight"
]:
    delta171 = array171(load(name171)) - array171(load_original(name171))
    if delta171.ndim == 2:
        rows171 = np.flatnonzero(np.max(np.abs(delta171), axis=1) > 1e-7)
        cols171 = np.flatnonzero(np.max(np.abs(delta171), axis=0) > 1e-7)
        print(name171, delta171.shape, "changed rows", rows171.tolist()[:40],
              "nrows", len(rows171), "changed cols", cols171.tolist()[:40],
              "ncols", len(cols171), "norm", float(np.linalg.norm(delta171)))
    else:
        nz171 = np.flatnonzero(np.abs(delta171) > 1e-7)
        print(name171, delta171.shape, "changed entries", nz171.tolist()[:40],
              "n", len(nz171), "norm", float(np.linalg.norm(delta171)))

print("\nCHOSEN CHANNEL, EXCLUDING WEIGHTS/CAPTURE ARRAYS")
for key171, val171 in chosen_channel116.items():
    if key171 not in ["weights", "capture"]:
        if isinstance(val171, (np.ndarray, torch.Tensor)):
            aa171 = array171(val171)
            print(key171, "shape", aa171.shape, "norm", float(np.linalg.norm(aa171)),
                  "values", aa171.tolist() if aa171.size <= 8 else "(omitted)")
        else:
            print(key171, str(val171)[:500])

print("\nREPAIR BANK")
for key171, val171 in repair_bank.items():
    print(key171, type(val171).__name__, getattr(val171, "shape", None),
          list(val171.keys())[:12] if isinstance(val171, dict) else "")

print("\nCHANNEL-CONSTRUCTION OBJECTS/FUNCTIONS")
for name171 in sorted(list(globals())):
    if re.search(r"(109|110|111|112|113|114|115|116|117|118)$", name171):
        obj171 = globals()[name171]
        if inspect.isfunction(obj171):
            print("FUNCTION", name171, inspect.signature(obj171),
                  "dependencies", obj171.__code__.co_names)
        elif name171 not in ["chosen_channel116", "channel_fits111", "channel_fits112"]:
            if isinstance(obj171, dict):
                print(name171, "dict keys", list(obj171.keys())[:14])
            else:
                print(name171, type(obj171).__name__, getattr(obj171, "shape", None),
                      str(obj171)[:100] if np.isscalar(obj171) else "")

print("\nCONFIG:", VOCAB["config"])

with open("/work/notes.md", "a") as log171:
    log171.write(
        "\n\n## Run 171\n"
        "Run-117/123 construction is explicitly insulin-only, explaining the lack "
        "of NE repair. Run-137 quadratic immediate fits: insulin linear -1.6589, "
        "quadratic -.0740; NE linear +2.3983, quadratic +.5930. "
        "These local observational fits do not establish doubled-dose extrapolation. "
        "Inspecting occupied latent/attention dimensions and channel-building interfaces "
        "to construct an independent NE innovation channel. No weights changed.\n"
    )
```
