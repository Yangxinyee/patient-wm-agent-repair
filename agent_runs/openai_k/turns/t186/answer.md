The retained estimates support immediate insulin effects near −2 per rate unit on glucose and norepinephrine effects near +3 on arterial MAP, with longer persistence for glucose. In those physical units, the high-penalty decoder’s factual noise may be tolerable. I’ll test whether the existing value-head features can express a regularized correction; held-out dose accuracy and small factual perturbations would support it.

```python
import numpy as np

def r170_brief(v):
    if isinstance(v, dict):
        return {str(k): r170_brief(x) for k, x in list(v.items())[:8]}
    if isinstance(v, (list, tuple)):
        return [r170_brief(x) for x in v[:5]]
    if hasattr(v, "shape"):
        a = np.asarray(v)
        return {"shape": tuple(a.shape), "first": a.reshape(-1)[:4].tolist()}
    return str(v)[:160]

print("=== DECODER AND CONTRAST DATA ===")
for k in ["decoders164", "basis164", "q164base", "q164cf", "dc164",
          "Y164", "noise164", "nuisance164", "groups151",
          "V144o", "V144e", "F143", "M143", "W144",
          "lastvalues151", "lastages151", "events151",
          "dt_next151", "baseline_contrast156", "dtrain158",
          "calibration_ids138"]:
    if k in globals():
        print(k, r170_brief(globals()[k]))

print("\n=== VALUE-HEAD / REGISTER HELPERS ===")
for k in ["views144", "assay_head110", "bin_assay112",
          "predictions119", "r49_readouts", "local_outputs72"]:
    if k in globals():
        c = globals()[k].__code__
        print(k, "args", c.co_varnames[:c.co_argcount],
              "globals", c.co_names,
              "constants", [x for x in c.co_consts
                            if isinstance(x, (str, int, float, tuple))][:28])

print("\n=== RELEVANT RETAINED HEAD VARIABLES ===")
r170_keys = [k for k, v in list(globals().items())
             if not k.startswith("_") and not callable(v)
             and any(s in k.lower() for s in
                     ["head", "register", "logit", "layout", "last",
                      "centre", "center", "bins", "chosen112"])]
print(" ".join(sorted(r170_keys))[:1600])

print("\n=== VOCAB NORMALIZATION AND CENTRES ===")
for k in VOCAB:
    if any(s in k.lower() for s in ["norm", "centre", "center", "code"]):
        print(k, r170_brief(VOCAB[k]))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 170: reconsider regularized value-head correction\n"
        "- Immediate record estimates: insulin about -2 per rate unit; NE "
        "arterial MAP about +3.05; glucose persistence exceeds MAP persistence.\n"
        "- High-penalty decoder noise may be tolerable after physical scaling. "
        "Preparing a no-gradient, factual-penalized correction using existing "
        "value-head features rather than large recurrent tokens.\n"
        "- Saved edits unchanged. Python runs: 170; forward passes: 10.\n"
    )
```