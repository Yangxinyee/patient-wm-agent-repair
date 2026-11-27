The dose results show the trade-off clearly: one multi-direction NE edit improved NE readouts to +1.89/−1.63, but its factual NLL increase was too large. The reduced rank-one trial had less effect. The last run stopped on a Torch/NumPy conversion error before inspecting the reconstruction helpers; I’ll fix that and test whether repaired-state inputs explain the leakage.

```python
import inspect, numpy as np, torch
from pathlib import Path

def asnp189(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

base189 = np.load("/work/baseline77.npz")["delta"]
cur189 = asnp189(load("core.b_in.weight"))
ori189 = asnp189(load_original("core.b_in.weight"))
print("BASELINE MATCH ERROR",
      float(np.max(np.abs(cur189-ori189-base189))))

print("\nNE ARCHIVE SUMMARY")
for fn189 in ["ne_reduced107.npz", "ne_candidate151.npz",
              "ne_candidate154.npz", "ne_candidate155.npz"]:
    with np.load("/work/"+fn189, allow_pickle=False) as a189:
        print(fn189, [(k, a189[k].shape) for k in a189.files])
        if "ne_delta" in a189.files:
            sv189 = np.linalg.svd(a189["ne_delta"], compute_uv=False)
            print("norm", round(float(np.linalg.norm(a189["ne_delta"])), 3),
                  "sv", np.round(sv189[:3], 3).tolist())

print("\nPAIRED TRAINING STATE SCHEMA")
for m189 in ["original", "edited"]:
    with np.load(f"/work/fwd_22/states_{m189}.npz") as a189:
        print(m189, [(k, a189[k].shape) for k in a189.files])

print("\nRECONSTRUCTION ENTRY POINTS")
selected189 = []
for k189, v189 in list(globals().items()):
    if inspect.isfunction(v189) and any(
        s in k189.lower() for s in
        ["prelude", "bundle", "recon", "context", "core", "encode", "inputs"]
    ):
        selected189.append((k189, v189.__code__.co_varnames[:7],
                            v189.__code__.co_names[:12]))
for item189 in selected189[:26]:
    print(item189)
print("Total matching helpers:", len(selected189))

with open("/work/notes.md", "a") as f189:
    f189.write("\n## Runs 188–189: complete effect recovery and safe tensor conversion\n"
               "Baseline rollout on second tuning sample: insulin -12.708/+4.335; "
               "NE +0.099/-1.004. Candidate151 NE +1.893/-1.633 but factual cost .011068 "
               "on first sample; reduced107 NE +.587/-1.282 with cost .004018 on second. "
               "Independent arterial/cuff dynamic evidence supports NE beta near3, persistence .70. "
               "Run188 stopped on Torch/NumPy max conversion; no weights changed. "
               "Python runs: 189; forwards: 22.\n")
```