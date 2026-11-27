The factual penalty is concentrated in glucose predictions, especially glucose_lab; MAP costs much less. That makes a modest insulin reduction worth investigating. I’ll identify its exact head components so I can scale insulin and MAP independently, then check whether the saved likelihood budget buys a useful MAP improvement.

```python
import numpy as np
import torch
import pandas as pd

print("HEAD TENSOR ORDER", head177_names)
ins241 = {}
with np.load("/work/best_insulin_v1.npz") as z:
    ins241 = {k: torch.as_tensor(z[k].copy(), dtype=torch.float32)
              for k in z.files}

for label, pack in [("insulin checkpoint", ins241),
                    ("compressed insulin base", head177_compressed)]:
    w0n, b0n, w2n, b2n = head177_names
    ow0 = torch.as_tensor(load_original(w0n))
    ob0 = torch.as_tensor(load_original(b0n))
    ow2 = torch.as_tensor(load_original(w2n))
    ob2 = torch.as_tensor(load_original(b2n))
    diff0 = torch.linalg.vector_norm(pack[w0n] - ow0, dim=1)
    ix = torch.where(diff0 > 1e-4)[0]
    print("\n", label, "changed input rows:", ix.tolist())
    for j in ix.tolist():
        print(" row", j, "input_delta", round(float(diff0[j]), 5),
              "bias", round(float(pack[b0n][j]), 5),
              "output_norm", round(float(torch.linalg.vector_norm(pack[w2n][:, j])), 5),
              "output_vs_tilt",
              round(float(torch.dot(pack[w2n][:, j],
                                    torch.as_tensor(wm_v151, dtype=torch.float32))), 5))
    print("final bias delta:", (pack[b2n] - ob2).numpy().round(5))
    print("largest output-column changes:",
          torch.topk(torch.linalg.vector_norm(pack[w2n] - ow2, dim=0), 8))

print("\nTRIAL-158 OBJECTS")
for k, v in sorted(dict(globals()).items()):
    if k.startswith("trial158"):
        print(k, getattr(v, "shape", type(v).__name__),
              str(list(v)[:8]) if isinstance(v, dict) else "")

print("\nCACHE RAW FIELDS")
print({k: getattr(v, "shape", type(v).__name__)
       for k, v in wm_train145["raw"].items()})
print("events:", np.unique(events206, return_counts=True))
print("fit counts:", np.unique(events206[fit236], return_counts=True))
print("held-out counts:", np.unique(events206[~fit236], return_counts=True))
print("tilt vector:", np.asarray(wm_v151))

print("\nSELECTIVE LINEAR SCREEN")
print("tab234b columns:", list(tab234b.columns))
print(tab234b.head(6).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 241: insulin component recovery\n"
        "- GPU factual costs are predominantly glucose: finger about +0.0059, lab +0.009–0.012 per reading.\n"
        "- Recovering independent insulin scaling to improve slightly overshooting rollouts and free MAP budget.\n"
        "- No weights changed.\n"
    )
```