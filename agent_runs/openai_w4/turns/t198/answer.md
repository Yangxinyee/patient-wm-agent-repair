The insulin signal uses a direction in latent space, not a single latent coordinate. I’ll check whether its reserved attention head has an unused value direction and recover the exact action inputs. A rank-one latent map and spare value direction would support adding norepinephrine without disturbing insulin; otherwise the shared-head approach needs additional compensation.

```python
import inspect, numpy as np, torch, pandas as pd, pathlib

print("REFERENCE BANK")
print("stay IDs:", np.unique(repair_bank["stay_id"])[:8].tolist(),
      "n_stays:", len(np.unique(repair_bank["stay_id"])))
print("reference_x125:", type(reference_x125).__name__, getattr(reference_x125, "shape", None))
print("reference_bank125:", type(reference_bank125).__name__)
print("indices102:", len(indices102), "first:", indices102[:6].tolist())

zin173 = array171(load("z_in.weight"))
kv173 = array171(load("core.prelude.0.attn.kv.weight"))
out173 = array171(load("core.prelude.0.attn.out.weight"))
print("\nLATENT MAP SINGULAR VALUES:", np.linalg.svd(zin173, compute_uv=False).round(6).tolist())
print("latent axis norm:", np.linalg.norm(latent_axis104))
print("carrier/signature norms:", np.linalg.norm(carrier104), np.linalg.norm(signature104))
print("carrier/signature dot:", np.dot(carrier104, signature104))
print("lambda:", lambda104)
print("reserved-head VALUE ROW NORMS:",
      np.linalg.norm(kv173[240:288], axis=1).round(5).tolist())
print("reserved-head OUTPUT COLUMN NORMS:",
      np.linalg.norm(out173[:, 48:96], axis=0).round(5).tolist())

saved_rw173 = rW
saved_x173 = repair_bank["x"]
try:
    rW = {n: torch.as_tensor(load(n), dtype=torch.float32) for n in NAMES}
    repair_bank["x"] = reference_x125.copy()
    capture_test173 = capture_transition102(np.asarray(indices102[:2], dtype=int))
finally:
    rW = saved_rw173
    repair_bank["x"] = saved_x173

print("\nEXACT CAPTURE RETURN STRUCTURE")
items173 = list(capture_test173) if isinstance(capture_test173, tuple) else [capture_test173]
for item173 in items173:
    if isinstance(item173, dict):
        print({k: (type(v).__name__, getattr(v, "shape", None)) for k, v in item173.items()})
    else:
        print(type(item173).__name__, getattr(item173, "shape", None))

print("\nACTION ENCODING FUNCTIONS")
text173 = []
for name173 in sorted(list(globals())):
    fn173 = globals()[name173]
    if inspect.isfunction(fn173) and any(t in name173.lower() for t in ["atom", "embed", "action", "encod"]):
        text173.append(f"{name173}{inspect.signature(fn173)} globals={fn173.__code__.co_names}")
print("\n".join(text173)[:2300])

print("\nLARGE DATAFRAME CANDIDATES")
printed173 = 0
for name173 in sorted(list(globals())):
    obj173 = globals()[name173]
    if isinstance(obj173, pd.DataFrame) and len(obj173) > 20000:
        print(name173, obj173.shape, obj173.columns.tolist()[:18])
        printed173 += 1
        if printed173 == 6:
            break
print("\nTRAIN COLUMNS:", pd.read_csv("/data/train.csv", nrows=1).columns.tolist())
print("\nSCHEMA:", pathlib.Path("/data/README.md").read_text()[:1500])

with open("/work/notes.md", "a") as log173:
    log173.write(
        "\n\n## Run 173\n"
        "Checked latent-map rank, reserved attention value/output directions, "
        "original training-bank IDs and exact capture/action interfaces for a "
        "shared-head NE channel. Existing saved weights remain unchanged.\n"
    )
```
