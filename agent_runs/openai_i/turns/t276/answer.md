```python
from pathlib import Path
import json, re, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
arch246 = Path("/model/ARCHITECTURE.md").read_text()
print(arch246)

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== ACTUAL EDITS ===")
ed246 = list(edited())
print(ed246)
for n in ed246:
    a, b = np.asarray(load(n)), np.asarray(load_original(n))
    print(n, a.shape, "difference_norm", float(np.linalg.norm(a-b)))

print("\n=== MOST RECENT FORWARD FILES ===")
for k in (20, 21, 22):
    p = Path(f"/work/fwd_{k}")
    if not p.exists():
        continue
    for f in sorted(p.iterdir()):
        print(str(f), f.stat().st_size)
        if f.suffix in (".json", ".txt"):
            print(f.read_text()[:5000])
        elif f.suffix == ".csv":
            d = pd.read_csv(f)
            print("shape", d.shape, "columns", d.columns.tolist())
            print(d.head(3).to_string(index=False))
            for col in d.columns:
                if "nll" in col.lower() and pd.api.types.is_numeric_dtype(d[col]):
                    print(col, "mean", float(d[col].mean()))

print("\n=== SAVED CANDIDATE CONTENTS ===")
for fn in ("neutral209.npz", "small_route222.npz", "response233.npz", "sharp243.npz"):
    p = Path("/work") / fn
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            print(fn, [(k, z[k].shape) for k in z.files])
            common = [k for k in z.files if k in NAMES]
            if common:
                print("max difference from installed tensors",
                      max(float(np.max(np.abs(z[k]-np.asarray(load(k))))) for k in common))

print("\n=== EARLY RECORD EVIDENCE / ACCEPTED CANDIDATES ===")
notes246 = Path("/work/notes.md").read_text()
print(notes246[:14000])
sections246 = re.split(r"(?=^## )", notes246, flags=re.M)
for s in sections246:
    heading = s.splitlines()[0] if s.splitlines() else ""
    if any(re.search(rf"\b{r}\b", heading) for r in (184,185,186,187,188,189,190,203,204,209,210,211,212,213,214)):
        print(s[:4500])

print("\n=== RETAINED HELPERS / STAY LISTS ===")
for name, value in list(globals().items()):
    if name.startswith("__"):
        continue
    if ("tuning" in name.lower() or "tune" in name.lower() or "replay" in name.lower()
            or "readout" in name.lower() or "candidate" in name.lower()):
        if isinstance(value, np.ndarray):
            print(name, "array", value.shape, value.dtype)
        elif isinstance(value, pd.DataFrame):
            print(name, "DataFrame", value.shape, value.columns.tolist()[:12])
        elif isinstance(value, (list, tuple, dict)):
            print(name, type(value).__name__, "length", len(value),
                  "preview", str(value if len(value) < 8 else list(value)[:8])[:350])
        elif callable(value):
            print(name, "callable")

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 246 — restart consistency audit\n"
            "- Run245 failed only in variable introspection; it did not validate or alter a candidate.\n"
            "- Reading architecture, record schema, actual edited tensors, saved candidates, and latest forward files.\n"
            "- No new weights changed. Candidate acceptance still requires matched factual dNLL <= .005.\n")
```