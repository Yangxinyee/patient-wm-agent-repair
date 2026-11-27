The architecture has several normalisation layers, which may explain why increasing numeric dose gain from 4 to 16 barely changed the rollouts. I’ll compare the paired states to find where dose information survives and whether it occupies a low-variance direction. Such a direction could offer a more selective edit; if none exists, broad amplification is unlikely to be safe.

```python
import json
from scipy.linalg import eigh

text = Path("/work/notes.md").read_text()
start = text.find("- Nonlinear cross-fitted estimates:")
end = text.find("## Run 6")
print("=== IDENTIFICATION DETAILS ===")
print(text[start:end][:4300])

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in [
        "encode_action", "core.c_out", "core.b_in", "core.out_norm",
        "core.coda", "core.z_", "attr_heads.value_bin", "last_numeric"
    ]):
        print(name, SHAPES[name])

print("\n=== KERNEL VARIABLE NAMES ===")
print(" ".join(n for n in globals() if not n.startswith("_"))[:2200])

paired_o = np.load("/work/fwd_13/states_original.npz")
paired_e = np.load("/work/fwd_13/states_edited.npz")
print("\nNPZ arrays:", paired_o.files)
xo = paired_o["x"].astype(np.float64)
xe = paired_e["x"].astype(np.float64)
assert xo.shape == xe.shape
for k in ["stay_id", "hour", "has_insulin", "has_norepinephrine"]:
    assert np.array_equal(paired_o[k], paired_e[k])

drug_masks = {
    "insulin_only": paired_o["has_insulin"].astype(bool) & ~paired_o["has_norepinephrine"].astype(bool),
    "ne_only": paired_o["has_norepinephrine"].astype(bool) & ~paired_o["has_insulin"].astype(bool),
    "both": paired_o["has_insulin"].astype(bool) & paired_o["has_norepinephrine"].astype(bool),
    "no_drug": ~paired_o["has_insulin"].astype(bool) & ~paired_o["has_norepinephrine"].astype(bool)
}
state_geom = {}
print("\n=== PAIRED-STATE GEOMETRY ===")
for label, sl in [("out", slice(0,192)), ("h0", slice(192,384)),
                  ("h1", slice(384,576)), ("h2", slice(576,768)),
                  ("h3", slice(768,960)), ("z", slice(960,992))]:
    X = xo[:, sl]
    D = xe[:, sl] - X
    C = X.T @ X / len(X)
    Q = D.T @ D / len(D)
    reg = max(np.trace(C) / C.shape[0] * 1e-4, 1e-8)
    vals, vecs = eigh(Q, C + reg*np.eye(C.shape[0]))
    v = vecs[:, -1]
    v /= np.linalg.norm(v)
    state_geom[label] = {"direction": v, "ratio": float(vals[-1])}
    result = {
        "factual_rms": round(float(np.sqrt(np.mean(X*X))), 5),
        "delta_rms": round(float(np.sqrt(np.mean(D*D))), 5),
        "top_delta/factual_ratio": round(float(vals[-1]), 5),
        "projection_factual_rms": round(float(np.sqrt(np.mean((X@v)**2))), 5),
        "projection_delta_rms": round(float(np.sqrt(np.mean((D@v)**2))), 5),
        "delta_rms_by_bundle": {
            k: round(float(np.sqrt(np.mean(D[m]**2))), 5) if m.any() else None
            for k, m in drug_masks.items()
        }
    }
    print(label, result)

print("Bundle counts:", {k: int(m.sum()) for k,m in drug_masks.items()})
print("Stay IDs:", np.unique(paired_o["stay_id"])[:6])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 21: paired-state geometry\n")
    f.write("- Compared gain-4 and original states using generalized delta/factual second moments.\n")
    f.write("- Ratios: " + repr({k: v["ratio"] for k,v in state_geom.items()}) + "\n")
    f.write("- Python runs: 21; forward passes: 13.\n")
```