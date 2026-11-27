The recovered insulin helper uses a shared gate with equal-and-opposite innovation terms. I’ll check whether norepinephrine has compatible output columns, then screen a few analogous constructions locally without saving them. A factual-safe candidate would justify a GPU rollout test; a large NLL penalty would reject it.

```python
import numpy as np
import torch

r271_base = {n: clone258(x) for n, x in backup269.items()}
r271_w = a262(r271_base["z_post.0.weight"])
r271_b = a262(r271_base["z_post.0.bias"])
r271_o = a262(r271_base["z_post.2.weight"])
r271_rows = (156, 89)

print("NE PAIR GEOMETRY")
for j in r271_rows:
    print(j, "bias", float(r271_b[j]),
          "ctx/e/u norms",
          [float(np.linalg.norm(r271_w[j, k:k+192]))
           for k in (0, 192, 384)],
          "output norm", float(np.linalg.norm(r271_o[:, j])))
r271_antipodal = float(
    np.linalg.norm(r271_o[:, 156] + r271_o[:, 89]) /
    max(np.linalg.norm(r271_o[:, 156]), 1e-12)
)
print("RELATIVE OUTPUT ANTIPODAL ERROR", r271_antipodal)
print("REPLAY DEPENDENCIES", replay_factual125.__code__.co_names)
print("DECODER DEPENDENCIES", cpu_factual129.__code__.co_names)

def symmetric_ne271(gain, gate_shift=0.0, ins_gain=1.5, ins_gate=26.0):
    c = symmetric_ins265(ins_gain, ins_gate)
    w = a262(c["z_post.0.weight"]).copy()
    b = a262(c["z_post.0.bias"]).copy()
    gw, gb = w[89].copy(), float(b[89]) - gate_shift
    qw, qb = w[156] - w[89], float(b[156] - b[89])
    w[156], w[89] = gw + gain * qw / 2, gw - gain * qw / 2
    b[156], b[89] = gb + gain * qb / 2, gb - gain * qb / 2
    c["z_post.0.weight"] = like258(w, c["z_post.0.weight"])
    c["z_post.0.bias"] = like258(b, c["z_post.0.bias"])
    return c

r271_results = {}
r271_candidates = {}
r271_snapshots = [
    (obj, dict(obj)) for name, obj in list(globals().items())
    if isinstance(obj, dict) and "z_post.0.weight" in obj
]
try:
    if r271_antipodal < 1e-5:
        for g, t in [(0.6, 0.0), (0.8, 0.0), (1.0, 0.0), (1.25, 4.0)]:
            c = symmetric_ne271(g, t)
            _, losses, _ = factual_eval194(c)
            nll = float(np.mean(losses))
            r271_results[(g, t)] = {
                "nll": nll, "delta": nll - 1.6355364322662354
            }
            r271_candidates[(g, t)] = c
        print("UNSAVED NE SYMMETRY SCREEN", r271_results)
    else:
        print("No screen: NE output columns are not an opposite pair.")
finally:
    for obj, snapshot in r271_snapshots:
        obj.clear()
        obj.update(snapshot)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 271\n"
        "Checked NE rows 156/89 and screened unsaved shared-gate, symmetric "
        "innovation candidates where output-column geometry permits. "
        f"Antipodal error {r271_antipodal:.6g}; CPU results {r271_results}. "
        "All model-weight dictionaries restored; submitted weights unchanged. "
        "These tests address model transmission, not ICU-effect identification.\n"
    )
```