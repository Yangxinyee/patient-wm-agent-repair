The MAP trial failed decisively: factual NLL rose by about 3.61 nats per reading. Its small one-step distortion did not survive recurrent feedback. I’ll restore the validated glucose-only edit and inspect lower-norm MAP routes. Any replacement must pass full-sequence testing, not just local tests.

```python
import numpy as np, torch

r280_original = load_original("core.b_in.weight")
r280_before = np.asarray(load("core.b_in.weight")).copy()
with np.load("/work/map_feature_trial_backup.npz") as a:
    r280_backup = a["core_b_in"].astype(np.float32).copy()
save_edit("core.b_in.weight",
          torch.from_numpy(r280_backup) if isinstance(r280_original, torch.Tensor)
          else r280_backup)
print("RESTORED", edited(),
      "backup max error", float(np.max(np.abs(np.asarray(load("core.b_in.weight")) - r280_backup))))

def r280_matrix_report(label, matrix):
    x = np.asarray(matrix, dtype=np.float64)
    if x.shape == (192, 192):
        s = np.linalg.svd(x, compute_uv=False)
        print(label, {"norm": float(np.linalg.norm(x)),
                      "top_singular": np.round(s[:3], 5).tolist(),
                      "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(x))))})

r280_matrix_report("rejected current MAP addition", r280_before - r280_backup)
r280_matrix_report("restored glucose edit", r280_backup - np.asarray(r280_original))
for n in ["r272_trials", "r274_trials", "r276_trials", "r277_candidates", "r277_trial_weights"]:
    o = globals().get(n)
    print("\nOBJECT", n, type(o).__name__)
    if isinstance(o, dict):
        print("KEYS", list(o)[:15])
        for k, v in list(o.items())[:10]:
            if isinstance(v, dict):
                summary = {}
                for kk, vv in v.items():
                    if isinstance(vv, (float, int, str, bool, tuple)):
                        summary[kk] = vv
                    elif isinstance(vv, np.ndarray):
                        summary[kk] = {"shape": vv.shape, "norm": round(float(np.linalg.norm(vv)), 5)}
                print(k, summary)
            elif isinstance(v, np.ndarray):
                r280_matrix_report(str(k), v)
    elif isinstance(o, list):
        print("length", len(o), "first-entry type", type(o[0]).__name__ if o else "")
        if o:
            print("first-entry structure", repr(o[0])[:500])

print("\nRECENT HELPER OBJECTS")
for n in sorted(list(globals())):
    if n.startswith(("r267_", "r275_", "r276_", "r277_")):
        o = globals()[n]
        if callable(o):
            print(n, "callable")
        elif isinstance(o, np.ndarray):
            print(n, o.shape)
        elif isinstance(o, dict):
            print(n, "keys", list(o)[:8])
        elif isinstance(o, (float, int, str, tuple)):
            print(n, repr(o)[:140])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 280: reject unstable MAP feature route\n"
            "- Forward 26: edited factual NLL 5.24586 versus original 1.63155; catastrophic recurrent amplification.\n"
            "- Restored exact glucose-only strength-1.15 backup from map_feature_trial_backup.npz.\n"
            "- Local probability KL is insufficient for recurrent-edit eligibility; inspect lower-norm routes before retrying.\n"
            "- Python runs: 280; forward passes: 26.\n")
```