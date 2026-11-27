The estimated effects persist for several hours, so repairing only the next reading would be insufficient. The retained arrays now allow a direct check of the head correction’s units and its predicted versus realized response. I’ll also inspect the reserved latent channels to see whether they carry dose information that a better route could use.

```python
from pathlib import Path
import re, numpy as np, types, dis, io

print("=== HEAD FIT: UNITS AND REALIZED SIGNAL ===")
print("effects:", {k: np.asarray(v).tolist() for k, v in r180_effects.items()})
print("dose contrast by drug:", {
    str(d): {"n": int(np.sum(r179_drug == d)),
             "mean": float(np.mean(r179_dc[r179_drug == d])),
             "sd": float(np.std(r179_dc[r179_drug == d]))}
    for d in np.unique(r179_drug)})
for sensor184 in r180_D:
    m0184 = np.asarray(r179_head0[sensor184]["M"]).reshape(-1)
    mc184 = np.asarray(r179_headc[sensor184]["M"]).reshape(-1)
    y184 = np.asarray(r180_y[sensor184]).reshape(-1)
    p184 = r180_D[sensor184] @ r180_best["w"]
    print(sensor184,
          "baseline mean shift", float(np.mean(mc184-m0184)),
          "target RMS", float(np.sqrt(np.mean(y184*y184))),
          "linear correction RMS", float(np.sqrt(np.mean(p184*p184))),
          "target/fit corr", float(np.corrcoef(y184, p184)[0,1]),
          "centres range", [float(np.min(r179_head0[sensor184]["C"])),
                            float(np.max(r179_head0[sensor184]["C"]))])
print("\n=== HELPER DEPENDENCIES ===")
for name184 in ["r180_fit", "r180_apply", "r179_sensitivity", "r59_core",
                "r49_readouts", "rr35_scaled_action_encoding", "repaired_core_local", "lin0"]:
    fn184 = globals()[name184]
    print(name184, "args", fn184.__code__.co_varnames[:fn184.__code__.co_argcount],
          "globals", fn184.__code__.co_names)
print("\n=== RESERVED CHANNEL RECIPE ===")
for sec184 in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    title184 = sec184.splitlines()[0] if sec184.splitlines() else ""
    if re.search(r"\b(79|80|81|82|89|90|91|128)\b", title184):
        print(sec184[:500])
print("\n=== LATENT EDIT SIZES ===")
for name184 in edited():
    a184 = np.asarray(load(name184))
    b184 = np.asarray(load_original(name184))
    delta184 = a184-b184
    print(name184, "shape", a184.shape, "maxdelta", float(np.max(np.abs(delta184))),
          "delta-rank" if delta184.ndim == 2 else "nonzero",
          int(np.linalg.matrix_rank(delta184)) if delta184.ndim == 2 else int(np.count_nonzero(delta184)))
print("channel-related globals:",
      [n for n in sorted(globals()) if any(t in n.lower() for t in
       ["candidates90", "token", "reserved", "dose_decoder", "channel", "directions", "direction"])][:65])
with open("/work/notes.md", "a") as f184:
    f184.write("\n## Run 184: head signal and reserved-channel audit\n"
               "- Dynamic record estimates show multi-hour persistence, requiring a recurrent or feedback-compatible repair.\n"
               "- Checking head-fit units, helper dependencies, and saved latent-channel construction.\n"
               "- Python runs: 184; forward passes: 10. No saved changes.\n")
```