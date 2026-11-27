The edit uses two hidden units for each drug, with cancellation through the remaining 188 units. I’ll verify that I can reconstruct the installed checkpoint exactly, then screen modest amplitude changes without installing them. Lower cached NLL cost would identify candidates for rollout testing, not prove better causal behavior.

```python
import numpy as np
import pandas as pd
import torch

r246_keys = list(r235_safe)
r246_safe_np = {k: v.numpy().copy() for k, v in r235_safe.items()}
r246_reg = r240_regressions[(16, .001)]
r246_ins_rows = np.setdiff1d(r239_rows, np.array(r240_ne_rows))

def r246_candidate(ins_amp=.94, ne_amp=.50, cancel=1.0):
    w = {k: v.copy() for k, v in r246_safe_np.items()}
    kW, kb = "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"
    w[kW][:, r246_ins_rows] *= ins_amp / .94
    factor = ne_amp / .40
    w[kW][:, list(r240_ne_rows)] *= factor
    w[kW] -= (factor * cancel *
              np.outer(r240_q, r246_reg["beta"])).astype(np.float32)
    w[kb] -= (factor * cancel *
              r240_q * r246_reg["intercept"]).astype(np.float32)
    return w

r246_reconstruction = r246_candidate()
print("Reconstruction maximum errors:",
      {k: float(np.max(np.abs(v - load(k).numpy())))
       for k, v in r246_reconstruction.items()})
print("Insulin rows:", r246_ins_rows.tolist(),
      "NE rows:", list(r240_ne_rows))
print("NE output-pair match:",
      float(np.max(np.abs(r246_safe_np["attr_heads.value_bin.2.weight"][:, 48] - r240_q))),
      float(np.max(np.abs(r246_safe_np["attr_heads.value_bin.2.weight"][:, 171] + r240_q))))

r246_check = c210_eval(r246_reconstruction, c210_tune)
print("Evaluator output:",
      [(type(x).__name__, tuple(x.shape) if hasattr(x, "shape") else None)
       for x in r246_check] if isinstance(r246_check, tuple)
      else type(r246_check).__name__)
print("Train mask types:", r240_fit.dtype, r240_val.dtype)

def r246_loss(w, c):
    out = c210_eval(w, c)
    x = out[0] if isinstance(out, tuple) else out
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)

r246_orig = {k: load_original(k).numpy() for k in r246_keys}
r246_base = {
    "train": r246_loss(r246_orig, c210_train),
    "tune": r246_loss(r246_orig, c210_tune)
}
r246_candidates, r246_results = {}, []
if max(np.max(np.abs(v - load(k).numpy()))
       for k, v in r246_reconstruction.items()) < 2e-5:
    for ia in [.86, .90, .94]:
        for na in [.50, .60, .70]:
            w = r246_candidate(ia, na)
            r246_candidates[(ia, na)] = w
            dt = r246_loss(w, c210_train) - r246_base["train"]
            du = r246_loss(w, c210_tune) - r246_base["tune"]
            r246_results.append({
                "ins_amp": ia, "ne_amp": na,
                "train_fit": float(dt[r240_fit].mean()),
                "train_validation": float(dt[r240_val].mean()),
                "tune100": float(du.mean())
            })
    r246_table = pd.DataFrame(r246_results)
    print(r246_table.to_string(index=False))
else:
    print("Reconstruction failed; no screen and no edits installed.")

notes = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)(?=^## )", notes)
picked = [s for s in sections
          if any(w in s.lower() for w in
                 ("impulse response", "distributed lag", "persistence"))]
print("\nLong-horizon evidence:")
for s in picked[:3]:
    print(s[:650])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 246: amplitude reconstruction and screen\n"
            "Reconstructed current cancellation trial and screened independent "
            "insulin/NE output amplitudes. No weights installed. Cached fit "
            "is only a screening criterion; dose effects require rollout validation.\n")
    if r246_results:
        f.write(r246_table.to_string(index=False) + "\n")
```
