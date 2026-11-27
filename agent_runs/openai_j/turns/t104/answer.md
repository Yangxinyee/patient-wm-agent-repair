The coefficient conversion is correct. The apparent mismatch came from applying an already-converted coefficient to standardized features, so that explanation is refuted. I’ll now compare the reconstructed head with cached GPU predictions. Close agreement would shift attention to generalization; large systematic errors would mean the reconstruction itself must be fixed.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("=== HEAD DATA ===")
for name in ("head_meta", "head_factual", "repair_sf"):
    obj91 = globals()[name]
    print(name, obj91.shape, list(obj91.columns))
    print(obj91.head(2).to_string(index=False))

print("\n=== CACHED FACTUAL OUTPUTS ===")
audit_factual91 = {}
for p91 in sorted(Path("/work").glob("fwd_*/factual_original.csv")):
    df91 = pd.read_csv(p91)
    audit_factual91[str(p91)] = df91
    print(str(p91), df91.shape, list(df91.columns))
    nllcols91 = [c for c in df91.columns if "nll" in c.lower()]
    print("NLL means:", {c: float(df91[c].mean()) for c in nllcols91})
    print(df91.head(1).to_string(index=False))

print("\n=== RECONSTRUCTION NOTES ===")
nt91 = Path("/work/notes.md").read_text()
for title in ("Head reconstruction setup", "Reconstruction diagnostics",
              "Restart diagnostics", "Run 38: readout vs"):
    start91 = nt91.find("## " + title)
    if start91 >= 0:
        end91 = nt91.find("\n## ", start91 + 4)
        print(nt91[start91:end91 if end91 >= 0 else len(nt91)][:1100])

print("\n=== LOCAL RECONSTRUCTED BASELINES ===")
for label91, P91 in (("repair_fP", repair_fP), ("repair_base_P", repair_base_P),
                     ("head_probs", head_probs)):
    P91 = np.asarray(P91)
    mean91 = (P91 * repair_C17).sum(1)
    nll91 = -np.log(np.clip(P91[np.arange(len(P91)), repair_fbins], 1e-30, 1))
    print(label91, "NLL", nll91.mean(), "mean quantiles",
          np.quantile(mean91, [0, .1, .5, .9, 1]))
print("fP/base_P probability RMS:",
      np.sqrt(np.mean((np.asarray(repair_fP)-np.asarray(repair_base_P))**2)))
print("head_recovery keys:", list(head_recovery))
print("ACTIVE EDITS:", edited())
```