The architecture confirms that only the reading-value head is currently edited. The recovery output was truncated before the safety results, so I’ll retrieve those compactly. A comfortably sub-0.005 factual-loss increase would support keeping this candidate; a result near the limit would require more caution.

```python
import numpy as np, pandas as pd, re
from pathlib import Path

print("=== NOTEBOOK END ===")
print(Path("/work/notes.md").read_text()[-1700:])

print("\n=== EFFECT ESTIMATES ===")
for fn287 in ["kernel_targets.csv", "dml_results.csv", "dynamic_iv.csv"]:
    p287 = Path("/work") / fn287
    d287 = pd.read_csv(p287)
    print(fn287, d287.shape, list(d287.columns))
    print(d287.head(6).to_string(index=False))

print("\n=== SAVED CANDIDATES ===")
for fn287 in ["calibrated_backup285.npz", "safety_candidate285.npz"]:
    with np.load("/work/" + fn287, allow_pickle=False) as a287:
        print(fn287, [(k, a287[k].shape) for k in a287.files])
        for k287 in a287.files:
            if a287[k287].ndim == 0:
                print(k287, a287[k287].item())

print("\n=== RECENT FACTUAL FILES ===")
for n287 in [39, 40, 41, 42]:
    p287 = Path(f"/work/fwd_{n287}")
    if p287.exists():
        print(p287.name, sorted(x.name for x in p287.iterdir()))
        for q287 in p287.glob("factual_*.csv"):
            d287 = pd.read_csv(q287)
            print(q287.name, d287.shape, list(d287.columns))
            print(d287.head(1).to_string(index=False))
            for c287 in d287.columns:
                if "nll" in c287.lower():
                    print(c287, float(d287[c287].mean()))

print("\n=== RELEVANT FUNCTIONS / VARIABLES ===")
print("functions:", sorted(k for k,v in list(globals().items())
      if callable(v) and getattr(v,"__module__","") == "__main__"
      and re.search(r"build|score|restore|install|cal|head|candidate|param|predict", k))[:65])
print("variables:", sorted(k for k in list(globals())
      if re.search(r"285|284|283|tuning_ids|tune_ids|stay_ids|safety", k))[:65])

t287 = pd.read_csv("/data/tuning.csv", nrows=2)
print("\nTuning columns:", list(t287.columns))
with open("/work/notes.md", "a") as f287:
    f287.write("\n## Run 287: bounded safety recovery\n"
               "- Previous recovery output was truncated; reading compact candidate, effect-estimate, and factual summaries.\n"
               "- No weights changed. Counts: 287 Python runs, 40 forward passes.\n")
```