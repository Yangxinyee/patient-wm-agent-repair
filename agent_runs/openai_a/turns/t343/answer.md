The 0.93-amplitude edit increased factual NLL by 0.00397 on these 100 stays, below the 0.005 limit and less than the sharper full-amplitude edit. That is encouraging but not enough to accept it. I’ll recover the causal estimates and validation batches next; acceptance requires broader factual safety and retained rollout effects.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD DOCUMENTATION ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK: IDENTIFICATION AND TARGETS ===")
print(notes_text[:10500])

print("=== CHECKPOINTS AND VALIDATION VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if any(s in k.lower() for s in
           ("checkpoint", "tuning_ids", "tune_ids", "stay_ids",
            "causal", "target_effect", "iv_result", "rd_result")):
        detail = ""
        if isinstance(v, dict):
            detail = " keys=" + str(list(v.keys())[:12])
        elif hasattr(v, "shape"):
            detail = " shape=" + str(v.shape)
        elif isinstance(v, (list, tuple)):
            detail = " length=" + str(len(v)) + " first=" + str(v[:5])
        print(k, type(v).__name__, detail)

factual41_original = pd.read_csv("/work/fwd_41/factual_original.csv")
factual41_edited = pd.read_csv("/work/fwd_41/factual_edited.csv")
print("=== FORWARD 41 COLUMNS AND SAMPLE ===")
print(factual41_original.columns.tolist())
print(factual41_original.head(2).to_string(index=False))
print("=== TUNING RECORD COLUMNS ===")
print(pd.read_csv("/data/tuning.csv", nrows=2).to_string(index=False))

amplitude093_checkpoint = {
    n: np.asarray(load(n), dtype=np.float32).copy() for n in edited()
}
print("Protected current 0.93-amplitude checkpoint:", list(amplitude093_checkpoint))
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Amplitude093 first factual validation; Python 288, forwards 41/50\n"
        "- First100 factual NLL increase +0.00397; lower than sharp240's +0.00476.\n"
        "- Current four tensors protected as amplitude093_checkpoint.\n"
        "- Candidate remains provisional: independent factual and rollout validation required.\n"
        "- Recovering record-derived effect targets and validation batches; no weights changed.\n"
    )
```