The cached head contrast came from a 1.5× numeric-embedding probe, not the later 5× probe. Its separation from factual features is modest, so a large correction could harm fit. I’ll rebuild the matched 5× features and verify the normalization before fitting anything.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("NUMERIC VOCAB", str(VOCAB["numeric"])[:2300])
print("\nFORWARD 11 FILES", [p.name for p in Path("/work/fwd_11").iterdir()])
print("FORWARD 4 FILES", [p.name for p in Path("/work/fwd_4").iterdir()])
print("\nRECORD DRUG ROWS")
print(ds_records.loc[ds_records.event.isin(["insulin","norepinephrine"])].head(3).to_string(index=False))

p188 = Path("/work/fwd_11/states_edited.npz")
if p188.exists():
    pair5_s188 = dict(np.load(p188))
    pair0_s188 = rr_state4
    print("\nSTATE MATCH",
          pair0_s188["x"].shape, pair5_s188["x"].shape,
          "stay equal", np.array_equal(pair0_s188["stay_id"], pair5_s188["stay_id"]),
          "hour equal", np.array_equal(pair0_s188["hour"], pair5_s188["hour"]))
    if (np.array_equal(pair0_s188["stay_id"], pair5_s188["stay_id"]) and
        np.array_equal(pair0_s188["hour"], pair5_s188["hour"])):
        pair_records188 = ds_records.loc[
            ds_records.stay_id.isin(np.unique(pair5_s188["stay_id"]))].copy()
        pair0_head188 = ce_make_heads(pair_records188, pair0_s188)
        pair5_head188 = ce_make_heads(pair_records188, pair5_s188)
        pair0_phi188 = np185(pair0_head188["phi"]).astype(np.float64)
        pair5_phi188 = np185(pair5_head188["phi"]).astype(np.float64)
        print("HEAD MATCH", pair0_head188["rows"].equals(pair5_head188["rows"]),
              pair0_phi188.shape,
              "contrast RMS", float(np.sqrt(np.mean((pair5_phi188-pair0_phi188)**2))))
        print("rows", pair0_head188["rows"].head(2).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 187–188\nRun 187 inspected 1.5x (not 5x) cached head contrasts; "
            "largest contrast/factual generalized sensitivity was 0.05786. "
            "Run 187 stopped on list/dict mismatch before any edits. "
            "Run 188 checks normalization and reconstructs matched 5x head features. No edits.\n")
```