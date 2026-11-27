The fourth batch brings pooled NLL degradation to roughly +0.00425, within the observed limit but with limited margin. I’ll compute the exact paired result and stay-level uncertainty, and recover the record-effect estimates. An upper uncertainty bound above +0.005 would favor a safer, smaller edit.

```python
from pathlib import Path
import numpy as np
import pandas as pd

paired287 = []
for k in [32, 34, 35, 36]:
    orig = pd.read_csv(f"/work/fwd_{k}/factual_original.csv")
    edit = pd.read_csv(f"/work/fwd_{k}/factual_edited.csv")
    assert len(orig) == len(edit)
    nllcols = [c for c in orig.columns if "nll" in c.lower()]
    assert len(nllcols) == 1, (k, orig.columns.tolist(), nllcols)
    nllcol = nllcols[0]
    staycol = next(c for c in orig.columns if "stay" in c.lower())
    for c in ["stay_id", "hour", "event", "code", "value", "event_index", "bundle_index"]:
        if c in orig.columns and c in edit.columns:
            assert orig[c].equals(edit[c]), (k, c)
    part = orig.copy()
    part["paired_delta"] = edit[nllcol].to_numpy() - orig[nllcol].to_numpy()
    part["forward"] = k
    paired287.append(part)
    print("batch", k, "readings", len(part),
          "exact_delta", float(part["paired_delta"].mean()))

paired287 = pd.concat(paired287, ignore_index=True)
assert paired287[staycol].nunique() == 400
stay287 = paired287.groupby(staycol)["paired_delta"].agg(["sum", "count"])
rng287 = np.random.default_rng(287)
ix287 = rng287.integers(0, len(stay287), size=(10000, len(stay287)))
boot287 = (
    stay287["sum"].to_numpy()[ix287].sum(axis=1)
    / stay287["count"].to_numpy()[ix287].sum(axis=1)
)
pooled287 = float(paired287["paired_delta"].mean())
ci287 = np.quantile(boot287, [0.025, 0.5, 0.95, 0.975])
print("\nPOOLED", {"delta": pooled287, "n_readings": len(paired287),
                    "bootstrap_quantiles_025_50_95_975": ci287.tolist()})
eventcol287 = next(
    (c for c in ["event", "code", "event_code", "event_name"] if c in paired287.columns),
    None
)
if eventcol287:
    print("BY EVENT")
    print(paired287.groupby(eventcol287)["paired_delta"].agg(["mean", "count"]).to_string())
print("CSV COLUMNS", paired287.columns.tolist())

current287 = load("core.b_in.weight")
if hasattr(current287, "detach"):
    current287 = current287.detach().cpu().numpy()
with np.load("/work/balanced_candidate.npz") as archive287:
    assert np.array_equal(np.asarray(current287), archive287["core_b_in"])
np.savez(
    "/work/balanced_validated400.npz",
    core_b_in=np.asarray(current287),
    pooled_value_nll_delta=pooled287,
    bootstrap_quantiles=ci287,
)
paired287.to_csv("/work/balanced_paired400.csv", index=False)

print("\nEARLY RECORD-IDENTIFICATION NOTES")
print(Path("/work/notes.md").read_text()[:5800])
print("\nCOMPACT RETAINED TARGETS")
for name in ["effects63", "target56", "target88", "r244_targets",
             "factual_missing_effect0", "r180_effects"]:
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, np.ndarray) and obj.size <= 24:
            print(name, obj.tolist())
        elif isinstance(obj, dict):
            print(name, repr(obj)[:1100])
print("RUN-285 VARIABLES",
      sorted(n for n in globals() if "285" in n)[:45])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 287: balanced candidate, complete factual validation\n"
        f"- Forwards 32/34/35/36: paired 400-stay value-NLL delta {pooled287:.9f}, "
        f"{len(paired287)} readings; bootstrap quantiles "
        f"[2.5%,50%,95%,97.5%]={ci287.tolist()}.\n"
        "- Forward 33 predictions: insulin x2/half -11.712/+5.178; "
        "NE x2/half +0.926/-0.311. MAP remains substantially under-responsive "
        "relative to retained record estimates.\n"
        "- Archived exact current weights and pooled validation; no weights changed.\n"
        "- Python runs: 287; forward passes: 36.\n"
    )
```