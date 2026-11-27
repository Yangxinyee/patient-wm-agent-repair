The record-based notes support an insulin effect, but the output was truncated before the current factual score and longer-horizon estimates appeared. I’ll retrieve those compactly. A factual increase below 0.005 is necessary before we can keep this edit.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import re

print("=== FACTUAL 34 ===")
_f270 = {}
for _model270 in ["original", "edited"]:
    _p270 = Path(f"/work/fwd_34/factual_{_model270}.csv")
    if _p270.exists():
        _df270 = pd.read_csv(_p270)
        _f270[_model270] = _df270
        print(_model270, "rows", len(_df270), "columns", list(_df270.columns))
        _cols270 = [c for c in _df270 if "nll" in c.lower()]
        print("means", _df270[_cols270].mean(numeric_only=True).to_dict())
        if "event" in _df270:
            print(_df270.groupby("event")[_cols270].mean().to_string())
if len(_f270) == 2:
    _o270, _e270 = _f270["original"], _f270["edited"]
    for _c270 in [c for c in _o270 if "nll" in c.lower() and c in _e270]:
        print("PAIRED DELTA", _c270, float(_e270[_c270].mean()-_o270[_c270].mean()))
        if len(_o270) == len(_e270) and "stay_id" in _o270:
            _tmp270 = pd.DataFrame({"stay_id": _o270.stay_id,
                                   "delta": _e270[_c270].to_numpy()-_o270[_c270].to_numpy()})
            _cl270 = _tmp270.groupby("stay_id").delta.agg(["sum", "count"])
            _ix270 = np.random.default_rng(270).integers(
                0, len(_cl270), size=(4000, len(_cl270)))
            _boot270 = (_cl270["sum"].to_numpy()[_ix270].sum(1) /
                        _cl270["count"].to_numpy()[_ix270].sum(1))
            print("stay-bootstrap delta percentiles",
                  np.quantile(_boot270, [.025, .5, .975]).round(6).tolist())

print("\n=== JOINT ARCHIVES ===")
for _fn270 in ["joint236.npz", "joint268.npz"]:
    with np.load("/work/"+_fn270, allow_pickle=False) as _a270:
        print(_fn270)
        for _k270 in _a270.files:
            _v270 = _a270[_k270]
            print(_k270, _v270.shape,
                  _v270.tolist() if _v270.size <= 6 else
                  "norm="+str(round(float(np.linalg.norm(_v270)), 5)))

print("\n=== RECORD-DERIVED TARGET NOTES ===")
_notes270 = Path("/work/notes.md").read_text()
_matches270 = [s.strip() for s in _notes270.splitlines()
               if re.search(r"target|truth|beta=|persistence=|causal estimate", s, re.I)]
print("\n".join(_matches270[:35])[:4300])

print("\n=== RELEVANT LIVE VARIABLES ===")
_keys270 = [k for k in globals() if not k.startswith("_") and
            any(s in k.lower() for s in
                ["target", "causal", "insulin", "joint", "blend", "scale", "install"])]
print(_keys270[:110])

with Path("/work/notes.md").open("a") as _log270:
    _log270.write(
        "\n## Run 270: compact factual and target audit\n"
        "Run269 was truncated before factual34. Retrieving current paired factual NLL, "
        "cluster-bootstrap uncertainty, saved joint contracts and record-derived targets. "
        "No weights changed. Python runs: 270; forwards: 35.\n")
```