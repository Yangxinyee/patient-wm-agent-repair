The cached predictions differ substantially from the GPU predictions, so broad CPU-based calibration would be unsafe. I’ll instead check where the actual scale-7 NLL penalty occurs. If much of it falls on hours without recent insulin, a record-fitted activity gate could remove unnecessary changes while preserving treatment responses.

```python
import numpy as np, pandas as pd
from pathlib import Path

_gate_train_ids = set(pd.concat([
    sc_fact["first"]["meta"]["stay_id"],
    sc_fact["extra"]["meta"]["stay_id"]
]).unique())

def _read_gate_drugs(_path, _ids=None):
    _parts = []
    for _chunk in pd.read_csv(_path, usecols=["stay_id", "hour", "event", "rate"], chunksize=200000):
        _mask = _chunk["event"].isin(["insulin", "norepinephrine"])
        if _ids is not None:
            _mask &= _chunk["stay_id"].isin(_ids)
        _parts.append(_chunk.loc[_mask].copy())
    _drugs = pd.concat(_parts, ignore_index=True)
    _drugs["outcome_hour"] = np.floor(_drugs["hour"] + 1e-5).astype(int) + 1
    return _drugs.pivot_table(
        index=["stay_id", "outcome_hour"], columns="event", values="rate", aggfunc="sum"
    ).fillna(0).reset_index().rename(columns={"outcome_hour": "hour"})

_gate_train_drugs = _read_gate_drugs("/data/train.csv", _gate_train_ids)
_gate_tune_drugs = _read_gate_drugs("/data/tuning.csv")

def _attach_gate_drugs(_meta, _drug_table):
    _out = _meta.merge(_drug_table, on=["stay_id", "hour"], how="left", validate="many_to_one")
    for _drug in ["insulin", "norepinephrine"]:
        if _drug not in _out:
            _out[_drug] = 0.0
        _out[_drug] = _out[_drug].fillna(0.0)
    return _out

_gate_meta = {
    k: _attach_gate_drugs(v["meta"], _gate_tune_drugs if k == "tuning" else _gate_train_drugs)
    for k, v in sc_fact.items()
}
for _k, _m in _gate_meta.items():
    print(_k, pd.crosstab(_m["event"], _m["insulin"] > 0).to_string())

print("\nACTUAL SCALE-7 FACTUAL COST BY RECENT INSULIN")
for _k in [24, 26]:
    _o = pd.read_csv(f"/work/fwd_{_k}/factual_original.csv")
    _e = pd.read_csv(f"/work/fwd_{_k}/factual_edited.csv")
    _paired = _o.merge(_e, on=["stay_id", "hour", "event"], suffixes=("_o", "_e"), validate="one_to_one")
    _paired = _attach_gate_drugs(_paired, _gate_tune_drugs)
    _paired["delta_nll"] = _paired["nll_e"] - _paired["nll_o"]
    print("forward", _k)
    print(_paired.groupby(["event", _paired["insulin"] > 0])["delta_nll"].agg(["size", "mean"]).to_string())

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Calibration-cache audit; Python 230, forwards 26/50\n"
             "- CPU tuning cache matches GPU reading keys, but glucose mean RMSE is 8.83 mg/dL and CPU finger NLL exceeds GPU by 0.163 nats.\n"
             "- Broad calibration fitted to these approximate probabilities is not trusted.\n"
             "- Partitioning actual scale-7 GPU cost by recorded prior-hour insulin before considering an activity gate.\n"
             "- No weights changed; stronger6 remains installed.\n")
```