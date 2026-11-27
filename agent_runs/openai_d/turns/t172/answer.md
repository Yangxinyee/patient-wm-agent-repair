The insulin-only edit passes this check, but narrowly: its NLL increase is about 0.00484. The norepinephrine trial added roughly another 0.00244, so simply amplifying it is unsafe. I’ll examine the factual error by correction score and event; systematic bias or excess variance would suggest a more targeted repair.

```python
import pandas as pd
from scipy.special import ndtr

r150_o = pd.read_csv("/work/fwd_18/factual_original.csv")
r150_i = pd.read_csv("/work/fwd_18/factual_edited.csv")
r150_c = pd.read_csv("/work/fwd_16/factual_edited.csv")
_keys = ["stay_id", "hour", "event"]
r150_join = r150_o.merge(r150_i, on=_keys, suffixes=("_original", "_insulin"))
r150_join = r150_join.merge(r150_c[_keys + ["predicted_mean", "nll"]], on=_keys)
r150_join["insulin_loss"] = r150_join.nll_insulin - r150_join.nll_original
r150_join["NE_extra_loss"] = r150_join.nll - r150_join.nll_insulin
r150_join["insulin_mean_shift"] = r150_join.predicted_mean_insulin - r150_join.predicted_mean_original
r150_join["NE_extra_mean_shift"] = r150_join.predicted_mean - r150_join.predicted_mean_insulin
print("=== HELD-OUT FACTUAL CONTRIBUTIONS ===")
print(r150_join.groupby("event")[["insulin_loss", "NE_extra_loss",
                                  "insulin_mean_shift", "NE_extra_mean_shift"]].agg(["mean", "std"]))
print("stay-level average losses:")
print(r150_join.groupby("stay_id")[["insulin_loss", "NE_extra_loss"]].mean().describe())

print("=== CACHED GATED CORRECTIONS ===")
for _label, _mask in [("all", np.ones(len(r144_events), bool)),
                     ("glucose_finger", r144_events == "glucose_finger"),
                     ("map_arterial", r144_events == "map_arterial"),
                     ("insulin_active", r144_active),
                     ("NE_active", r132_active)]:
    _mask = np.asarray(_mask, dtype=bool)
    if _mask.shape != (len(r144_events),):
        print(_label, "mask shape", _mask.shape)
        continue
    print(_label, "n", int(_mask.sum()),
          "D", np.round(np.quantile(r144_D[:, _mask], [.01, .5, .99]), 4),
          "ins_gate", np.round(np.quantile(r144_G[:, _mask], [.01, .5, .99]), 3),
          "NE_gate", np.round(np.quantile(r138_G[:, _mask], [.01, .5, .99]), 3))
print("NE feature keys:", r139_feature.keys())
print("NE paired score quantiles:", np.quantile(r139_feature["pair"], [0, .01, .5, .99, 1]))
print("insulin tilt:", np.round(r145_Cins, 5))
print("NE tilt:", np.round(r138_tilt, 5))
print("active residual labels:")
print(aligned99.loc[np.asarray(r144_active, bool),
                    ["dose", "dose_expected", "rd", "ry"]].describe())
print("candidate construction scalars:")
for _n in ["theta104", "alpha104", "gb104", "db104", "gate_scale104",
           "gate_truth99", "gate_target99", "dv104", "gv104"]:
    if _n in globals():
        _o = globals()[_n]
        print(_n, (np.quantile(_o, [0, .5, 1]) if hasattr(_o, "shape")
                   else repr(_o)))

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Forward 18 / Run 150: insulin fallback passes narrowly\n"
             "On tuning 300..399, strength-2 insulin factual NLL is 1.64733 vs original 1.64249 "
             "(delta approximately +0.00484). NE trial's extra factual loss is approximately +0.00244. "
             "Need better factual efficiency before scaling either correction. Insulin fallback remains installed.\n")
```
