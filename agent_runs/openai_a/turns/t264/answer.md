The caches contain factual readings from two training groups and one tuning group, plus separate insulin-dose contrasts. Before fitting calibration, I’ll compare their predictions with the actual GPU factual output. Close agreement would support using them for calibration; systematic disagreement would make such an edit risky.

```python
import inspect, numpy as np, pandas as pd, torch
from pathlib import Path

for _n in ["selective_cache", "sc_evaluate", "sc_cache_inputs"]:
    _f = globals()[_n]
    print(_n, inspect.signature(_f), "names:", _f.__code__.co_names)
print("nodes:", sc_nodes, "rank:", np.round(sc_rank, 3))
print("fit:", {k: (tuple(v.shape) if hasattr(v, "shape") else v)
               for k, v in repair224_fit.items() if k in ["w", "b", "scale"]})

_cal_cache = sc_fact["tuning"]
_cal_meta = _cal_cache["meta"].copy()
print("META COLUMNS:", list(_cal_meta.columns))
print(_cal_meta.head(2).to_string(index=False))

_cal_probs = torch.softmax(torch.from_numpy(_cal_cache["logits"]), -1).mean(0).numpy()
_cal_nll = -np.log(np.maximum(_cal_probs[np.arange(len(_cal_meta)), _cal_cache["bins"]], 1e-30))
print("cached logits mean NLL:", _cal_nll.mean(), "original_nll:", _cal_cache["original_nll"])

_cal_gpu = pd.read_csv("/work/fwd_24/factual_original.csv")
_cal_meta["cpu_original_mean"] = _cal_cache["original_mean"]
_cal_meta["cpu_base_mean"] = _cal_cache["base_mean"]
_cal_meta["cpu_base_nll"] = _cal_nll
_cal_join = _cal_meta.merge(_cal_gpu, on=["stay_id", "hour", "event"], how="inner", suffixes=("_cache", "_gpu"))
print("MATCHED:", len(_cal_join), "of", len(_cal_meta), "CPU and", len(_cal_gpu), "GPU readings")
for _event, _group in _cal_join.groupby("event"):
    print(_event, "n", len(_group),
          "mean discrepancy", round(float((_group["cpu_original_mean"] - _group["predicted_mean"]).mean()), 4),
          "mean RMSE", round(float(np.sqrt(np.mean((_group["cpu_original_mean"] - _group["predicted_mean"])**2))), 4),
          "NLL discrepancy", round(float((_group["cpu_base_nll"] - _group["nll"]).mean()), 5),
          "GPU outcome-minus-mean", round(float((_group["observed"] - _group["predicted_mean"]).mean()), 4))

print("\nCURRENT GPU FACTUAL RESIDUALS")
_cal_edited = pd.read_csv("/work/fwd_24/factual_edited.csv")
for _event, _g in _cal_edited.groupby("event"):
    print(_event, "residual mean", round(float((_g["observed"] - _g["predicted_mean"]).mean()), 4),
          "mean NLL", round(float(_g["nll"].mean()), 5))

print("\nMAKE_CONTRAST_GRAFT END")
import dis
print("; ".join(f"{i.opname} {i.argrepr}" for i in list(dis.get_instructions(make_contrast_graft))[135:]))
```