Three extreme glucose corrections explain almost all the warm-up trial’s extra loss. The cached decoder also produces much larger values where the correction is inactive, suggesting a gate extrapolation problem. I’ll scan a guard against that region; it should preserve supported late corrections while preventing those extreme activations.

```python
import numpy as np
import pandas as pd
from scipy.special import ndtr

r159_X = r157_np(r145_X).astype(np.float64)
r159_p, r159_m = map(int, r151_slots)
r159_W1 = r157_np(r115_strength2["attr_heads.value_bin.0.weight"]).astype(np.float64)
r159_b1 = r157_np(r115_strength2["attr_heads.value_bin.0.bias"]).astype(np.float64)
r159_gv = (r159_W1[r159_p] + r159_W1[r159_m]) / 2
r159_gb = (r159_b1[r159_p] + r159_b1[r159_m]) / 2
r159_dv = (r159_W1[r159_p] - r159_W1[r159_m]) / 2
r159_db = (r159_b1[r159_p] - r159_b1[r159_m]) / 2
r159_G = np.einsum("sni,i->sn", r159_X, r159_gv) + r159_gb
r159_D = np.einsum("sni,i->sn", r159_X, r159_dv) + r159_db
r159_F = r159_D / 0.2
r159_C = r157_np(r115_strength2["attr_heads.value_bin.2.weight"])[:, r159_p].astype(np.float64)
r159_pair0 = r157_np(r145_Hins2)[:, :, r159_p] - r157_np(r145_Hins2)[:, :, r159_m]
r159_const = r157_np(r145_Lins2).astype(np.float64) - r159_pair0[:, :, None] * r159_C
r159_original_loss = r157_np(r145_loss_orig)
r159_late_active = (r152_hours >= 13) & np.asarray(r144_active, dtype=bool)

def r159_pair(g, d):
    return (g + d) * ndtr(g + d) - (g - d) * ndtr(g - d)

print("decoder consistency", float(np.max(np.abs(r159_F - r144_D))))
print("active counts by event")
for _event in np.unique(r144_events):
    _mask = r144_events == _event
    print(_event, int(_mask.sum()), int((r144_active & _mask).sum()))
for _label, _mask in [
    ("active", np.asarray(r144_active, bool)),
    ("inactive glucose", np.char.startswith(r144_events.astype(str), "glucose") & ~r144_active)
]:
    print(_label, "F", np.quantile(r159_F[:, _mask], [0, .01, .1, .5, .9, .99, 1]).round(3),
          "G", np.quantile(r159_G[:, _mask], [0, .01, .1, .5, .9, .99, 1]).round(3))

r159_sources = {
    "original": (r159_gv, r159_gb),
    "warmup": (r152_gates[(2, 0.1, 3.0)]["v"],
               r152_gates[(2, 0.1, 3.0)]["b"])
}
r159_candidates = {}
r159_results = []
for _source, (_gv, _gb) in r159_sources.items():
    for _penalty in [0, 1, 2, 4, 6, 8]:
        for _threshold in ([0] if _penalty == 0 else [0.5, 1.0]):
            _v = np.asarray(_gv) - _penalty * r159_dv / 0.2
            _b = float(_gb) - _penalty * (r159_db / 0.2 - _threshold)
            _G = np.einsum("sni,i->sn", r159_X, _v) + _b
            _pair = r159_pair(_G, r159_D)
            _L = r159_const + 1.3 * _pair[:, :, None] * r159_C
            _loss = r138_nll(_L) - r159_original_loss
            _eps = 0.01
            _deriv = (
                r159_pair(_G - _penalty * _eps, r159_D + 0.2 * _eps)
                - r159_pair(_G + _penalty * _eps, r159_D - 0.2 * _eps)
            ) / (2 * _eps * 0.4)
            _key = (_source, _penalty, _threshold)
            r159_candidates[_key] = {"v": _v, "b": _b}
            r159_results.append({
                "source": _source, "penalty": _penalty, "threshold": _threshold,
                "all": float(_loss.mean()),
                "fit": float(_loss[r137_fit].mean()),
                "held_out": float(_loss[r137_val].mean()),
                "late_response": float(_deriv[:, r159_late_active].mean()),
                "late_response_p01": float(np.quantile(_deriv[:, r159_late_active], .01)),
                "inactive_glucose_gate_on": float((_G[:, np.char.startswith(
                    r144_events.astype(str), "glucose") & ~r144_active] > 0).mean()),
                "max_loss": float(_loss.max())
            })
r159_table = pd.DataFrame(r159_results)
print("=== SUPPORT-GUARD SCAN, INSULIN STRENGTH 2.6 ===")
print(r159_table.sort_values(["held_out", "all"]).round(6).to_string(index=False))
```
