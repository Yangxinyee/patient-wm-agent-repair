The record-derived insulin estimate implies roughly −11 glucose units for doubling and +5.5 for halving, though it depends on the identification assumptions. The current rollout is −9.5 and +2.8. I’ll now measure bin-centre calibration error and screen spare neurons. A useful candidate needs a consistent calibration signal and negligible disruption to other readings.

```python
import torch.nn.functional as F
from scipy.special import softmax

calib_edges267 = np.asarray(next(
    e["fine_bin_edges"] for e in VOCAB["numeric"]
    if e["code"] == "glucose_finger" and e["attribute"] == "value"))
calib_centres267 = np.asarray(VOCAB["value_bin_centres"]["glucose_finger"], dtype=float)

_m = calibration_cache["meta"]
_vals = np.asarray(_m["value"], dtype=float)
_mask = np.asarray(_m["event"] == "glucose_finger")
for _side in ("left", "right"):
    _b = np.searchsorted(calib_edges267, _vals[_mask], side=_side)+1
    print("bin convention", _side, "mismatches",
          int(np.sum(_b != np.asarray(_m["observed_bin"])[_mask])))
calib_side267 = min(("left", "right"), key=lambda s:
    np.sum(np.searchsorted(calib_edges267, _vals[_mask], side=s)+1
           != np.asarray(_m["observed_bin"])[_mask]))

calib_gpu267 = finger_gpu400.copy()
calib_gpu267["observed_bin"] = np.searchsorted(
    calib_edges267, calib_gpu267["observed"], side=calib_side267)+1
calib_gpu267["observed_centre"] = calib_centres267[
    calib_gpu267["observed_bin"].to_numpy()-1]
calib_gpu267["centre_error_original"] = (
    calib_gpu267["predicted_mean_original"]-calib_gpu267["observed_centre"])
calib_gpu267["centre_error_edited"] = (
    calib_gpu267["predicted_mean_edited"]-calib_gpu267["observed_centre"])
print("GPU bin-centre errors:")
print(calib_gpu267.groupby("insulin_active")[
    ["centre_error_original", "centre_error_edited", "delta_nll"]].agg(["mean", "count"]))
print("By factual batch:")
print(calib_gpu267.groupby("batch")[
    ["centre_error_original", "centre_error_edited", "delta_nll"]].mean())
print("By previous glucose:")
print(calib_gpu267.groupby("last_group", observed=True)[
    ["centre_error_original", "centre_error_edited", "delta_nll"]].mean().to_string())

calib_x267 = np.asarray(calibration_cache["x"], dtype=np.float32)
calib_h267 = F.gelu(
    torch.from_numpy(calib_x267) @ torch.as_tensor(
        np.asarray(sc_original[sc_names[0]]), dtype=torch.float32).T
    + torch.as_tensor(np.asarray(sc_original[sc_names[1]]), dtype=torch.float32)
).numpy()
calib_hmean267 = calib_h267.mean(axis=(0, 1))
calib_hsd267 = calib_h267.std(axis=(0, 1))
calib_w2orig267 = np.asarray(sc_original[sc_names[2]])
calib_neuron_cost267 = calib_hsd267 * np.linalg.norm(calib_w2orig267, axis=0)
calib_spare267 = [int(j) for j in np.argsort(calib_neuron_cost267)
                 if int(j) not in set(map(int, sc_nodes))][:8]
print("Spare-neuron proxy costs:",
      [(j, float(calib_neuron_cost267[j]), float(calib_hmean267[j]))
       for j in calib_spare267])

print("Finger gate activation by event:")
for _label, _cache in [("factual", calibration_cache)] + [
        (f"dose{m}", sc_syn[m]) for m in (0.5, 1.0, 2.0)]:
    _s = np.asarray(_cache["x"]) @ np.asarray(selective_sep_w) + selective_sep_b
    _g = torch.from_numpy(np.asarray(30*_s, dtype=np.float32))
    _f = (F.gelu(_g+1)-F.gelu(_g)).numpy().mean(axis=0)
    for _event in sorted(set(_cache["meta"]["event"])):
        _sel = np.asarray(_cache["meta"]["event"] == _event)
        print(_label, _event,
              np.round(np.quantile(_f[_sel], [0, .01, .5, .99, 1]), 4))
```
