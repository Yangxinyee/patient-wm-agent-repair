The stored targets include hours with no prior drug exposure, whereas the rollout readout uses exposed hours. I will use matching training states to estimate only the extra factual offset caused by longer insulin persistence. This is a centering correction, not a new estimate of drug effects; poor held-out prediction of that offset would argue against it.

```python
from safetensors.torch import load_file, save_file

W274 = {k: torch.as_tensor(load(k), dtype=torch.float32).clone() for k in NAMES}
B274 = {k: torch.as_tensor(load_original(k), dtype=torch.float32).clone() for k in NAMES}
B274.update({k: v.clone() for k, v in BASE264.items()})
sf274 = {k: np.asarray(v) for k, v in _s271.items()}
oldsf274 = dict(np.load("/work/fwd_15/states_edited.npz"))
print("baseline/fresh states aligned:",
      np.array_equal(oldsf274["stay_id"], sf274["stay_id"]),
      np.allclose(oldsf274["hour"], sf274["hour"]))
print("baseline state dimensions:", oldsf274["x"].shape)

rows274 = head_rows[
    head_rows.stay_id.isin(np.unique(sf274["stay_id"]))
].reset_index(drop=True).copy()
prev274 = np.full(len(rows274), -1, dtype=np.int64)
for sid, inds in rows274.groupby("stay_id", sort=False).groups.items():
    inds = np.asarray(list(inds))
    si = np.flatnonzero(sf274["stay_id"] == sid)
    pos = np.searchsorted(sf274["hour"][si], rows274.loc[inds, "hour"], side="left") - 1
    ok = pos >= 0
    prev274[inds[ok]] = si[pos[ok]]
keep274 = prev274 >= 0
rows274 = rows274.loc[keep274].reset_index(drop=True)
prev274 = prev274[keep274]

inputs274 = make_head_inputs212(W274, sf274, rows274, prev274)
oldinputs274 = make_head_inputs212(B274, oldsf274, rows274, prev274)
print("builder return:", type(inputs274).__name__)
if isinstance(inputs274, (tuple, list)):
    print("components:", [(type(x).__name__, getattr(x, "shape", None)) for x in inputs274])
else:
    print("shape:", getattr(inputs274, "shape", None))

A274 = np.asarray(load("attr_heads.value_bin.0.weight"))
b274 = np.asarray(load("attr_heads.value_bin.0.bias"))
C274 = np.asarray(load("attr_heads.value_bin.2.weight"))
for sensor, (signal, control) in sensor_neurons.items():
    print(sensor, "neurons", signal, control)
    for neuron in [signal, control]:
        ix = np.argsort(np.abs(A274[neuron]))[-8:][::-1]
        print(neuron, "bias", float(b274[neuron]), "largest weights",
              [(int(i), round(float(A274[neuron, i]), 5)) for i in ix],
              "output norm", round(float(np.linalg.norm(C274[:, neuron])), 5))
print("reached-hour targets:",
      _r257.groupby("edit")[["shift", "estimated_effect"]].mean().round(4).to_dict())
```
