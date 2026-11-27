The installed head uses a centered exposure signal with sensor-specific GELU gates, so gating remains a plausible source of asymmetry. The diagnostic stopped on a tensor-conversion error without changing weights. I’ll fix that and check whether the intended sensor gates are safely in their linear range.

```python
def array154(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

W154 = array154(load("attr_heads.value_bin.0.weight")).astype(np.float64)
b154 = array154(load("attr_heads.value_bin.0.bias")).astype(np.float64)
wc154 = array154(_r147_wc).astype(np.float64)
pre154 = wc154 @ W154.T + b154
dose_vectors154 = []

print("=== GATE AND CENTERED-EXPOSURE AUDIT ===")
for j154, sp154 in enumerate(_r148_specs):
    sn154, cn154 = _r148_reserved[2*j154:2*j154+2]
    own154 = _r148_event_array == sp154["event"]
    dv154 = (
        (sp154["row_signal"] - sp154["row_control"]) / sp154["c"]
        + np.asarray(sp154["proxy"].coef_)
    )
    dose_vectors154.append(dv154)
    decoded154 = (wc154 @ dv154).mean(axis=0)
    true154 = _r147_dose[:, sp154["drug"]]
    test154 = own154 & _r147_test.astype(bool)
    print(sp154["event"], "n", int(own154.sum()), "test", int(test154.sum()))
    print("own control gate q05/q50/q95:",
          np.round(np.quantile(pre154[:, own154, cn154], [.05, .5, .95]), 4))
    print("own signal gate q05/q50/q95:",
          np.round(np.quantile(pre154[:, own154, sn154], [.05, .5, .95]), 4))
    print("own centered exposure q05/q50/q95:",
          np.round(np.quantile(
              (pre154[:, own154, sn154]-pre154[:, own154, cn154])/sp154["c"],
              [.05, .5, .95]), 4))
    print("decoded dose / true mean:", round(float(decoded154[own154].mean()), 4),
          round(float(true154[own154].mean()), 4))
    print("held-out decoded-dose RMSE:",
          round(float(np.sqrt(np.mean((decoded154[test154]-true154[test154])**2))), 4))
    unit154 = dv154 / (dv154 @ dv154)
    print("unit dose change: signal/control preactivation:",
          np.round((W154 @ unit154)[[sn154, cn154]], 7))
    print("largest other-neuron response:",
          float(np.max(np.abs((W154 @ unit154)[_r148_keep]))))

print("=== CENTRE-TABLE LAYOUT ===")
print("codes:", VOCAB["codes"])
centres154 = VOCAB["value_bin_centres"]
print("centre object:", type(centres154).__name__)
if isinstance(centres154, dict):
    print("keys:", list(centres154))
    for key154 in list(centres154)[:2]:
        print(key154, centres154[key154])
else:
    print("shape:", np.asarray(centres154).shape)
    print(np.asarray(centres154)[:2])
print("other persistent centre objects:",
      [k for k, v in snapshot150 if re.search("centre|center", k, re.I)])

print("=== RECORD IDENTIFICATION CHECKS ===")
print(Path("/work/notes.md").read_text()[4500:7400])
```
