The edit has four sensor-specific controls, each paired with a reference neuron. That lets me change causal readout strength without rebuilding the recurrent model. I’ll inspect those pairs and the saved rollout summaries next. Linear, drug-specific pair responses would favor readout recalibration; nonlinear or mixed responses would point back to dose transport.

```python
print("=== SMALL DESIGN OBJECTS ===")
for k in ["sensor_neurons", "private_pairs", "priv", "reserved", "_controls",
          "_pair_for_control"]:
    if k in globals():
        print(k, repr(globals()[k])[:700])

H237 = torch.as_tensor(load("attr_heads.value_bin.0.weight")).clone()
b237 = torch.as_tensor(load("attr_heads.value_bin.0.bias")).clone()
O237 = torch.as_tensor(load("attr_heads.value_bin.2.weight")).clone()
ob237 = torch.as_tensor(load("attr_heads.value_bin.2.bias")).clone()
print("=== RESERVED HEAD ROWS ===")
for i in [7,107,171,74,10,109,124,64]:
    row = H237[i]
    top = torch.argsort(row.abs())[-8:].flip(0)
    print(i, "bias", round(float(b237[i]),4),
          "top", [(int(j), round(float(row[j]),4)) for j in top],
          "pair coefficients", [round(float(row[j]),4) for j in [21,61,15,66]],
          "output", np.round(O237[:,i].numpy(),3).tolist())

print("=== SAVED ROLLOUT SUMMARIES ===")
for k in [1,2,16,17,18,19,20]:
    folder = f"/work/fwd_{k}"
    if not os.path.isdir(folder):
        continue
    files = os.listdir(folder)
    print(k, files)
    for fn in files:
        if fn.startswith("dose_edit_") and fn.endswith(".csv"):
            df237 = pd.read_csv(folder+"/"+fn)
            print(fn, df237.shape, df237.columns.tolist())
            if "edit" in df237 and "shift" in df237:
                print(df237.groupby("edit")["shift"].mean().round(4).to_dict())

print("=== TUNING IDS ===")
t237 = tuning if "tuning" in globals() else pd.read_csv("/data/tuning.csv")
ids237 = sorted(t237.stay_id.unique().tolist())
print("count", len(ids237), "first 12", ids237[:12])
print("=== NUMERIC DRUG NORMALIZATION ===")
print([s for s in VOCAB["numeric"] if s["code"] in ["insulin","norepinephrine"]])
```
