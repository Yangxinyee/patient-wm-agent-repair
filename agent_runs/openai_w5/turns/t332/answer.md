The trial improves all four responses over the original, but insulin remains too weak relative to the estimated targets, especially when halved. I’ll save this feasible checkpoint and test intermediate persistence at 1,250 seconds. Passing factual validation while moving both insulin responses toward the estimates would support keeping it.

```python
from safetensors.torch import save_file as st_save

st_save(
    {n: torch.as_tensor(load(n)).contiguous() for n in edited()},
    "/work/checkpoint_feasible900.safetensors"
)

flow286 = torch.as_tensor(load("core.flow.a")).clone()
flow286[[21, 61]] = -np.log(1250.0)
save_edit("core.flow.a", flow286)
for name in ("attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias"):
    save_edit(name, cp264_285[name] + 0.60 * (cp282_285[name] - cp264_285[name]))

print("Sensor neuron pairs:", sensor_neurons)
rows286 = sorted(set(np.asarray(list(sensor_neurons.values()), dtype=int).ravel().tolist()))
w0286 = torch.as_tensor(load("attr_heads.value_bin.0.weight"))
b0286 = torch.as_tensor(load("attr_heads.value_bin.0.bias"))
print("Paired-head diagnostics [row, bias, insulin coefficient, NE coefficient]:")
for j in rows286:
    print(j, round(float(b0286[j]), 4),
          round(float(w0286[j,21] - w0286[j,61]), 4),
          round(float(w0286[j,15] - w0286[j,66]), 4))
print("Code embedding:", VOCAB["codes"])
print("Code contribution plus bias for reserved neurons:")
print((w0286[rows286] @ torch.as_tensor(load("embed.code.weight")).T
       + b0286[rows286, None]).numpy().round(2))
print("Input reconstruction helper:", inspect.signature(make_head_inputs212))
print("Helper dependencies:", make_head_inputs212.__code__.co_names)
print("Head helper:", inspect.signature(head225))
print("Head dependencies:", head225.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write("\n## Forwards 30–31 / run 286: feasible baseline and intermediate trial\n"
            "- Tau=900 tuning-A factual NLL 1.63892 vs original 1.63933: -0.00041.\n"
            "- Tau=900 paired shifts: insulin x2 -5.633, half +1.384; NE x2 +4.235, half -1.083.\n"
            "- Saved this locally feasible candidate in checkpoint_feasible900.safetensors.\n"
            "- Trial: insulin tau=1250 s, with 60% of the incremental tau=1500 factual center. All strengths unchanged.\n"
            "- Trial is unaccepted pending factual and rollout checks. Record-derived target estimates remain assumption-dependent.\n")
```
