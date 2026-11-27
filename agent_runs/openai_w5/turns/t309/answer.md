The current head uses eight reserved neurons to turn isolated dose signals into sensor-specific logit shifts. Earlier fixed-history tests were approximately symmetric, so the full-rollout asymmetry is not explained by the readout alone. I’ll recover the transport diagnostics and parameter layout. A dose-dependent transport distortion would support repairing that pathway rather than simply increasing head strength.

```python
import numpy as np, torch, re
from pathlib import Path
print("=== PAIRS AND SENSOR NEURONS ===")
print("private_pairs:", private_pairs)
print("sensor_neurons:", sensor_neurons)
print("sensor controls:", r249_sensor_controls)
print("=== FINITE TRANSPORT NOTES ===")
nt269=Path("/work/notes.md").read_text()
p269=nt269.find("## Runs 156")
print(nt269[p269:p269+4600])
print("=== AVAILABLE HELPERS ===")
for k,v in list(globals().items()):
    if callable(v) and hasattr(v,"__code__") and not k.startswith("_"):
        print(k, v.__code__.co_varnames[:v.__code__.co_argcount])
print("=== RESERVED NEURON PARAMETERS ===")
A269=torch.as_tensor(load("attr_heads.value_bin.0.weight"))
b269=torch.as_tensor(load("attr_heads.value_bin.0.bias"))
C269=torch.as_tensor(load("attr_heads.value_bin.2.weight"))
for sensor269, ids269 in sensor_neurons.items():
    print(sensor269, ids269)
    for n269 in np.asarray(ids269).reshape(-1):
        n269=int(n269)
        print(n269,"bias",float(b269[n269]),"private weights",
              A269[n269,[15,21,61,66]].tolist(),
              "C norm",float(C269[:,n269].norm()))
print("=== STATE AND DATA OBJECTS ===")
for k,v in list(globals().items()):
    if isinstance(v,(dict,pd.DataFrame,np.ndarray,torch.Tensor)) and re.search(r"(^s[0-9]|^df|^train|^tune|^rows|^W|^O|^codes|^norm|^r249)",k):
        desc=str(getattr(v,"shape",""))
        if isinstance(v,dict): desc="keys="+str(list(v)[:7])
        print(k,type(v).__name__,desc[:130])
```