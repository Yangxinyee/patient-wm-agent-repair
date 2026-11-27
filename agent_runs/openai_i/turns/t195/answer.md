The readout helper and recurrent replay both ran successfully; only the final diagnostic print failed because it mixed NumPy and Torch operations. I’ll use paired dose perturbations next, keeping the factual state anchored to the exposed state. Consistent changes in the head inputs would support a direct head edit, even if the original predicted means respond weakly.

```python
import numpy as np, torch

def asnp175(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

print("Posterior original RMS:",
      float(np.sqrt(np.mean(asnp175(state_cal161)[:, 960:992]**2))))
print("Readout local variable names:", readout153.__code__.co_varnames)
print("\n=== CAPTURED READOUT FIELDS ===")
for key175, val175 in captured174["glucose_finger"][1].items():
    if hasattr(val175, "shape"):
        print(key175, tuple(val175.shape))
    elif isinstance(val175, (str, int, float, np.number)):
        print(key175, val175)

hbase175, obase175 = local174
print("\nReplay anchor errors:",
      float(torch.sqrt(torch.mean((hbase175-h174)**2))),
      float(torch.sqrt(torch.mean((obase175-out174)**2))))

xbase175 = X169[i174:i174+1]
zbase175 = mlp136(xbase175, "z_post")[:, :32]
probes175 = {}
for drug175, scales175 in [
    ("insulin", (0.0, 1.0)), ("insulin", (0.5, 1.0)),
    ("insulin", (2.0, 1.0)), ("insulin", (3.0, 1.0)),
    ("norepinephrine", (1.0, 0.0)), ("norepinephrine", (1.0, 0.5)),
    ("norepinephrine", (1.0, 2.0)), ("norepinephrine", (1.0, 3.0))
]:
    ucf175 = Uaug168[scales175][i174:i174+1]
    xcf175 = torch.cat([xbase175[:, :384], ucf175], dim=1)
    zcf175 = mlp136(xcf175, "z_post")[:, :32]
    acf175 = list(args174)
    acf175[3] = ucf175
    acf175[4] = args174[4] + zcf175-zbase175
    hcf175, ocf175 = call_core161(acf175, "input")
    anchored_h175 = h174 + hcf175-hbase175
    anchored_o175 = out174 + ocf175-obase175
    shifts175 = {}
    for ev175 in ["glucose_finger", "map_arterial"]:
        capcf175 = capture159(anchored_h175, anchored_o175, stay174, hour174, ev175)
        shifts175[ev175] = float(capcf175[0][0]-captured174[ev175][0][0])
        probes175[(scales175, ev175)] = capcf175
    print(drug175, scales175, "out change RMS",
          float(torch.sqrt(torch.mean((ocf175-obase175)**2))),
          "mean shifts", shifts175)

print("Active edits:", edited())
```