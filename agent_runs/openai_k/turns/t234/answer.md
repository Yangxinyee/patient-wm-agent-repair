Zeroing empty actions made the posterior match worse and produced large spurious dose-channel values, so I reject that change. Drug-bundle posterior means already match official states to about \(1.6\times10^{-5}\) RMS. I’ll now audit the downstream state/readout interface; it must reproduce official edit differences before I use it to select another routing edit.

```python
import numpy as np
import torch
from pathlib import Path

print("=== READOUT TENSOR LAYOUT ===")
for name in NAMES:
    if any(s in name for s in ("attr_heads.value","last_numeric","last_status",
                              "query_norm","type_emb","z_head","z_prior","gap.",
                              "out_norm","embed_transition.value")):
        print(name,SHAPES[name])

print("\n=== REPLAY/SURROGATE FUNCTIONS ===")
for name in ("r131_f","repaired_core_local","posterior_token154",
             "local_token154","dynamic_views154","dynamic_delta154",
             "r186_heads","r186_means","solve_routing158"):
    fn = globals().get(name)
    if callable(fn) and hasattr(fn,"__code__"):
        c = fn.__code__
        print(name,"args",c.co_varnames[:c.co_argcount],
              "locals",c.co_varnames[c.co_argcount:c.co_argcount+12],
              "uses",c.co_names)

print("\n=== SURROGATE AND PROBE OBJECTS ===")
for name,obj in list(globals().items()):
    if not any(s in name for s in ("144","148","150","151","154","158","186")):
        continue
    if torch.is_tensor(obj) or isinstance(obj,np.ndarray):
        print(name,type(obj).__name__,tuple(obj.shape))
    elif isinstance(obj,dict):
        print(name,"dict",len(obj),"keys",list(obj)[:10])
    elif isinstance(obj,(list,tuple)) and len(obj)<30:
        print(name,type(obj).__name__,len(obj))
    elif isinstance(obj,(float,int,str)) and len(str(obj))<100:
        print(name,repr(obj))

print("\n=== EARLY CORE VALIDATION EVIDENCE ===")
lines213 = Path("/work/notes.md").read_text().splitlines()
hits213 = [line for line in lines213[:900]
           if "core" in line.lower() and any(s in line.lower() for s in
              ("rms","match","exact","validated","error","replay"))]
for line in hits213[:14]:
    print(line[:300])

with open("/work/notes.md","a") as f:
    f.write("\n## Run 213: reject zero-empty-action change; audit downstream surrogate\n"
            "- Zeroing empty actions worsened original posterior RMS (0.105 to 0.112) and created large false reserved-channel values; rejected.\n"
            "- Drug-bundle posterior RMS is 1.57e-5 under existing encoding, supporting its use for direct drug-bundle tests.\n"
            "- Auditing official paired-state readout calibration before another routing edit. No weights changed.\n"
            "- Python runs: 213; forward passes: 14.\n")
```