Raw head reconstruction succeeded. The rejected units’ train and tuning activation ranges are broadly similar, so simple dose-range extrapolation is not yet established. I’ll calculate exact fine-bin losses and inspect the fitting targets; this should reveal whether the failure comes from a few extreme cases or a broadly overfit readout.

```python
from pathlib import Path
import re, numpy as np

print("=== BIN METADATA ===")
print("num42 keys:",list(num42)[:8])
for key in list(num42)[:2]:
    print(repr(key),repr(num42[key])[:1700])
print("register constants:",[x for x in register67.__code__.co_consts if isinstance(x,str)])
print("numeric top-level sample:",repr(VOCAB["numeric"])[:900])

print("\n=== PAIRED FITTING TARGETS ===")
key227 = ("insulin",2.0)
for name in ["paired214","cases207","canon_cases200","targets179"]:
    if name not in globals():
        continue
    obj=globals()[name]
    print(name,"keys",list(obj)[:12] if isinstance(obj,dict) else getattr(obj,"shape",None))
    if isinstance(obj,dict) and key227 in obj:
        item=obj[key227]
        if isinstance(item,dict):
            for k,v in item.items():
                if isinstance(v,np.ndarray):
                    print(" ",k,v.shape,"mean/std",round(float(v.mean()),5),round(float(v.std()),5))
                else:
                    print(" ",k,type(v).__name__,str(v)[:100])
        elif isinstance(item,tuple):
            print(" tuple:",[getattr(v,"shape",type(v).__name__) for v in item])
        else:
            print(" ",getattr(item,"shape",type(item).__name__))
print("family204:",{k:(getattr(v,"shape",None),float(np.asarray(v).mean())) for k,v in family204.items()})

print("\n=== READOUT-TARGET NOTES ===")
for sec in re.split(r"(?=^## )",Path("/work/notes.md").read_text(),flags=re.M):
    m=re.match(r"## .*?[Rr]un(?:s)? (\d+)",sec)
    if m and int(m.group(1)) in [80,82,86,87,88,94,96,97,101,178,179,180,214,215]:
        print(sec[:480])
```