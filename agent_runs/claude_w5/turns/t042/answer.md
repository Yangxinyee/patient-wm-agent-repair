The last run printed the architecture, my notebook and the kernel's variables. Doses reach the glucose prediction only through the recurrent state. The value head sees only the last glucose reading, not past doses. So any dose effect must be carried in h or out and read out at the head.

Next I'll list the tensor names and shapes, look at the fwd_2 summary, and print the dose-decoding results (`cv_r2`, `tgt`) from my earlier state regressions, which aren't in the notebook. I want to know whether the original model's state linearly encodes recent insulin and norepinephrine doses.
- If it does, high cross-validated R² for lagged doses, I can edit only the head by adding a calibrated readout of that dose memory.
- If it doesn't, the edit has to create a dose memory in the recurrent path, which is harder.

```python
import json
print(json.dumps(json.load(open('/work/fwd_2/summary.json')))[:1500])
groups={}
for n in NAMES:
    groups.setdefault(n.split('.')[0]+'.'+n.split('.')[1] if '.' in n else n,[]).append(n)
for kk,v in groups.items():
    print(kk, [(x[len(kk)+1:],tuple(SHAPES[x])) for x in v])
for nm_ in ['cv_r2','tgt','feat','lag','D']:
    if nm_ in globals():
        v=globals()[nm_]
        print(nm_, type(v), (v if not hasattr(v,'shape') or np.size(v)<60 else (v.shape)))
```