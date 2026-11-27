The offset reverses sign on the tool, and it does so in the teacher-forced part. Insulin-treated stays get +2.96 mg/dL on average, every one of them positive, while untreated stays sit near 0. The offset correlates +0.43 with the insulin rate, and MAP's offset correlates −0.50 with the NE rate. So my edit moves glucose up with more insulin and MAP down with more NE. That is the opposite of what my local check claimed, −1.27 mg/dL per u/h.

Either the local model does this too and my earlier check was wrong, or the tool's computation differs. To tell which, I need local head predictions on arbitrary stays and states, starting with the tuning stays at hour 13, and later a local rollout. So I'll look at the stored head inputs (`H`, `fixed`, `TRr`), the VOCAB normalisation, and what ARCHITECTURE.md says about how the rollout draws readings.

```python
print(type(H[0]), H[0] if not isinstance(H[0], dict) else {k:(v if not hasattr(v,'shape') else v.shape) for k,v in H[0].items()})
print(type(fixed), getattr(fixed,'shape',None), type(TRr), TRr.shape, type(V0), V0.shape)
print(list(VOCAB.keys()))
for k in VOCAB:
    v = VOCAB[k]
    print(k, (str(v)[:300]))
arch = open('/model/ARCHITECTURE.md').read()
import re
for kw in ['draw','sample','rollout','path','dose_edit','hidden']:
    for mm in re.finditer(kw, arch):
        s = max(0, mm.start()-200); print('['+kw+']', arch[s:mm.start()+300].replace('\n',' ')); print()
        break
```