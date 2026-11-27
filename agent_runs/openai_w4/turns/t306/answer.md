The lower-gate screen found factual-safe candidates, including gain 1.25/gate 24 at about +0.00323 NLL. The insulin output columns are exactly opposite, which lets me test a symmetric response pair without changing the downstream connection. I’ll screen that construction; better factual headroom at higher gains would justify a rollout test, not establish accuracy by itself.

```python
import numpy as np, torch

def symmetric_ins265(gain, gate_shift):
    c = candidate258(1.0, gate_shift)
    w = a262(c["z_post.0.weight"]).copy()
    b = a262(c["z_post.0.bias"]).copy()
    gw = w[184].copy()
    gb = float(b[184])
    qw = a262(ins_dw257)[96]
    qb = float(a262(ins_gain_db257)[96])
    w[96], w[184] = gw + gain*qw/2, gw - gain*qw/2
    b[96], b[184] = gb + gain*qb/2, gb - gain*qb/2
    c["z_post.0.weight"] = like258(w,c["z_post.0.weight"])
    c["z_post.0.bias"] = like258(b,c["z_post.0.bias"])
    return c

screen265 = {}
old265 = (rW, rp_linear, repair_bank)
try:
    for key in ((1.25,22.0),(1.5,22.0),(1.5,26.0),
                (2.0,22.0),(2.0,26.0),(2.5,26.0)):
        c = symmetric_ins265(*key)
        score = factual_eval194(c)
        delta = float(np.mean(score[1])-np.mean(original_score264[1]))
        screen265[key] = {"weights":c, "nll":float(np.mean(score[1])),
                          "delta":delta}
        print("SYMMETRIC",key,"NLL",round(float(np.mean(score[1])),8),
              "delta",round(delta,8))
finally:
    rW, rp_linear, repair_bank = old265

print("\nNE RESPONSE INTERFACES")
names265 = [k for k in list(globals()) if
            any(t in k for t in ("206","209","187","193","194","210"))]
for k in sorted(names265)[:65]:
    v = globals()[k]
    if isinstance(v, dict):
        print(k,"dict",len(v),list(v)[:6])
    elif isinstance(v,(np.ndarray,torch.Tensor)):
        print(k,type(v).__name__,tuple(v.shape))
    elif callable(v):
        print(k,"function",str(inspect.signature(v))[:120])
    elif isinstance(v,(int,float,str,tuple)):
        print(k,repr(v)[:160])

with open("/work/notes.md","a") as f:
    f.write("\n## Run 265\nUNSAVED symmetric insulin-pair factual screen: "
            +repr({k:{s:v[s] for s in ("nll","delta")}
                   for k,v in screen265.items()})
            +". Pair uses shared action gate with +/- half the innovation, "
             "and exact opposite output columns. CPU globals restored; "
             "submitted tensors unchanged.\n")
```
