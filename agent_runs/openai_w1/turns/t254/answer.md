The fitted directions are large because the local attention-to-reading sensitivity is small; that makes finite-amplitude testing essential. I’ll test those responses and the marker scales directly. The candidate is credible only if useful drug contrasts survive while factual marker leakage stays small.

```python
import numpy as np, torch

print("RELAY KEYS:", list(relay206.keys()))
print("MARKER CF SAMPLE:",
      {k: (v.shape if hasattr(v, "shape") else type(v).__name__)
       for k,v in cf215[("insulin",2.0)].items()})
print("RESERVED ROWS:", reserved222)
print("DIRECTION NORMS:", np.linalg.norm(B221, axis=0))
print("MARKER CALIBRATION:", A215)
print("MARKER FACTUAL:",
      np.quantile(Ff215, [.01,.5,.99], axis=0),
      "rms", np.sqrt(np.mean(Ff215**2,axis=0)))
print("F222:", np.quantile(F222,[.01,.5,.99],axis=0))
print("V215 norms:", np.linalg.norm(V215,axis=1))

print("\nRUN-222 TENSOR DIFFERENCES FROM CURRENT:")
for _n226, _x226 in [
    ("core.prelude.0.attn.q.weight",q222),
    ("core.prelude.0.attn.kv.weight",kv222),
    ("core.prelude.0.attn.out.weight",out222)]:
    _old226=np.asarray(load(_n226))
    _diff226=np.asarray(_x226)-_old226
    print(_n226, "norm", np.linalg.norm(_diff226),
          "changed rows", np.flatnonzero(np.max(np.abs(_diff226),axis=1)>1e-7)[:55])

print("\nFINITE RESPONSE: columns =", events221)
print("zero", finite_readout221(np.zeros(192)))
for _j226 in range(2):
    for _amp226 in [-1.0,-0.3,-0.1,0.1,0.3,1.0]:
        print("direction",_j226,"amplitude",_amp226,
              finite_readout221(_amp226*B221[:,_j226]))

print("\nKERNEL NOTES:")
_matches226=list(re.finditer(r"(?im)^.*(?:impulse|six-lag|six-hour|dynamic.*effect).*$",_notes224))
for _m226 in _matches226[:6]:
    print(_notes224[_m226.start():min(len(_notes224),_m226.end()+210)])

with open("/work/notes.md","a") as _f226:
    _f226.write("\n## Run 226\nChecking finite-amplitude reading responses of the two closed-form attention directions and factual marker leakage. Large direction norms make infinitesimal Jacobian success insufficient. No weights changed.\n")
```