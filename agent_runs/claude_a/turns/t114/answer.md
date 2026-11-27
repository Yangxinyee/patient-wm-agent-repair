Two facts from this run change the plan.

- **The state is rewritten every bundle.** Every dimension of h is multiplied by `abar` ≈ 0.30–0.45 at each bundle, and `softplus(delta_in)` ≈ 1. Persistent information therefore survives only by being re-written through the prelude. A dose memory with hour-scale decay is still possible: set `a_in` very negative for one dimension, so `abar` ≈ 1 there, and let `flow.a` (dt in seconds) set the decay per hour.
- **No dimension of h is unused.** The smallest per-dimension std is about 2.3 (dim 1). A memory therefore has to repurpose dimensions and accept losing what they carried. The h vector's spread across dimensions is large (about 65, with only 11% variation), so a memory value of a few units barely moves the LayerNorm statistics. That is good for side effects.
- **The evaluator is fast.** Teacher-forced evaluation runs in 0.1 s per 10 stays, so I can measure the factual cost of repurposing each dimension precisely and cheaply.

**Next step:** the readout must be code-gated, so the insulin memory moves glucose but not MAP. Whether that is clean depends on whether the value head uses its own code and bin embeddings or shares them with the encoder. I will read the head and core sections of ARCHITECTURE.md again. I will also check what `tf_eval` returns, so I can use it for per-dimension cost measurements.

**What would confirm or refute the design:** if the head-side tables (`last_numeric`, `last_status`) are head-only and the code embedding enters `with_code` additively, then the gated-unit readout is implementable as planned.

```python
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
import re
keys = ['with_code','query','view','value_bin','last_numeric','embed_transition','type_emb','filter','pending','z_head','gap','out_norm','slots','b_in','delta_in','a_in','flow']
lines = txt.split('\n')
hit = [i for i,l in enumerate(lines) if any(k in l for k in keys)]
shown=set(); out=[]
for i in hit:
    for j in range(max(0,i-1), min(len(lines), i+2)):
        if j not in shown: shown.add(j); out.append(f"{j}: {lines[j]}")
print("\n".join(out)[:7000])
print("tf_eval returns:", [ (type(x).__name__, getattr(x,'shape',None) if hasattr(x,'shape') else (x if np.isscalar(x) else len(x))) for x in r_])
```