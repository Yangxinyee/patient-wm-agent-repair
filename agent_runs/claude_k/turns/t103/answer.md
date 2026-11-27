The tensor list shows the core is mostly residual: c_out has no bias, the coda is a pre-norm transformer layer, and h is updated by diagonal decays plus block increments. Every step still mixes all dimensions through c_out, attention and LayerNorms, so an exact hand-built dose memory inside the core would be fragile.

The practical route is the one in my notes: read the dose information already in the head's input q with a linear probe. I would add it inside the value head through hidden units gated by the reading's code, so the insulin signal shifts glucose and the norepinephrine signal shifts MAP. I would then calibrate the gains on dose_edit rollouts against my data-based effect estimates.

Runs 74–86 left arrays behind (Q99, WC99, Hact99, R99, TR99, BASE99, D99 and others). Before redoing any work I'll check what they hold: shapes, plus a short look at the dict and list variables. If they include q, with_code and the hidden activations at each reading of the 99 state stays, I can fit probes and cost edits offline right away.

```python
import numpy as np
cands = [k for k in vars_ if k.endswith('99') or k in ('Q','Qg','PRE','PREall','WCall','WCc','T_all','DSall','Hh','ZP','ZPR','ZT','XS','XA','Xne','Xi','cand','cand2','cand_ids','both_ids','base_nll','best','sel')]
for k in cands:
    v = globals().get(k)
    if isinstance(v, np.ndarray):
        print(k, v.shape, v.dtype, f"{np.nanmean(v):.3g}")
    elif hasattr(v, 'shape'):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, (list, tuple, dict)):
        print(k, type(v).__name__, len(v), (list(v.keys())[:6] if isinstance(v, dict) else str(v[:3])[:80]))
    else:
        print(k, type(v).__name__, str(v)[:80])
```