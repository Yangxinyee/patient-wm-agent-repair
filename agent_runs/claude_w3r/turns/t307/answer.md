The value head is a two-layer network over 17 classes (0 means no value, 1–16 are fine bins). A per-code, per-last-bin table, `embed_transition.value_bin`, is added to its logits, and the predicted mean is the probability-weighted sum of bin centres. All core edits are still in the kernel (EDIT, KVn, Won, Bin2, DECL).

So the plan is unchanged. Two inactive hidden units per drug will read the rate signal in the head input, gated on the reading's code, and add a tilt along the bin centres. The `embed_transition` rows can carry the insulin band term.

First I need a few facts:
- how the with-code helper functions work;
- the bin-centre layout and code indices;
- whether the gate and rate read directions are disturbed by the added drug signal.

If GD·DQ is small and RD reads its own drug at about 1 and the other drug at about 0, the pair-unit design will work as planned.

```python
import numpy as np
for fn in ['wc_of','wc_full','wc_row','pick','apply_local']:
    c = globals()[fn].__code__
    print(fn, c.co_varnames[:c.co_argcount], 'names:', ','.join(c.co_names)[:250])
VC = VOCAB['value_bin_centres']
print(type(VC).__name__, (list(VC.keys())[:15] if isinstance(VC, dict) else np.shape(VC)))
print(str(VOCAB['value_bin_note'])[:250]); print(str(VOCAB['codes'])[:400])
arrs = {k: v.shape for k, v in list(globals().items()) if isinstance(v, np.ndarray) and v.ndim >= 2 and not k.startswith('_')}
print(str(arrs)[:1800])
for d in ['ins','ne']:
    for g in ['gf','map']:
        x = DQ[d] @ GD[g]; print('GD',g,'.DQ',d, round(x.mean(),4), round(x.std(),4))
    o = 'ne' if d=='ins' else 'ins'
    print('RD',d,'own', round((DQ[d]@RD[d]).mean(),4), round((DQ[d]@RD[d]).std(),4), 'other', round((DQ[o]@RD[d]).mean(),4))
print([round(float(np.linalg.norm(v)),2) for v in [RD['ins'],RD['ne'],GD['gf'],GD['map']]])
```