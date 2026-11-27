Calibrating insulin on all 400 tuning stays narrows down f. My truth estimate there is −13.41 for ×2 and +6.71 for ×½:

| | insulin ×2 | insulin ×½ |
|---|---|---|
| f = 1.1 (offline) | −13.25 | +6.36 |
| f = 1.2 (offline) | −14.17 | +6.89 |
| truth (400 stays) | −13.41 | +6.71 |

The offline copy runs about 2.5% stronger than the tool, so I need an offline value near 1.025 × truth. That gives f ≈ 1.154 for ×2 and about 1.20 for ×½. I will take f = 1.16, which favours the main readout. The expected tool values are about −13.5 (truth −13.41) and +6.5 (truth +6.71). Since only the two readout columns are scaled, the factual cost should rise only about 1.16² ≈ 1.35-fold, to roughly +0.0008 (drug-heavy set) and +0.0002 (general set).

For norepinephrine, the truth rises from +1.7 at hour 13 to a plateau of about +4.8 by hour 17. The ×½ edit mirrors this, from −0.9 to −2.4. The model gives about −0.4 throughout, so it has no real drug effect.

Plan for norepinephrine: reuse the original model's own encoding of the dose in its state, with a readout gated to map_arterial. This only works cheaply if doubling norepinephrine pushes the state along directions that vary little in factual data. Otherwise the readout would also move factual MAP predictions and cost NLL.

This run does six things:
1. Saves f = 1.16 and records the insulin results in the notebook.
2. Measures how the state at MAP queries shifts under norepinephrine ×2 and ×½, with readings held fixed.
3. Fits the most cost-efficient readout direction on one set of 300 train stays and tests it on another 300.
4. Checks that the ×½ shift is about −0.5 times the ×2 shift along that direction.
5. Ranks the value-head units by how cheap they are to repurpose.
6. Prints the vocabulary entries I need: the MAP code and its bin centres.

What would confirm or refute the plan:
- **Feasible:** the out-of-sample signal-to-noise (D²) is at least about 20, and the ×½/×2 ratio is near −0.5.
- **Not feasible:** a D² of about 5 or less would mean any norepinephrine readout costs too much factual fit. I would then keep the insulin-only edit.

```python
import time, numpy as np, torch, pandas as pd, json, traceback as tb
import torch.nn.functional as F_
T0 = time.time()
key2='attr_heads.value_bin.2.weight'
Pins = scaledP(1.16); save_edit(key2, Pins[key2]); Pcur = {n: load(n) for n in NAMES}
print('saved f=1.16; edited', len(edited()), torch.equal(Pcur[key2], Pins[key2]))
with open('/work/notes.md','a') as f:
    f.write('- Run284-287: tool fwd5 (sel1, theta-2.45): ed ins_x2 -12.13 half 5.67; truth on those rows -13.25/6.63 (old -14.30 target used other rows). 40-subset tool -14.66/6.83 vs offline -14.93/7.19 => replica ~2.5% stronger than tool. Model ins effect grows ~linearly with hour (-2 h13 .. -22.6 h24), truth plateaus ~-15 by h18; model too steep in glucose level.\n  400 tuning offline: f=1.1 -13.25/6.36, f=1.2 -14.17/6.89 (truth -13.41/6.71). SAVED f=1.16 (W2 cols 107,109 x1.16). Expect tool ~-13.5/6.5.\n- NE truth profile sel1: x2 +1.7 h13 -> plateau ~4.8 by h17; half -0.9 -> -2.4; model ~-0.4 flat. NE plan B: readout on original state dirs (whitened Sigma^-1 Delta), gated to map_arterial, 2 cheap value units.\n')
s4 = open('/work/lib4.py').read(); i = s4.find('def reenc'); print(s4[i:i+500])
print('VOCAB', {k: (list(v.keys())[:10] if isinstance(v, dict) else (v if not hasattr(v,'__len__') or len(v) < 30 else len(v))) for k, v in VOCAB.items()})
def cpy(ds):
    d = dict(ds)
    for k in ('e','u'):
        if k in d: d[k] = d[k].clone() if torch.is_tensor(d[k]) else np.array(d[k], copy=True)
    return d
def renc(ds, P):
    d = cpy(ds); r_ = reenc(d, P); return r_ if r_ is not None else d
NE = {}
try:
    for nm, d0 in [('E', dsE), ('F', dsF)]:
        d = renc(d0, Pcur); wcf, _, rf = capture_wc(Pcur, d)
        wc2 = capture_wc(Pcur, renc(ds_edit(d, 'norepinephrine', 2.0), Pcur))[0]
        wch = capture_wc(Pcur, renc(ds_edit(d, 'norepinephrine', 0.5), Pcur))[0]
        NE[nm] = dict(d=d, wcf=wcf, wc2=wc2, wch=wch, rf=rf)
        print(nm, 'wc', tuple(wcf.shape), 'r', [tuple(getattr(x,'shape',())) for x in rf], 'nll', float(torch.as_tensor(rf[0]).float().mean()), round(time.time()-T0))
    for k, v in NE['E']['d'].items():
        print(' ds', k, type(v).__name__, (len(v), list(v.columns)[:12]) if isinstance(v, pd.DataFrame) else tuple(getattr(v, 'shape', ())))
except Exception: print('NE err', tb.format_exc()[-900:])
try:
    nq = NE['E']['wcf'].shape[0]
    def qcodes(d, nq):
        for k, v in d.items():
            if isinstance(v, pd.DataFrame) and len(v) == nq:
                for c in ('event','code','name','ev'):
                    if c in v.columns: return k, c, v[c].values
        return None
    qc = {nm: qcodes(NE[nm]['d'], NE[nm]['wcf'].shape[0]) for nm in NE}
    print('qcodes', {nm: (q[0], q[1], pd.Series(q[2]).value_counts().head(8).to_dict()) if q else None for nm, q in qc.items()})
except Exception: print('qc err', tb.format_exc()[-600:])
print('sec', round(time.time()-T0))
```