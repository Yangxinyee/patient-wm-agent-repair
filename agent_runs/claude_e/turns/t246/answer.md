The raw NE-rate readout settles it. On map_arterial rows the head input mostly carries whether NE is running at all. The readout is 0.03 with NE off and 0.35 at rates around 0.05. It then barely rises: 0.59 at about 0.6, and 0.63 at 1.0, with a scatter of about 0.14. The NE rate level has almost no linear presence in what the value head sees, and the deviation that drives a causal response has even less (R² 0.19).

So no head-level edit can produce an NE response without amplifying noise far beyond the factual budget. A clean fix would need edits deep in the core, which my remaining runs cannot support. I am leaving NE unedited.

For insulin, the asymmetric fix would only trade x2 error for x1/2 error at similar cost. The remaining lever is the overall gain:
- at −2.6 (fwd_12 on the same stays): x2 −13.01 against a truth estimate of about −13.18, and x1/2 +5.41;
- at −2.5: −12.57 and +5.21.

Expected cost is about 0.0039 at −2.6, but I have only measured it on one tuning batch (0.00427). I will save `ins_only(-2.6)` and check its factual cost on the batches I have not yet measured at that gain. I'll switch to −2.6 only if every batch stays well under 0.005, roughly ≤0.0045.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 212-227 (restart 6)
- fwd_10 (gated3 NE a_ne=3) by NE rate: 62% of NE reached rows have rate in (.75,1.0] (train NE rate capped at 1.0, 12% of MA rows at exactly 1.0).
  NE readout response wrong-signed at low rates (x2 -.92, half +.80), right for half at high rates (-2.28), x2 at high rates ~0 (OOD 2.0).
- Raw NE-rate readout from head input (MA rows, ridge, CV R2 .51) SATURATES: pred .035 (off), .35 (r~.05), .43, .51, .59, .62, .63 (r=1.0), sd .14.
  -> head input encodes NE presence, hardly NE level/deviation. Any NE head readout = mostly noise. NE abandoned for good (needs core edits; no budget).
- Insulin asymmetry (fwd_16, -2.5): by rate (0,1]: x2 -2.1 half -.6; (1,2]: -12.9/+2.8; (2,3]: -24.7/+16.9. Gate (units 26/41 sum) always on for hi rows (121..194), off (-450..-260) else.
  Readout term vs dev1 slope pos .56, neg .44 (mild). Negative-side boost would trade x2 vs half error at ~same cost -> not done.
- Plan: compare a_ins -2.6 (x2 -13.01, half 5.41 on f3, fwd_12) vs -2.5. Measure factual at -2.6 on tu2, f3, tu1 (tu0 was +.00427).
""")
E26=ins_only(-2.6)
cur={k:load(KMAP[k]).numpy().astype(np.float64) for k in ['W0','b0','W2','TR']}
for k in ['W0','b0','W2','TR']:
    o=ORIG[k] if k in ORIG else None
    print(k,'diff vs orig',float(np.abs(np.asarray(E26[k],dtype=np.float64)-o).max()),'diff vs current saved',float(np.abs(np.asarray(E26[k],dtype=np.float64)-cur[k]).max()))
import torch
for k in ['W0','b0','W2']:
    save_edit(KMAP[k], torch.tensor(np.asarray(E26[k]),dtype=torch.float32))
print(edited())
```