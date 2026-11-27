The difference between `gated3(-2.7, 2)` and the saved `ins_only(-2.7)` is confined to head units 131 and 159: their input rows, biases and output columns. The TR table is identical. So a combined edit is just the current edit plus a separable NE unit pair, and the saved edit is confirmed to be exactly `ins_only(-2.7)`.

The head is 192 → 192 → 17. Every repurposed unit loses its original contribution. This was measured as costing .0004–.0013 for the four units together.

Two levers might make room for an NE part within the budget:
1. Pick a moderate NE gain.
2. Compensate the lost contribution of the repurposed units, both the insulin pair 26/41 and the NE pair 131/159. I would do this by regressing each unit's logit contribution on the other units' activations and a constant over factual rows. That is a small fitted statistical model, folded back into W2 and the output bias.

First I need the offline evaluator's output format and a baseline. I will run `head_run` on the sel1 rows with the original edit and with `ins_only(-2.7)`, then check that the per-event cost matches my earlier offline numbers.

```python
import time
print([k for k in ['X','RR','Xs','R1','XA','RA'] if k in globals()])
print(type(X), getattr(X,'shape',None)); print(RR.columns.tolist()[:30], len(RR))
R1 = RR[RR['src']=='1']
orig = {k: load_original(n) for k,n in KMAP.items()}
t=time.time(); o = head_run(X, R1.iloc[:200], orig); print('t', round(time.time()-t,2))
print(type(o))
if isinstance(o, dict): print({k: getattr(v,'shape',type(v)) for k,v in o.items()})
elif isinstance(o, (tuple,list)): print([getattr(v,'shape',type(v)) for v in o])
else: print(getattr(o,'shape',None), o[:3] if hasattr(o,'__getitem__') else o)
```