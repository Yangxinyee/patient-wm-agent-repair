The offline evaluator works again after the fix, but it has two problems.

**An artefact.** Even the zero edit, `gated3(0,0)`, shows nonzero changes against the stored baseline. It moves map_cuff rows by 0.0015–0.0032, and the full edit changes those rows by the same amount, where the forward tool showed no change. So the stored baseline is offset from my local head, and costs must be read as `gated3(a) − gated3(0,0)`.

**Offline costs are optimistic.** Compared that way, `gated3(−1.9,3)` costs per row offline:

| Rows | Offline cost |
|---|---|
| glucose_finger after insulin | +0.0034 |
| map_arterial after NE | +0.0152 |
| map_arterial without NE | +0.0047 |
| lab glucose after insulin | +0.0015 |

The tool on tuning stays measured +0.0062 per glucose_finger reading over all glucose_finger rows, about twice the offline figure. The offline datasets are the same train stays the readouts were fitted on, so they are in-sample and understate the readout noise.

I need an honest out-of-sample offline evaluator before choosing a larger insulin gain. The sel1 tuning stays have states (`fwd_2`), so I'll check that their reading table `RR` has the same format as the train datasets and see how `base_out` is built.

```python
d0s, d0r = datasets[0]
print(type(d0s), getattr(d0s,'shape',None), type(d0r), getattr(d0r,'shape',None))
if hasattr(d0r,'columns'): print('dataset rows cols:', list(d0r.columns))
print('RR', type(RR), getattr(RR,'shape',None), (list(RR.columns) if hasattr(RR,'columns') else None))
print('X', getattr(X,'shape',None))
bo = base_out[0]; print('base_out[0]', type(bo), len(bo), [getattr(t,'shape',type(t)) for t in bo])
print('A_ cols', list(A_.columns) if hasattr(A_,'columns') else type(A_), getattr(A_,'shape',None))
co = eval_edit.__code__; print('eval_edit consts', [c for c in co.co_consts if isinstance(c,(str,int,float))][:30])
co = head_run.__code__; print('head_run vars', co.co_varnames[:30])
print('stays_s1', len(stays_s1), 'sel1', len(sel1), 'f3_ids', len(f3_ids) if hasattr(f3_ids,'__len__') else f3_ids)
print('overlap sel1&f3', len(set(map(int,sel1)) & set(map(int,f3_ids))))
```