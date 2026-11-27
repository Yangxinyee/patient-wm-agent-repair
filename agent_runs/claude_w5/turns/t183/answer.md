The record follows a fixed hourly pattern:
- readings (glucose finger, arterial and cuff MAP) at h:00;
- drug bundles at h:05, holding the END atoms of the previous infusions and the new starts.

Drug bundles never contain readings. So when a drug bundle is processed, the state already holds the h:00 reading that set the dose, and the next glucose prediction, at (h+1):00, is made directly from the state after the drug bundle.

**Design:** edit the output matrix of the last block's FFN (`core.blocks.0.1.ffn.2.weight`). Its output is added straight into h_new, so a low-rank term P·Rᵀ gives Δh_new = P·(Rᵀ·hidden) with no further transformation inside the step.
- **Gate (R):** a ridge readout of that FFN's 768 hidden units. Target: dose minus policy-expected dose at drug bundles, including synthetic x2 and x1/2 versions so it extrapolates; 0 elsewhere.
- **Direction (P):** the h_new change that a lower glucose reading at h:00 would produce at the next drug bundle.

Next I need the head helpers' formats (`H`, `V0`, `fixed`, `TRr`, `probs_from_v0`). Then I can measure how much a state-only change moves the next predicted glucose.

```python
for k in ['H','V0','fixed','TRr','cen']:
    o=globals()[k]; print(k,type(o).__name__,getattr(o,'shape',len(o) if hasattr(o,'__len__') else ''))
if hasattr(H,'columns'): print(list(H.columns)[:20]); print(H.head(2).to_string()[:400])
c=probs_from_v0.__code__; print(c.co_varnames); print(c.co_names); print([x for x in c.co_consts if isinstance(x,(str,int,float))])
```