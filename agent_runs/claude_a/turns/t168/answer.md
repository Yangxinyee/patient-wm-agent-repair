The architecture recap fills in the head path. View = out + out_norm(slot mean of h) + gap + z_head, then q = query_norm(view + type_emb), and with_code = q + code embedding + last-value terms. The value MLP plus embed_transition[code·17 + last bin] gives 17-bin logits.

The readout is the predicted mean (not the drawn value) at glucose_finger readings, edited minus factual, at hours after the drug ran.

My plan (insulin first):
- **Memory:** a persistent memory dim holding insulin rate minus the policy rate implied by the triggering glucose. On factual data this is mean-zero dose noise, so factual fit should hold; under ×2 it carries the extra dose.
- **Readout:** a value-MLP unit gated to glucose_finger only. The gate uses differences between killed dims, which depend only on code and last bin, so their constant offsets can be cancelled exactly in embed_transition.

First I'll measure the factual cost of the kill scaffold `PS` against the original. I use my offline factual-NLL function, which should reproduce the tool's 1.40394. If killing the 8 dims costs well under 0.005, the scaffold is affordable.

```python
t0=time.time(); r0=tf_nll(PO, ret=True); t1=time.time()
print('type', type(r0), (len(r0) if isinstance(r0,(tuple,list)) else ''), '%.1fs'%(t1-t0))
if isinstance(r0,(tuple,list)):
    for i,x in enumerate(r0): print(i, type(x).__name__, getattr(x,'shape',None), (float(x) if np.isscalar(x) or (hasattr(x,'numel') and x.numel()==1) else ''))
nll0=r0[0] if isinstance(r0,(tuple,list)) else r0
rS=tf_nll(PS, ret=True); nllS=rS[0] if isinstance(rS,(tuple,list)) else rS
print('nll PO %.5f  PS %.5f  diff %.5f'%(float(nll0), float(nllS), float(nllS)-float(nll0)))
```