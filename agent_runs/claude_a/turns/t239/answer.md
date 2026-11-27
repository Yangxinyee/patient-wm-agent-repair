Head 3 attends the u token at most drug bundles (mean weight about 0.91–0.94) and almost never elsewhere (≤0.023). But the weight drops to 0 at some (bundle, slot) pairs, which breaks both the gate and the rate decoder (insulin R² 0.22). I suspect it depends on the slot, or on whether both drugs are in the bundle. If one or more slots reliably attend u, the write can come from those slots only. Next I'll break the attention weight down by slot and bundle type, for the original model and the dimension-cut base.

- **Usable:** some slot shows weight >0.99 at nearly all drug bundles.
- **Not usable:** the failures are spread across all slots.

```python
with torch.no_grad(): prO=prelude(PO, st['hf'], dsE['e'], dsE['u'], st['z'])
a3uO=prO['att'][:,3,:4,5].numpy()
both=ins&ne; ionly=ins&~ne; nonly=ne&~ins
for nm,A_ in [('PK',a3u),('PO',a3uO)]:
    for gn,mk in [('ins-only',ionly),('both',both),('NE-only',nonly)]:
        print(nm,gn,mk.sum(),' '.join('s%d:%.2f/%.2f'%(k,(A_[mk,k]>0.99).mean(),(A_[mk,k]<0.01).mean()) for k in range(4)))
```