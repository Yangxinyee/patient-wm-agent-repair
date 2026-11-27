The prelude-FFN gate idea fails:
- On the prelude FFN input, the insulin gate separates only barely (0.27 against 0.23).
- The NE gate overlaps badly.
- The residual's sd at drug bundles varies with a CV of 0.185, so every gated write would carry roughly 18–26% multiplicative noise.

Any gate also carries a robustness risk. The raw head-3 staging can be 30–150 times larger away from drug bundles than on them, so a single gate miss would wreck the memory.

**A gate-free channel.** Instead of decoding the rate from head 3's mixed output, put a clean rate signal into the drug token itself:
- Give the action encoder an output dim jWI equal to a rate decoder, exactly zero for the empty set.
- Repurpose one value channel of head 3 to read only the difference (LN1 at jWI − LN1 at jR).

For slot tokens both dims are killed and zero, and z-token rows there are irrelevant, so the channel outputs exactly zero for every token except u. Head 3 attends u with weight 1.0 at drug bundles, so the staging would be the rate there and about zero elsewhere, with no gate at all.

The one gap is e tokens: their jWI/jR components are original encoder content. So this works only if head 3 puts negligible weight on the e and z tokens at non-drug bundles.

This run measures head 3's attention, split by token, in drug and non-drug bundles. How I will read it:
- e and z weights below about 1e-3 at non-drug bundles make the clean channel exact.
- If they are larger, I need e's two dims equalised as well.

```python
drugF=(RS['fac']>0)|isNE
A3=[]
for i in range(0, len(STS['fac']['hf']), 2000):
    s=slice(i,i+2000); pr=prelude(P2, STS['fac']['hf'][s], dsE['e'][s], dsE['u'][s], STS['fac']['z'][s])
    A3.append(pr['att'][:,3,:4,:])
A3=torch.cat(A3).numpy()   # B x 4 slots x 7 tokens
for nm,m in (('drug',drugF),('nondrug',~drugF)):
    a=A3[m]; print(nm, 'head3 weight: slots mean', round(a[:,:,:4].sum(-1).mean(),4), '| e mean/p99/max', np.round([a[:,:,4].mean(), np.percentile(a[:,:,4],99), a[:,:,4].max()],4), '| u mean/min', np.round([a[:,:,5].mean(), a[:,:,5].min()],4), '| z mean/p99/max', np.round([a[:,:,6].mean(), np.percentile(a[:,:,6],99), a[:,:,6].max()],4))
```