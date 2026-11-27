Run 252 locates the problem in the encoder. When the insulin rate changes, u changes roughly in proportion up to ×2, then flattens (the ×3 change per unit is about half). The ×2 and ×0.5 changes point in nearly unrelated directions (cos +0.16 instead of −1). Downstream stages show the same pattern (cos −0.3).

So the model stores dose like a category (separate bin embeddings per rate), not along one line. No linear readout of the state can respond correctly to both raised and lowered doses. The repair therefore needs a new, linear dose channel that starts at the one linear dose input, the numeric z_rate feature. It must be carried through the encoder MLPs and LayerNorm into u, then through the prelude attention into the slots and out. Finally, gated units in the value head would turn it into a glucose or MAP shift.

The weak point is the prelude attention: the channel reaches the slots only through each head's attention weight on the u token. That gives a usable gain only if some head attends to u strongly and steadily at drug bundles. This run checks that from the stored prelude attention (`attT`), and lists the encoder and prelude tensor shapes I would edit. A head with high, low-variance attention to u at drug bundles confirms the route. Only small or erratic weights would refute it.

```python
at=attT if isinstance(attT,np.ndarray) else attT.numpy()
print('attT', at.shape, 'au_iT', np.shape(au_iT), 'au_nT', np.shape(au_nT))
uk=at[:,:,:4,5]   # slots->u key
for h_ in range(4):
    v_=uk[:,h_,:].mean(1)
    print('head',h_,'att slots->u mean %.3f sd %.3f p10 %.3f p90 %.3f'%(v_.mean(),v_.std(),np.percentile(v_,10),np.percentile(v_,90)),
          ' per-slot means',np.round(uk[:,h_,:].mean(0),3))
pre=['embed.numeric','encode_action','core.prelude','core.b_in','core.c_out','core.coda','attr_heads.value_bin','embed_transition.value','query_norm','core.out_norm','core.prelude_norm']
print([ (n_,tuple(SHAPES[n_])) for n_ in NAMES if any(n_.startswith(p) for p in pre)])
```