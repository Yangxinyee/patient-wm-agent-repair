The NameError is my own labelling slip. In run 249 I marked torch tensors with a "T" suffix, so the stored prelude attention tensor is `att` (shape [5101, 4, 7, 7]), not `attT`. The question is unchanged: does some prelude head put a large, steady attention weight on the u token at drug bundles? If so, a linear dose channel can be injected through that head's value projection with a near-constant gain. I'll also list the encoder, prelude and value-head tensor shapes I'd need to edit.

```python
at=att.numpy() if torch.is_tensor(att) else np.asarray(att)
print('att', at.shape)
uk=at[:,:,:4,5]   # slot queries -> u key (token 5)
for h_ in range(4):
    v_=uk[:,h_,:].mean(1)
    print('head',h_,'slots->u mean %.3f sd %.3f p10 %.3f p90 %.3f'%(v_.mean(),v_.std(),np.percentile(v_,10),np.percentile(v_,90)),
          'per-slot',np.round(uk[:,h_,:].mean(0),3))
pre=['embed.numeric','encode_action','core.prelude','core.b_in','core.c_out','attr_heads.value_bin','embed_transition.value','query_norm','core.prelude_norm']
print([(n_,tuple(SHAPES[n_])) for n_ in NAMES if any(n_.startswith(p) for p in pre)])
```