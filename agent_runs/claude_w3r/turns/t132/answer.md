Both checks came out well.
- With the norepinephrine dose fixed (dose = 4.8 × rate), my synthetic u matches the real encoder exactly (max difference 8e-7).
- The insulin rate is linearly decodable from encode_action's mlp2 hidden layer, residual sd 0.056 u/h. The norepinephrine rate is decodable with residual sd 0.002. So a rank-1 addition to mlp2's output weights can write each drug's rate linearly into u.
- E has many directions with almost no normal variation (sd ~0.0005). A signal steered there could be amplified into h through a rank-1 edit of core.b_in without amplifying noise.

Next I will size the encoder-side pieces on the rate grid:
- the pre-normalisation vector v that encode_action's norm sees: how much its standard deviation varies (this sets the gain stability of the channel), and how many directions its normal variation spans;
- the same for the prelude-normalised u token that head 2 reads.

What would confirm the plan:
- v's standard deviation is fairly constant;
- both families span clearly fewer than 192 directions, leaving clean orthogonal directions for the insulin and norepinephrine channels.

```python
V_=mlp_t(Sg,'encode_action.mlp2'); stdv=V_.std(1).numpy()
print('pre-LN v std over grid: mean %.3f  min %.3f  max %.3f'%(stdv.mean(),stdv.min(),stdv.max()))
grp_=np.where(Ti==0,'ne',np.where(Tn==0,'ins','both'))
for g_ in ['ins','ne','both']: print('  ',g_,'std v mean %.3f sd %.3f'%(stdv[grp_==g_].mean(),stdv[grp_==g_].std()))
Vc=(V_-V_.mean(0)).numpy(); svv=np.linalg.svd(Vc,compute_uv=False); print('v centred sv (rel):', (svv/svv[0])[[0,5,10,20,30,40,60,80]].round(5))
Ug=torch.stack([LNt(v_,'encode_action.norm') for v_ in V_]+[torch.zeros(192,dtype=torch.float64)])
L1g=LNt(Ug,'core.prelude.0.norm1').numpy(); svl=np.linalg.svd(L1g,compute_uv=False); print('LN1(u) (uncentred) sv (rel):', (svl/svl[0])[[0,5,10,20,30,40,60,80]].round(5))
print('n dims with rel sv > 1e-3: v', int((svv/svv[0]>1e-3).sum()), ' LN1u', int((svl/svl[0]>1e-3).sum()))
print('gamma enc norm: mean %.3f min %.3f max %.3f'%(Wt['encode_action.norm.weight'].mean(),Wt['encode_action.norm.weight'].min(),Wt['encode_action.norm.weight'].max()))
print('gamma prelude norm1: mean %.3f min %.3f max %.3f'%(Wt['core.prelude.0.norm1.weight'].mean(),Wt['core.prelude.0.norm1.weight'].min(),Wt['core.prelude.0.norm1.weight'].max()))
print('gamma prelude_norm: mean %.3f min %.3f max %.3f'%(Wt['core.prelude_norm.weight'].mean(),Wt['core.prelude_norm.weight'].min(),Wt['core.prelude_norm.weight'].max()))
```