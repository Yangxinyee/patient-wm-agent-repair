On the full 100 stays, freeing dimensions 49 and 185 raises the NLL from 1.5279 to 1.5313, i.e. +0.0034 per reading, mostly on code-6 and code-10 readings. That fits under the 0.005 limit, but it leaves too little margin for two channels plus their injection. I need cheaper dimensions.

Next I'll split each dimension's influence into a core part (state spread times reader weights) and a head part. The head part is the spread of `out_j` times the `query_norm` gamma times the `value_bin.0` column norm. Dimensions that are low on both should cost less to free. If some score far below 49 and 185 on the combined measure, I'll test those.

```python
O=X[:,:192]; sdo=O.std(0)
q=np.abs(W('query_norm.weight')); w0=np.linalg.norm(W('attr_heads.value_bin.0.weight'),axis=0)
zp=np.linalg.norm(W('z_prior.0.weight')[:,:192],axis=0)+np.linalg.norm(W('z_post.0.weight')[:,:192],axis=0)
hd2=sdo*q*w0; zc=sdo*zp
core=score/np.median(score); head=hd2/np.median(hd2); zz=zc/np.median(zc)
tot=core+head+zz; o=np.argsort(tot)
print('pair 49,185 tot',np.round(tot[[49,185]],2),'median',np.round(np.median(tot),2))
for j in o[:10]: print(j,'core %.2f head %.2f z %.2f tot %.2f sdo %.3f'%(core[j],head[j],zz[j],tot[j],sdo[j]))
```