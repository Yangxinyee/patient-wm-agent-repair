The 48-dim head-3 value of the drug token decodes the insulin rate badly. Fitted jointly on all four dose versions, the slopes are only 0.15–0.39, with residual sd 0.46–1.7 units/h. That contradicts my earlier note (R² 0.995 on factual doses).

The likely cause is the encoder, not the extraction. At doubled rates the atom features may saturate: rate bins top out, and the robust z-score may be clipped. If so, no linear decoder can tell rate 5 from rate 9, and the joint fit drags down the factual slope as well.

To test this I will:
- print VOCAB's numeric note and insulin entry, looking for clipping or binning of rate;
- encode one real insulin-only bundle at rates on a grid from 0.1 to 12, and fit a linear decoder from head 3's value along that curve;
- refit on the factual doses alone, to compare with my earlier R² of 0.995.

How I will read it:
- If the grid shows the encoding flattening above some rate, the memory cannot be written from u alone beyond that point. I would then have to accept saturation or use a nonlinear inverse.
- If the factual-only fit is good and the grid is smooth, a range-weighted decoder may be enough.

```python
print(str(VOCAB.get('numeric_note'))[:400])
nm=VOCAB['numeric']; print({k:nm[k] for k in list(nm)[:1]} if isinstance(nm,dict) else str(nm)[:300])
print('insulin numeric:', {k:v for k,v in nm.items() if 'insulin' in k} if isinstance(nm,dict) else '')
# grid encoding of one insulin-only bundle
ins_rows=set(at[at.event=='insulin']['row']); ne_rows=set(at[at.event=='norepinephrine']['row'])
r0=sorted(ins_rows-ne_rows)[10]; sub=at[at.row==r0].copy()
grid=np.array([.1,.25,.5,1,1.5,2,3,4,5,6,7,8,10,12.])
subs=[]; 
for i,g_ in enumerate(grid):
    s_=sub.copy(); s_['rate']=g_; s_['dose']=g_; s_['row']=i; subs.append(s_)
sg=pd.concat(subs); eg,ug=encode_bundles(sg, np.arange(len(grid)).astype(np.int64))
ug=torch.as_tensor(np.asarray(ug),dtype=torch.float32)
n1=LNx(ug, P2, 'core.prelude.0.norm1'); vg=(n1@P2['core.prelude.0.attn.kv.weight'][192:].T)[:,144:192].numpy()
print('||u(r)-u(1)||', np.round(np.linalg.norm(ug.numpy()-ug.numpy()[3],axis=1),2))
print('||v3(r)-v3(r_prev)||', np.round(np.r_[0,np.linalg.norm(np.diff(vg,axis=0),axis=1)],2))
m=RS['fac']>0; X1=np.c_[V3['fac'][m],np.ones(m.sum())]; w_=np.linalg.lstsq(X1,RS['fac'][m],rcond=None)[0]
print('fac-only decode resid sd', round(np.std(X1@w_-RS['fac'][m]),3), 'R2', round(1-np.var(X1@w_-RS['fac'][m])/np.var(RS['fac'][m]),4))
```