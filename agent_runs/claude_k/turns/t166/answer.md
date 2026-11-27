The writer edit reached z18 and z28 on drug bundles, but it doesn't behave as I planned:
- **Insulin bundles:** dz18 ≈ 0.36×rate − 0.18 (corr 0.91), not the clean 0.29×rate I expected.
- **Norepinephrine bundles:** dz28 tracks the rate only loosely (slope 1.3, corr 0.58).
- **Bundles with no drug:** z18 and z28 still shift by about 1.5 and 1.0. If the cancelling bias worked, those shifts should be zero.
- Other z dims barely move (max 0.003), so the leak is confined to the two channel dims.

Two explanations fit:
1. The edit itself isn't neutral when u = 0 (wrong bias or a wrong input block).
2. A feedback loop: z18 is written into the state h, and the posterior for dim 18 at later bundles reads it back from ctx.

To tell them apart I'll inspect the edited weights directly and print one stay's trajectory. If dz18 on non-drug bundles is about constant everywhere, including before the first dose, it's explanation 1. If it starts after the first insulin and builds or decays with dosing, it's explanation 2.

```python
W0=load('z_post.0.weight'); b0=load('z_post.0.bias'); W2=load('z_post.2.weight'); b2=load('z_post.2.bias')
W0o=load_original('z_post.0.weight'); b0o=load_original('z_post.0.bias'); W2o=load_original('z_post.2.weight'); b2o=load_original('z_post.2.bias')
for u_ in [55,109]:
    w=W0[u_]; print('unit',u_,'b0',round(float(b0[u_]),3),'|w ctx|',round(float(np.abs(w[:192]).sum()),4),'|w e|',round(float(np.abs(w[192:384]).sum()),4),'|w u|',round(float(np.linalg.norm(w[384:])),2),
          'W2 col nonzero rows',np.nonzero(np.abs(W2[:,u_])>1e-6)[0][:6], np.round(W2[np.abs(W2[:,u_])>1e-6,u_],3))
print('b2 change rows', np.nonzero(np.abs(b2-b2o)>1e-6)[0], np.round((b2-b2o)[np.abs(b2-b2o)>1e-6],3))
print('W0 changed rows', np.nonzero(np.abs(W0-W0o).sum(1)>0)[0], 'W2 changed cols', np.nonzero(np.abs(W2-W2o).sum(0)>0)[0])
m0=~hi&~hn
first_ins=np.full(len(sid8),np.inf)
for s in np.unique(sid8):
    mm=(sid8==s)&hi
    if mm.any(): first_ins[sid8==s]=hr8[mm].min()
pre_m=m0&(hr8<first_ins); post_m=m0&(hr8>first_ins)
print('nondrug before first insulin: n',pre_m.sum(),'mean dz18',round(dz[pre_m,18].mean(),3),'sd',round(dz[pre_m,18].std(),3),'| after: n',post_m.sum(),'mean',round(dz[post_m,18].mean(),3),'sd',round(dz[post_m,18].std(),3))
s0_=sid8[hi][0]; mm=np.nonzero(sid8==s0_)[0]
print('stay',s0_); print(np.round(np.c_[hr8[mm],np.nan_to_num(ri[mm]),dz[mm,18],np.nan_to_num(rn[mm]),dz[mm,28]][:44],2).T)
dout=np.linalg.norm(X8e[:,:192]-X8o[:,:192],axis=1); dh=np.linalg.norm(X8e[:,192:960]-X8o[:,192:960],axis=1)
print('|out| ',round(np.linalg.norm(X8o[:,:192],axis=1).mean(),2),'d out mean',round(dout.mean(),3),'| |h|',round(np.linalg.norm(X8o[:,192:960],axis=1).mean(),2),'dh mean',round(dh.mean(),3))
```