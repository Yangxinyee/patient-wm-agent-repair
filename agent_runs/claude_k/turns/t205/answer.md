The benchmark separates the two drugs sharply.

**Insulin.** Policy features reach R² 0.73 (residual SD 0.62). The linear ctx decoder reaches 0.71 (residual SD 0.65), and the two predictions correlate at 0.95. So the writer can form the insulin innovation from ctx with an extra decode error of only about 0.19 units/h.

**Norepinephrine.** Policy features reach R² 0.87 (innovation SD 0.13). ctx reaches only 0.59 (residual 0.23), and the previous rate decodes from ctx with R² 0.62. The original state simply does not carry the previous norepinephrine rate in a form z_post can read.

A norepinephrine innovation built from ctx alone would mostly carry decode error. My estimate is that this costs several hundredths of a nat on MAP readings at full gain, or leaves the effect badly attenuated. So norepinephrine needs a carried channel that makes the previous rate readable at the next drug bundle.

I already have data on exactly that: fwd8 wrote a norepinephrine level signal (1.83 × rate into z28) at every drug bundle. The constant offsets from the bug do not matter here.

Test: decode the previous rate from the **edited** ctx and from the change in ctx, then compare with the original's R² of 0.62.
- If edited-ctx R² rises to about 0.95 or more, a level channel makes the norepinephrine innovation computable in the writer.
- If it barely moves, writes into z are washed out by the state update, and I need a different route.

This also gives the first direct look at how a z write propagates into the next ctx: its size, and how linear it is in the rate.

```python
F8o=np.load('/work/fwd_8/states_original.npz'); F8e=np.load('/work/fwd_8/states_edited.npz')
print('same rows', np.array_equal(F8o['hour'],F8e['hour']), np.array_equal(F8o['stay_id'],F8e['stay_id']), F8o['x'].shape)
def drug_rows(S):
    st=S['stay_id'];hr=S['hour']
    return np.array([k for k in range(1,len(hr)) if st[k]==st[k-1] and abs((hr[k]%1)-0.0833)<0.01 and np.floor(hr[k])<=24])
k8=drug_rows(F8o); xo=F8o['x'].astype(np.float64); xe=F8e['x'].astype(np.float64)
hr8_=F8o['hour'][k8]; st8_=F8o['stay_id'][k8]; dt8_=(F8o['hour'][k8]-F8o['hour'][k8-1])*3600
Co=np.array([ctx_from(xo[k-1],d) for k,d in zip(k8,dt8_)]); Ce=np.array([ctx_from(xe[k-1],d) for k,d in zip(k8,dt8_)])
h8=np.floor(hr8_).astype(int); r8=np.array([sid2i[s] for s in st8_])
Dn=ne_all[r8,h8]; Dn1=np.where(h8>=1,ne_all[r8,np.maximum(h8-1,0)],0.); Dn2=np.where(h8>=2,ne_all[r8,np.maximum(h8-2,0)],0.)
Di=ins_all[r8,h8]; Di1=np.where(h8>=1,ins_all[r8,np.maximum(h8-1,0)],0.)
gk=GroupKFold(5)
def cv(X,y,al=0.1):
    p=np.zeros_like(y)
    for a,b in gk.split(X,y,st8_): p[b]=Ridge(alpha=al).fit(X[a],y[a]).predict(X[b])
    return 1-((y-p)**2).mean()/y.var()
dC=Ce-Co
print('n',len(k8),'|dctx| mean %.3f  |ctx| mean %.3f  ctx sd/dim %.3f'%(np.linalg.norm(dC,axis=1).mean(),np.linalg.norm(Co,axis=1).mean(),Co.std(0).mean()))
for nm,y in [('Dne_prev',Dn1),('Dins_prev',Di1),('Dne_prev2',Dn2)]:
    print(nm,'R2 from ctx_orig %.3f | ctx_edit %.3f | dctx %.3f'%(cv(Co,y),cv(Ce,y),cv(dC,y)))
# how much of dctx is explained linearly by lagged doses (+const)
Xl=np.c_[np.ones(len(k8)),Dn1,Dn2,Di1,np.where(h8>=2,ins_all[r8,np.maximum(h8-2,0)],0.)]
B_,*_=np.linalg.lstsq(Xl,dC,rcond=None); res_=dC-Xl@B_
print('dctx var explained by lagged doses: %.3f'%(1-res_.var(0).sum()/dC.var(0).sum()))
print('norm of response per unit: NE lag1 %.3f lag2 %.3f | INS lag1 %.3f lag2 %.3f | const %.3f'%tuple(np.linalg.norm(B_[[1,2,3,4,0]],axis=1)))
```