This resolves the composition question. These bundles really do contain an END atom (n_atoms = 2). But the encoding that matches the tool leaves it out: start atom only, error 0.029 and 0.007, the same as genuine single-atom bundles. So **END atoms do not enter the action encoder**, and u is built from the start atoms of the bundle only. A bundle with no start atoms then gives U_EMPTY.

Next I'll check that this corrected rebuild holds on all 3305 drug bundles in fwd_8, including those with both drugs (two start atoms, so √2 and size(log 3) apply). The rule is:
- insulin start atom with unit 3 / rate_unit 2 / route 1;
- norepinephrine start atom with unit 4 / rate_unit 3 / route 1;
- no END atoms.

If the error stays at ~0.03–0.06 in every composition group, the rebuild is exact enough to refit the decoders offline, including for doubled and tripled doses.

```python
def u_rec(ri_, rn_, mi=1.0, mn=1.0):
    at=[]
    if ri_>0: at.append(datom(8, ri_*mi))
    if rn_>0: at.append(datom(9, rn_*mn))
    return np.asarray(setenc_my(at,'encode_action')) if at else U_EMPTY.copy()
ucache={}
def u_c(a,b,mi=1.0,mn=1.0):
    k_=(round(float(a),4),round(float(b),4),mi,mn)
    if k_ not in ucache: ucache[k_]=u_rec(a,b,mi,mn)
    return ucache[k_]
fin=np.isfinite(imp_ins[ok_])&np.isfinite(imp_ne[ok_])
PI=np.array([p_ins_@u_c(a,b) for a,b in zip(cur_i,cur_n)]); PN=np.array([p_ne_@u_c(a,b) for a,b in zip(cur_i,cur_n)])
eI=PI-imp_ins[ok_]; eN=PN-imp_ne[ok_]
grpdf['eI']=eI; grpdf['eN']=eN; grpdf['fin']=fin
g_=grpdf[grpdf.fin].groupby(['ins_now','ne_now']).agg(n=('eI','count'),rmseI=('eI',lambda x: np.sqrt(np.mean(x**2))),rmseN=('eN',lambda x: np.sqrt(np.mean(x**2))),biasI=('eI','mean'),biasN=('eN','mean'))
print(g_.round(3))
print('overall rmseI',np.round(np.sqrt(np.mean(eI[fin]**2)),3),'rmseN',np.round(np.sqrt(np.mean(eN[fin]**2)),3),'n nonfinite',(~fin).sum())
print('distinct u configs',len(ucache))
```