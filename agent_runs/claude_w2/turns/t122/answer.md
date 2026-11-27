With finger glucose on unit row 0, lab glucose on row 5 and MAP on row 2, all reading bundles are now exact (error 0.0000). Action bundles still show 0.0016, and the search wavers between unit rows 2 and 4 for insulin and puts insulin's rate unit on row 0. That looks like compensation for another small error in the drug atoms, which is exactly the pathway this task depends on.

Next I split the action-bundle error by content (insulin only, norepinephrine only, both, with or without an END atom). I also try every combination of insulin unit row (2, 4) and rate-unit row (0, 2), and of norepinephrine unit row (4, 5) and rate-unit row (0, 3). An exact combination fixes the mapping. If the error persists only in bundles with END atoms, the END-atom treatment, not the unit rows, is the issue.

```python
key0=A0n.stay_id.values.astype(np.int64)*100000+A0n.bidx.values
uk0,gid0=np.unique(key0,return_inverse=True)
def bset(cond):
    s=np.zeros(len(uk0),bool); s[gid0[cond]]=True; return s
has_ins=bset((A0n.code.values==8)&(A0n.kind.values==0)); has_ne=bset((A0n.code.values==9)&(A0n.kind.values==0))
has_end=bset(A0n.kind.values==2)
cat=np.where(has_ins&has_ne,'both',np.where(has_ins,'ins',np.where(has_ne,'ne','none')))
cat=np.char.add(cat,np.where(has_end,'+E','')); cat_s=cat[rowB][nf]
def zerr_cat(E,U):
    zm=zpost_mean(ctx_,E[rowB][nf],U[rowB][nf],WO); ab=(zm-Zs[nf]).abs().mean(1).numpy()
    return {c:round(float(ab[cat_s==c].mean()),5) for c in np.unique(cat_s) if c.startswith(('ins','ne','both'))}
rows_=[]
for iu in [2,4]:
    for ir in [0,2]:
        for nu in [4,5]:
            for nr in [0,3]:
                um_=dict(um); um_[8]=iu; um_[9]=nu; rm_=dict(rmp); rm_[8]=ir; rm_[9]=nr
                rows_.append(((iu,ir,nu,nr),zerr_cat(*encode_from_xa(A0n,xa_with(um_,rm_),WO))))
for k_,v_ in sorted(rows_,key=lambda r: sum(r[1].values()))[:6]: print(k_,v_)
```