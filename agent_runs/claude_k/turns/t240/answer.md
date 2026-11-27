Run 223 shows my z posterior rebuild at drug bundles is not exact (mean error 0.051, max 0.44). Since ctx is exact, either the observation input e at drug bundles is not `E_EMPTY` or my action encoding `u_rec` is slightly off.

One suspect is the drug END atoms. The atom count includes them, and they may be encoded on the observation side. The earlier choice of "dead" units 55 and 109 is also doubtful: they are not among the least active units on drug bundles.

Next I'll break the rebuild error down by atom count and by which drugs are running. I'll also check how strongly the error correlates with the rates. If the error depends on atom count, END atoms belong in e. If it grows with rate, the action encoding is the problem.

```python
errN=np.linalg.norm(zr-zt,axis=1) if False else None
inpE=np.column_stack([CTA[sub_],np.tile(np.asarray(E_EMPTY,dtype=np.float64),(len(sub_),1)),U_dr]); zrE=gelu_np(inpE@Wz0.T+bz0)@Wz2[:32].T+bz2[:32]
errN=np.linalg.norm(zrE-zt,axis=1)
nat=np.concatenate([s['n_atoms'] for s in Sx])[dr[sub_]]
for v in np.unique(nat)[:8]: print('n_atoms',v,'n',int((nat==v).sum()),'mean err',np.round(errN[nat==v].mean(),3))
for a_ in [0,1]:
    for b_ in [0,1]:
        m_=((riA[sub_]>0)==a_)&((rnA[sub_]>0)==b_); print('ins>0',a_,'ne>0',b_,'n',int(m_.sum()),'err',np.round(errN[m_].mean(),3) if m_.any() else '-')
print('corr err-ri',np.round(np.corrcoef(errN,riA[sub_])[0,1],3),'corr err-rn',np.round(np.corrcoef(errN,rnA[sub_])[0,1],3))
```