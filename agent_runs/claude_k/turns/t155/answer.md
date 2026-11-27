My action encoder rebuild is consistent: the empty measurement set matches exactly. The END-atom mode makes no difference, and the posterior mean of z is reproduced with mean absolute error 0.052, the same level as for reading bundles earlier. That residual looks like a small context mismatch rather than an error in u_k.

It doesn't matter much for the injection plan, for two reasons:
- The writer adds λ·(p·u) through its own repurposed unit, independently of the other units.
- The tool's state files include the edited model's posterior z, so I can check exactly what was written.

Next I'll fit the writer's readout p on u_k over the drug bundles of the 99 state stays:
- insulin rate from u_k;
- norepinephrine rate from u_k;
- linearity under doubled doses (`drug_atoms(..., mult=(2,1))` and `(1,2)`), since the evaluation doubles, halves, triples and zeroes doses.

A cross-validated R² near 1 and doubled-dose decodes near 2× the rate would confirm the writer works. Strong saturation for large doses would mean adding a correction, such as a second unit, for the high range.

```python
print('H', H if 'H' in globals() else None, ' ins shape', ins.shape, ' ne_all shape', ne_all.shape)
rows_all=[j for j in np.where(drg)[0] if j>0]
U1=[];U2=[];U3=[];Y=[]
for j in rows_all:
    i=sid2i[int(sid2[j])]; t=int(np.floor(hr2[j]))
    U1.append(setenc_my(drug_atoms(i,t,'both'),'encode_action'))
    U2.append(setenc_my(drug_atoms(i,t,'both',mult=(2.0,1.0)),'encode_action'))
    U3.append(setenc_my(drug_atoms(i,t,'both',mult=(1.0,2.0)),'encode_action'))
    Y.append([ins_all[i,t] if 0<=t<25 else np.nan, ne_all[i,t] if 0<=t<25 else np.nan])
U1,U2,U3,Y=map(np.array,(U1,U2,U3,Y)); ok=~np.isnan(Y).any(1); print('n drug bundles',ok.sum())
from sklearn.model_selection import cross_val_predict
for c,nm_ in [(0,'insulin'),(1,'norepi')]:
    pr=cross_val_predict(Ridge(alpha=1e-3),U1[ok],Y[ok,c],cv=5); m=Ridge(alpha=1e-3).fit(U1[ok],Y[ok,c])
    on_=Y[ok,c]>0
    print(nm_,'CV R2 %.4f'%(1-((pr-Y[ok,c])**2).mean()/Y[ok,c].var()),
          '| decode x2 (ins edit) mean ratio on-rows %.3f'%((m.predict(U2[ok][on_])-m.intercept_).mean()/(Y[ok,c][on_].mean()) if c==0 else (m.predict(U3[ok][on_])-m.intercept_).mean()/(Y[ok,c][on_].mean())),
          '| cross: other-drug x2 shifts decode by %.4f'%((m.predict(U3[ok] if c==0 else U2[ok])-m.predict(U1[ok])).mean()), '| intercept %.3f'%m.intercept_)
```