The insulin readout from u_k is factually excellent (cross-validated R² 0.990, norepinephrine 0.96). But two problems show up for the writer:
- **Intercept.** The fit has intercept −10.4. Since u_k = 0 for bundles without drugs, a writer computing p·u would inject a large offset into every drug bundle, including those with no insulin. The decoder must be fitted without an intercept.
- **Cross-talk.** Doubling norepinephrine shifts the insulin decode by +0.5 units/h, so a norepinephrine edit would wrongly inject insulin signal.

The doubling ratios printed in run 142 were computed incorrectly; I'm recomputing them properly here.

The fix is to fit each decoder on an augmented set of encodings that includes dose-edited versions of every drug bundle:
- insulin ×0, ½, 1.5, 2, 3 with norepinephrine fixed;
- norepinephrine ×0, ½, 2, 3 with insulin fixed.

Targets are multiplier × recorded rate. Then p·u should track the edited rate and ignore the other drug.

Per-variant R² near 1, decode-to-target ratios near 1 in every variant, and small cross-shifts would confirm a usable writer. Systematic deviation at ×3 or ×0 would show where the encoder saturates.

```python
variants=[(1,1),(0,1),(0.5,1),(1.5,1),(2,1),(3,1),(1,0),(1,0.5),(1,2),(1,3)]
UA=[];YA=[];VA=[]
for j in rows_all:
    i=sid2i[int(sid2[j])]; t=int(np.floor(hr2[j]))
    if not (0<=t<25): continue
    for vi,(mi,mn) in enumerate(variants):
        UA.append(setenc_my(drug_atoms(i,t,'both',mult=(float(mi),float(mn))),'encode_action'))
        YA.append([ins_all[i,t]*mi, ne_all[i,t]*mn]); VA.append(vi)
UA,YA,VA=np.array(UA),np.array(YA),np.array(VA); print('aug n',len(UA))
dec={}
for c,nm_ in [(0,'insulin'),(1,'norepi')]:
    m=Ridge(alpha=1e-4,fit_intercept=False).fit(UA,YA[:,c]); dec[nm_]=m.coef_.copy()
    pr=UA@m.coef_; print(nm_,'overall R2 %.4f'%(1-((pr-YA[:,c])**2).mean()/YA[:,c].var()), '|p|=%.2f'%np.linalg.norm(m.coef_))
    for vi,v in enumerate(variants):
        s_=VA==vi; on_=YA[s_,c]>0
        print('   ',v,'rmse %.3f'%np.sqrt(((pr[s_]-YA[s_,c])**2).mean()), 'ratio(on) %.3f'%(pr[s_][on_].mean()/max(YA[s_,c][on_].mean(),1e-9)) if on_.any() else '', 'mean dec(off) %.3f'%pr[s_][~on_].mean())
```