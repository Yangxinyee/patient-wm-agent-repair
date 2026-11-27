Norepinephrine is titrated. When MAP is below 65 the rate goes up (mean log-ratio +0.1 to +0.18). From 65 to 75 it holds (about 0). Above 75 it is weaned (-0.25). The step is sharp in the recorded MAP, so the policy reads the recorded MAP. Each hour also carries a large multiplicative scatter, SD about 0.3 in log terms.

Unlike insulin's, this scatter carries forward: a random bump at t persists in later rates through the titration. So the impulse response must be recovered by deconvolving, with the first-stage effect of the shock on later rates and the reduced-form effect on MAP: h_k = total_k − Σ_j h_{k−j}·f_j.

Next I'll define the shock as the residual of the rate given (previous rate × MAP band), excluding capped rates. Then I'll estimate its effect on rates at t+j (first stage, f_j) and on MAP at t+k (reduced form), and check that the shock doesn't persist within a stay as a stay-level trait. A clear positive MAP response with f_j near 1 would confirm a real pressor effect. A flat MAP response would point to little or no norepinephrine effect, despite the strong confounded association.

```python
mband = lambda m: np.digitize(m, [55,60,65,66,70,75,80,90])
rows=[]
for t in range(1,24):
    j=col(t)
    d={'t':t,'m0':ma[:,j],'mc0':mc[:,j],'mlast':mlast[:,j],'mb':mband(mlast[:,j]),'r':ne[:,j],'rp':ne[:,j-1],'stay':np.arange(len(ids))}
    for k in range(1,8):
        jj=j+k
        d[f'm{k}'] = ma[:,jj] if jj<=col(24) else np.nan
        d[f'r{k}'] = ne[:,jj] if jj<=col(24) else np.nan
    rows.append(pd.DataFrame(d))
N = pd.concat(rows, ignore_index=True)
N = N[(N.r>0)&(N.rp>0)&(N.r<1)&(N.rp<1)].copy()
N['lr'] = np.log(N.r/N.rp)
N['lrhat'] = N.groupby('mb').lr.transform('mean')
N['dev'] = N.r - N.rp*np.exp(N.lrhat)   # rate-unit shock (approx)
N['ldev'] = N.lr - N.lrhat
print("ldev sd", N.ldev.std().round(3), " stay-mean var share", (N.groupby('stay').ldev.mean().var()/N.ldev.var()).round(3), "n/stay", N.groupby('stay').size().mean().round(1))
for k in range(1,8):
    s = N.dropna(subset=['m0',f'm{k}',f'r{k}'])
    X=[s.dev.values, s.rp.values, s.m0.values]+[(s.mb==b).values.astype(float) for b in sorted(s.mb.unique())[1:]]
    nm=['dev','rp','m0']+[f'b{b}' for b in sorted(s.mb.unique())[1:]]
    ry = ols((s[f'm{k}']-s.m0).values, X, nm); rf = ols(s[f'r{k}'].values, X, nm)
    print(k, len(s), "MAP resp", ry.loc['dev','b'].round(2), "+-", ry.loc['dev','se'].round(2), " rate resp", rf.loc['dev','b'].round(3), "+-", rf.loc['dev','se'].round(3))
```