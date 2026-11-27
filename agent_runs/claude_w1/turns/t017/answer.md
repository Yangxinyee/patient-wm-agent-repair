A random +1 u/h insulin shock at hour s lowers glucose by 2.1 mg/dL at s+1 and 2.3 at s+2. The effect then fades: 1.9, 1.4, 0.8, 0.5, 0.3, … and is about 0 by 10–12 hours. So the effect is transient: it peaks about 2 hours after the dose and decays with a time constant of a few hours.

Two other effects are small:
- **Clinician feedback on later doses:** −0.06 u/h, because the scale's steps are coarse.
- **Selection through measurement:** −0.005.

Total effects are therefore close to the fixed-dose impulse response the readout needs. A rough estimate: doubling a constant 2 u/h infusion from hour 12 would lower glucose by about 15 mg/dL on average over hours 13–24. Halving would raise it by about 8.

Before trusting this, I need to know whether the effect is linear in dose and constant across glucose levels. Doubling is a much bigger change than the natural ±35% dose noise, so the functional form matters. Next I'll split the shock effect by base level (glucose band), and test curvature in the dose: shock² terms, and log dose against linear dose.

```python
def lp(S, k, extra_cols=None, mask=None):
    sub = S if mask is None else S[mask]
    sub = sub[sub[f'G+{k}'].notna()]
    cols = [np.ones(len(sub)), sub.g_now] + [ (sub.ins_base==b).astype(float) for b in [2,3,4,6]]
    names = ['c','g','b2','b3','b4','b6']
    for nm in extra_cols:
        cols.append(sub[nm].values); names.append(nm)
    r,_ = ols(sub[f'G+{k}'], np.column_stack(cols), names)
    return r, len(sub)
S['shock2'] = S.shock**2
S['lnI'] = np.log(S.insulin)
out=[]
for b in [1,2,3,4]:
    for k in [1,2,3,4]:
        r,n = lp(S,k,['shock'], S.ins_base==b)
        out.append((b,k,n,r.loc['shock','b'],r.loc['shock','se'], S[S.ins_base==b].g_now.mean()))
print(pd.DataFrame(out,columns=['base','k','n','eff','se','gmean']).to_string())
for k in [1,2,3]:
    r,n = lp(S,k,['shock','shock2'])
    print(k, r.loc[['shock','shock2']].to_string())
```