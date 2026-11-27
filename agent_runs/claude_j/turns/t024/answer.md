The notebook is saved. Now I need the original model's answers to compare with the truth. A dose_edit pass on tuning stays gives the model's predicted shifts for all four edits at once. I want stays where the edited drug actually runs after hour 12, ideally both drugs.

Next I will pick 100 tuning stays: those with both insulin and norepinephrine from hour 12 on first, then those with either. I will also compute, with my estimated kernels, the true expected shift for each stay. Then the comparison is immediate once the forward pass comes back. I am not testing a hypothesis here, only preparing a well-chosen probe.

```python
PU = PU.sort_values(['stay_id','h']).reset_index(drop=True)
PU['I0']=PU.I.fillna(0.0); PU['N0']=PU.N.fillna(0.0)
info = PU[PU.h>=12].groupby('stay_id').agg(nI=('I', lambda x: x.notna().sum()), nN=('N', lambda x: x.notna().sum()))
print(info.describe())
both = info[(info.nI>0)&(info.nN>0)].index.tolist()
onlyI = info[(info.nI>0)&(info.nN==0)].index.tolist()
onlyN = info[(info.nI==0)&(info.nN>0)].index.tolist()
print(len(both), len(onlyI), len(onlyN))
KI = np.array([-2*k*0.6**(k-1) for k in range(1,25)])
KN = np.array([3.3*k*0.39**(k-1) for k in range(1,25)])
def true_shift(Pdf, stay, drug, mult, start=12, K=None):
    s = Pdf[Pdf.stay_id==stay].set_index('h')
    dose = s[drug+'0'] if drug+'0' in s else s[drug].fillna(0)
    hours = [t for t in range(start,25) if dose.get(t,0)>0]
    if not hours: return None
    first = hours[0]
    res = {}
    for sh in range(start+1, 25):
        if sh <= first: continue
        res[sh] = sum(K[sh-t-1]*dose.get(t,0)*(mult-1) for t in range(start, sh))
    return res
ex = true_shift(PU, both[0], 'I', 2.0, K=KI); print(both[0], ex)
```