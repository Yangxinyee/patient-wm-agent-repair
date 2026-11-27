The ids I sent are not exactly sel1, so the factual output and the saved states cover different stay sets. The factual file is still useful for measuring the model's own dose sensitivity. For checking the head copy, I can use the stays that appear in both sets.

Next I will write the head computation from the doc:
- the view built from the saved state: `out`, the time-decayed slots, the gap term and the prior z;
- the query, the register of last readings, the value MLP and the transition table.

I will search the unknowns (`type_emb` index, gap scaling, layout of the 8 `last_numeric` inputs, `last_status`) by how well predicted means match the tool's.

A configuration with mean absolute error well under 1 mg/dL means the head copy is right, and I can then design and score the edit offline. Errors of several units mean something structural is still wrong.

```python
from scipy.special import erf
Wt={n:load_original(n).numpy().astype(np.float64) if hasattr(load_original(n),'numpy') else np.asarray(load_original(n),np.float64) for n in NAMES}
ov=sorted(set(sel1)&set(range(2020000000,2020000100))); print("overlap stays:",len(ov))
def lnorm(x,w,b,eps=1e-5):
    mu=x.mean(-1,keepdims=True); va=x.var(-1,keepdims=True); return (x-mu)/np.sqrt(va+eps)*w+b
def gelu(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def L(x,n): 
    y=x@Wt[n+'.weight'].T; return y+Wt[n+'.bias'] if n+'.bias' in Wt else y
NUM={(d['code'],d['attribute']):d for d in VOCAB['numeric']}
CEN={k:np.array(v,float) for k,v in VOCAB['value_bin_centres'].items()}
def fbin(code,x,strict=True):
    e=np.array(NUM[(code,'value')]['fine_bin_edges']); return 1+((e[None,:]<x[:,None]) if strict else (e[None,:]<=x[:,None])).sum(1)
def zval(code,x): d=NUM[(code,'value')]; return np.clip((x-d['median'])/d['iqr'],-10,10)
# reading table on overlap stays
F=FA[FA.stay_id.isin(ov)].copy().sort_values(['stay_id','hour']).reset_index(drop=True)
F['prev_val']=F.groupby(['stay_id','event']).observed.shift(1); F['prev_hr']=F.groupby(['stay_id','event']).hour.shift(1)
sidx={}; 
for i,(s_,h_) in enumerate(zip(S2['stay_id'],S2['hour'])): sidx.setdefault(s_,[]).append((h_,i))
kk=[]; dts=[]
for s_,h_ in zip(F.stay_id,F.hour):
    lst=[(hh,i) for hh,i in sidx[s_] if hh<h_-1e-6]; hh,i=lst[-1]; kk.append(i); dts.append((h_-hh)*3600)
F['k']=kk; F['dt']=dts; print(F.shape, F.dt.describe()[['min','50%','max']].round(0).to_dict())
```