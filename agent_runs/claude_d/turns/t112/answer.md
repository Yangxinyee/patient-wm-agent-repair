All 8 `last_numeric` columns and all 15 `type_emb` rows carry weight, so none is dead. The 15 types plausibly index (event kind × atom kind), which would make a glucose reading type 3 or 5. Next I'll assemble a first head re-implementation from the recorded states: view, LayerNorm, registers, MLP, softmax mean. I'll compare it with fwd_4's predicted means over a small grid of the undocumented choices (type index, gap input scaling, register layout), using the prior mean for z. The variant with the lowest error points to the right structure. An error near 0 would confirm it, and a uniformly large error would mean a structural piece is missing.

```python
from scipy.special import erf
gelu=lambda x:0.5*x*(1+erf(x/np.sqrt(2)))
def LNf(x,p,eps=1e-5):
    m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+eps)*load(p+'.weight')+load(p+'.bias')
def Lin(x,p,b=True): y=x@load(p+'.weight').T; return y+load(p+'.bias') if b else y
fa=np.exp(load('core.flow.a').astype(np.float64))
# index states by stay and minute
smin=np.round(hr3.astype(np.float64)*60).astype(int)
rows_by={}
for i,(s,m) in enumerate(zip(sid3,smin)): rows_by.setdefault(s,[]).append((m,i))
f4['min']=np.round(f4.hour.astype(np.float64)*60).astype(int)
prev=[];dtm=[]
for s,m in zip(f4.stay_id,f4['min']):
    L_=rows_by[s]; c=[(mm,i) for mm,i in L_ if mm<m]; mm,i=c[-1]; prev.append(i); dtm.append(m-mm)
f4['prev']=prev; f4['dt_s']=np.array(dtm)*60.0
print(f4.dt_s.describe()[['min','50%','max']].to_dict())
```