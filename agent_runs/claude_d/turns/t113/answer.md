Run 101 failed only because `load()` returns a torch tensor, not a numpy array. I'll add a small wrapper that converts weights to float64 numpy and rerun the same set-up. The set-up pairs each factual reading with the state of the bundle before it and the elapsed time. Its output will be the time-gap range.

```python
Wn=lambda n: load(n).double().numpy()
def LNf(x,p,eps=1e-5):
    m=x.mean(-1,keepdims=True); v=x.var(-1,keepdims=True); return (x-m)/np.sqrt(v+eps)*Wn(p+'.weight')+Wn(p+'.bias')
def Lin(x,p,b=True): y=x@Wn(p+'.weight').T; return y+Wn(p+'.bias') if b else y
fa=np.exp(Wn('core.flow.a'))
smin=np.round(hr3.astype(np.float64)*60).astype(int)
rows_by={}
for i,(s,m) in enumerate(zip(sid3,smin)): rows_by.setdefault(s,[]).append((m,i))
f4['min']=np.round(f4.hour.astype(np.float64)*60).astype(int)
prev=[];dtm=[]
for s,m in zip(f4.stay_id,f4['min']):
    c=[(mm,i) for mm,i in rows_by[s] if mm<m]; mm,i=c[-1]; prev.append(i); dtm.append(m-mm)
f4['prev']=prev; f4['dt_s']=np.array(dtm)*60.0
print(f4.dt_s.describe()[['min','50%','max']].to_dict())
```