The value head is Linear(192→192), GELU, Linear(192→17), plus a logit bias for each (code, last bin) from embed_transition.value_bin, with no normalisation before it. A decoder on the head input can therefore feed a hidden unit directly. The DEC entries are sklearn Ridge models. Next I will check four things: the shapes of their coefficients, the vocab code ids of the four reading types, whether some hidden units are nearly dead and can be reused, and how cleanly a linear gate can separate glucose rows from MAP rows.

```python
import numpy as np
for k,r in DEC.items():
    print(k,'coef',np.shape(r.coef_),'icpt',np.round(np.atleast_1d(r.intercept_),3),'alpha',getattr(r,'alpha',None))
cv=VOCAB.get('codes'); print(type(cv).__name__, (list(cv.items())[:16] if isinstance(cv,dict) else cv[:16]))
P=Xf@W1.T+b1
mx=P.max(0); q99=np.percentile(P,99.5,axis=0)
print('units max<-3:',int((mx<-3).sum()),'q99.5<-3:',int((q99<-3).sum()))
G=0.5*P*(1+__import__('scipy').special.erf(P/np.sqrt(2)))
contrib=G.std(0)*np.linalg.norm(W2,axis=0)
o=np.argsort(contrib); print('lowest contrib units',o[:6],np.round(contrib[o[:6]],4),'median',np.round(np.median(contrib),3))
y=np.where(np.isin(cf,[0,1]),1.0,-1.0); A=np.c_[Xf,np.ones(len(Xf))]
gw=np.linalg.lstsq(A,y,rcond=None)[0]; s=A@gw
print('gate glu min',np.round(s[y>0].min(),3),'map max',np.round(s[y<0].max(),3))
```