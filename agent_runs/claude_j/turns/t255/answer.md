Run 227 confirms that the saved fwd_3 states, 5,116 bundles, line up one-to-one with the replica's bundles. The reading tables also give each reading's bundle index and the model's predicted mean. So I can regress the z posterior (the last 32 of the 992 numbers) on the glucose_finger surprise at the bundles that contain that reading.

Next I will fit that regression. It gives a direction g in z-space meaning "glucose lower or higher than expected", plus how well z encodes the surprise. I will also check what `zpost` returns, so I can patch it next. A high R² supports the z lever. A low one means z carries little glucose information and refutes it.

```python
xs=Z3['x']; sid=Z3['stay_id']; off={}; 
for i,s in enumerate(sid):
    off.setdefault(int(s),i)
Zr=[];Sr=[]
for s in stays100:
    rt=RTs[s]; g_=rt[rt.event=='glucose_finger']
    for b_,ob,pm in zip(g_.b.values,g_.observed.values,g_.predicted_mean.values):
        Zr.append(xs[off[s]+int(b_),960:992]); Sr.append(ob-pm)
Zr=np.array(Zr); Sr=np.array(Sr)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
print(len(Sr),'sd surprise %.1f'%Sr.std(),'R2 surprise<-z (cv) %.3f'%cross_val_score(RidgeCV(alphas=[0.1,1,10]),Zr,Sr,cv=5).mean())
gz=np.linalg.lstsq(np.c_[np.ones(len(Sr)),Sr],Zr-Zr.mean(0),rcond=None)[0][1]; print('|g| per mg/dL %.4f, z sd mean %.3f'%(np.linalg.norm(gz),Zr.std(0).mean()))
k=11; e_,u_=vecs(s0,k); o_=torch.tensor(xs[off[s0]+k-1,:192]); h_=torch.tensor(xs[off[s0]+k-1,192:960]).reshape(4,192)
zz=zpost(o_,h_,60.0*(BUN[s0][k][0]-BUN[s0][k-1][0]),e_,u_); print(type(zz).__name__, [tuple(t.shape) for t in (zz if isinstance(zz,(tuple,list)) else [zz])])
```