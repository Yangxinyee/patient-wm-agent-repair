This settles the injection route. At bundles with an infusion, prelude head 2 sends about 0.99 of every slot's attention to the u token (token 5), and no head ever attends to the z token. At bundles without an infusion, head 2 instead attends to the bundle encoding e (0.93).

So head 2's value vector for u reaches each slot almost unweighted. The plan is to set the prelude output row for the memory dimension to a fixed linear readout over head 2's 48 value dimensions. That costs nothing beyond the skeleton, because that row is zeroed anyway.

Next I'll check how well the code-8 and code-9 start rates can be read linearly from head 2's value vector of u. I'll also check whether the readout stays proportional when the code-8 rates are doubled. A cross-validated R² above about 0.9 and a ratio near 2 would confirm the route.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict as cvp
Wkv=W('core.prelude.0.attn.kv.weight'); g1=W('core.prelude.0.norm1.weight'); b1=W('core.prelude.0.norm1.bias')
def tok_v(i,k,B,g):
    o_,h_=prev_state(i); dt=float(B[k].minute.iloc[0]-B[k-1].minute.iloc[0])*60
    e,u,_,_=enc_bundle(g); z=zpost_mean(o_,h_,dt,e,u)
    Xt=np.asarray(pre_tokens(h_,dt,e,u,z),dtype=np.float64)
    n=(Xt-Xt.mean(1,keepdims=True))/np.sqrt(Xt.var(1,keepdims=True)+1e-5)*g1+b1
    return (n@Wkv[192:].T)[:,96:144]
def rts(g): return g[(g.code==8)&(g.kind==0)].rate.sum(), g[(g.code==9)&(g.kind==0)].rate.sum()
def dbl(g,c=8,f=2.0):
    g=g.copy(); m=g.code==c; g.loc[m,'rate']*=f; g.loc[m,'dose']*=f; return g
Vu=[];Ve=[];R=[];A_=[];Vu2=[];R2=[]
for s in sids:
    s=int(s); B=stay_bundles(s); idx=np.where(D['stay_id']==s)[0]
    for k in range(1,len(B)):
        i=idx[k]; V=tok_v(i,k,B,B[k]); act=bool(D['has_insulin'][i] or D['has_norepinephrine'][i])
        Vu.append(V[5]); Ve.append(V[4]); R.append(rts(B[k])); A_.append(act)
        if act and rts(B[k])[0]>0:
            Vu2.append(tok_v(i,k,B,dbl(B[k]))[5]); R2.append(rts(B[k])[0])
Vu=np.array(Vu);Ve=np.array(Ve);R=np.array(R);A_=np.array(A_);Vu2=np.array(Vu2);R2=np.array(R2)
for c in [0,1]:
    m=Ridge(alpha=1e-2); p=cvp(m,Vu[A_],R[A_,c],cv=5)
    print('code',8+c,'cv R2 %.3f'%(1-((p-R[A_,c])**2).mean()/R[A_,c].var()))
m8=Ridge(alpha=1e-2).fit(Vu[A_],R[A_,0])
d1=m8.predict(Vu[A_][R[A_,0]>0]); d2=m8.predict(Vu2)
print('n',len(R2),'ratio of mean decoded x2/x1 %.2f'%(d2.mean()/d1.mean()), 'slope d2 on 2r %.2f'%np.polyfit(2*R2,d2,1)[0])
```