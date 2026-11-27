Run 200 shows the bundle timing suits a fast-decaying channel. Bundles fall only at minutes 0, 5 and 10. Of the readings, 1716 of 1865 glucose_finger and 2206 of 2481 map_arterial directly follow the s:05 action bundle (55 minutes earlier). So an injection there reaches nearly every relevant reading before it decays.

I'll make the injection in the pre-coda state c = c_out(h_mid). Two things could spoil it:
- the coda might turn a fast-dimension injection into persistent components;
- the fast dimensions might vary a lot naturally.

So next I'll capture c at an action bundle and check that my re-implementation of the coda reproduces the state. Then I'll measure:
- the natural variance of the fast dimensions;
- the minimum-variance direction within them, the norm of its h_mid pre-image (through c_out's inverse) and how much of the coda's response stays inside the fast subspace.

The plan stands if the coda leaks little (≲10%) outside the fast subspace and the pre-image is not huge.

```python
CAPc={}
def LNt_c2(x,name):
    y=_LNt0(x,name)
    if name=='core.coda.0.norm1': CAPc['c']=x.clone()
    return y
globals()['LNt']=LNt_c2
try: hk_,ok_=step_at(B_,rows_,EU_,k0)
finally: globals()['LNt']=_LNt0
c_=CAPc['c']
def coda(c):
    l1=_LNt0(c,'core.coda.0.norm1'); c1=c+_mha0(l1,l1,l1,'core.coda.0.attn')[0]
    return c1+_ffn0(_LNt0(c1,'core.coda.0.norm3'),'core.coda.0.ffn')
print('coda recon err', float((coda(c_)-hk_).abs().max()))
ym_=P_['ym'].astype(np.float64)
for nf in [10,20,30]:
    F_=np.argsort(tau)[:nf]; Cf=np.cov(ym_[:,F_].T); evf,Vf=np.linalg.eigh(Cf)
    a_y=np.zeros(192); a_y[F_]=Vf[:,0]
    pre=np.linalg.solve(Co,a_y)
    resp=[]
    for s_ in [1.0,5.0]:
        dy_=(coda(c_+torch.tensor(a_y*s_))-coda(c_)).mean(0).numpy()/s_
        inF=np.zeros(192,bool); inF[F_]=True
        resp.append((round(float(dy_@a_y),3), round(float(np.linalg.norm(dy_[~inF])),3)))
    print('nf',nf,'fast-dim natural sd range %.3f..%.2f'%(np.sqrt(evf[0]),np.sqrt(evf[-1])),'min-var dir sd %.4f'%np.sqrt(evf[0]),'| |C^-1 a_y| %.2f'%np.linalg.norm(pre),'| coda resp (along a_y, outside fast) per unit:',resp)
```