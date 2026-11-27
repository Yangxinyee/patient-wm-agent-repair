Run 231 gives what the writer needs. The rebuild at reading bundles is approximate (z error 0.15), but good enough to rank units. Units 12, 88 and 105 never exceed a pre-activation of about −4.5 on any bundle type I checked, so their GELU output is essentially zero. Repurposing them will not change normal behaviour.

**Next run: fit three linear read-outs on the START-atom action encoding u**, whose rebuild is exact:
- a gate that equals 1 on any drug bundle;
- the insulin rate;
- the norepinephrine rate.

All three must read exactly 0 on the empty encoding. I'll fit them over a grid of rates, including the scaled rates the dose edits produce, and report how accurate they are.

This is confirmed if rate errors are small (well below the innovation spreads, 0.66 and 0.23) across realistic rates and the gate stays near 1. Large errors would mean the encoding is too nonlinear for a single linear read-out, and I'd need a piecewise or multi-unit version.

```python
gi_=[0,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5,6,7,8,10,12,15,18,24,30]; gn_=[0,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,1.5,2.0,2.5,3.0]
rowsU=[];tg=[];ta=[];tb=[];wt=[]
def uenc(atoms): return np.asarray(setenc_my(atoms,'encode_action'),dtype=np.float64).ravel()
for a in gi_:
    for b in gn_:
        rowsU.append(uenc([datom(8,a,0),datom(9,b,0)])); tg.append(1);ta.append(a);tb.append(b); wt.append(1.0 if (a<=12 and b<=1.5) else 0.3)
for a in gi_: rowsU.append(uenc([datom(8,a,0)])); tg.append(1);ta.append(a);tb.append(0); wt.append(1.0 if a<=12 else 0.3)
for b in gn_: rowsU.append(uenc([datom(9,b,0)])); tg.append(1);ta.append(0);tb.append(b); wt.append(1.0 if b<=1.5 else 0.3)
UG=np.stack(rowsU); UE=np.asarray(U_EMPTY,dtype=np.float64).ravel()
Xu=np.vstack([UG,UE[None,:]]); wv=np.sqrt(np.array(wt+[1e4]))
PD={}
for nmv,t_ in [('g',tg),('A',ta),('B',tb)]:
    y_=np.array(t_+[0.0],dtype=np.float64)
    for lam_ in [1e-4,1e-3,1e-2]:
        A_=np.vstack([Xu*wv[:,None],np.sqrt(lam_)*np.eye(192)]); yy_=np.concatenate([y_*wv,np.zeros(192)])
        p_=np.linalg.lstsq(A_,yy_,rcond=None)[0]; e_=UG@p_-y_[:-1]; real=np.array(wt)==1.0
        print(nmv,'lam',lam_,'|p| %.0f  err real max %.3f rms %.3f  ext max %.3f  empty %.2e'%(np.linalg.norm(p_),np.abs(e_[real]).max(),np.sqrt((e_[real]**2).mean()),np.abs(e_[~real]).max(),UE@p_))
        PD[(nmv,lam_)]=p_
```