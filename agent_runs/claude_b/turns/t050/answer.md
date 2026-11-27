The MAP readings are integers, and the titration thresholds are sharp but sit exactly on two integer values. At readings ≤64 the rate is stepped up (+0.13 log), at 65 the mean step is half of that, and at 66–74 it is held. Downward steps begin halfway at 75 and fully at 76 (−0.20).

So clinicians act on the unrounded measurement. A recorded 65 or 75 is a coin flip between two actions, decided by sub-integer information that is slightly related to true MAP. The change happens within one integer, so the rule uses the measured value, not true MAP.

My 2-mmHg cells, [64,66) and [74,76), each mix a full-step reading with a half-step reading. Within those cells, a positive residual marks the lower reading and so slightly lower later MAP. That biases the norepinephrine estimate negative at both thresholds.

Next, I rebuild the titration residual with:
- exact-integer MAP cells, split by arterial or cuff source;
- the two ambiguous readings 65 and 75 excluded;
- previous rate below 0.75, so the cap at 1.0 can't truncate steps.

Then I redo the exogeneity test, the impulse responses, persistence, deconvolution and the implied x2 shift.

What would settle it:
- If the −0.3 effect largely disappears, it was this artefact, and zero is right for norepinephrine.
- If a clearly significant negative effect remains, I'll treat it as possibly real.

```python
src=np.where(np.isnan(Ar),1,0)  # 0 art used, 1 cuff used
Mi=np.where(np.isnan(Mu),-1,np.clip(Mu,45,95)).astype(int)
prevN=Ne[:,:-1]; curN=Ne[:,1:]
good=(prevN>0)&(prevN<0.75)&(Mi[:,1:]>=0)&(Mi[:,1:]!=65)&(Mi[:,1:]!=75)
onG=good&(curN>0)
cellC=((Mi[:,1:]-44)*2+src[:,1:])*16+np.clip(np.floor(prevN/0.05),0,15).astype(int)
ncell=cellC.max()+2
ratio=np.where(onG,curN/np.maximum(prevN,1e-9),0)
rm=np.bincount(cellC[onG],weights=ratio[onG],minlength=ncell)/np.maximum(np.bincount(cellC[onG],minlength=ncell),1)
NAc=np.zeros_like(Ne); NAc[:,1:]=np.where(onG,prevN*(ratio-rm[np.clip(cellC,0,ncell-1)]),0)
print('NAc sd %.4f, n %d'%(NAc[:,1:][onG].std(), onG.sum()))
# exogeneity of NAc
ys=[];x1=[];f1=[];ss=[]
for s in range(2,25):
    si=s+1; ok=onG[:,si-1]&~np.isnan(Mu[:,si-1])
    ys.append(NAc[ok,si]); x1.append(np.stack([Mu[ok,si-1]/10, np.nan_to_num(Ar[ok,si]-Cf[ok,si]), np.isnan(Ar[ok,si]-Cf[ok,si]), Ne[ok,si-1], np.nan_to_num(Cf[ok,si-1])/10],1)); f1.append(cellC[ok,si-1]); ss.append(np.where(ok)[0])
b,V=fe_ols(np.concatenate(ys),np.concatenate(x1),np.concatenate(f1),[],np.concatenate(ss))
print('NAc exog: prevMAP/10 %+.4f(%.4f) art-cuff %+.5f(%.5f) prevprevRate %+.4f(%.4f)'%(b[0],np.sqrt(V[0,0]),b[1],np.sqrt(V[1,1]),b[3],np.sqrt(V[3,3])))
okNc=lambda si: onG[:,si-1]
cellfnC=lambda si: cellC[:,si-1]
bNc,sNc=impulse(Ar,NAc,okNc,cellfnC,ctrl_ne,label='NE NAc -> art MAP')
pNc,_=impulse(Ne,NAc,okNc,cellfnC,ctrl_ne,K=11,label='NE NAc -> later rate')
delta=np.zeros(13)
for k in range(1,13): delta[k]=bNc[k]-sum(delta[l]*pNc[k-l] for l in range(1,k))
print('delta:',np.round(delta[1:],3))
implied(delta,sNc,Ne,Ar,label='NE clean direct (deconv)')
```