Run 278 confirmed the encoder reimplementation and the empty-bundle constant, and the per-drug ridge maps fit essentially perfectly (held-out R² 1.0000 and 0.99996). In the tuning data the drugs never appear with rate 0, so real bundles are single-drug or both-drug with positive rates; the reference rates are the means, 1.70 and 0.49.

The amplification itself is badly behaved. Before the LayerNorm, the encoder output is huge (norm about 1200). Much of the rate-induced change only rescales that vector, which the norm then removes. So amplifying before the norm gives erratic results: some cases barely move, others jump by 12 in norm.

The fix is to amplify the deviation in normalised coordinates, scaled back by each bundle's own spread, so that it survives the LayerNorm. This should give u' ≈ u + (k−1)(u(r) − u(r_ref)), measured in the space the model actually reads.

This run will:
- refit the two maps on that target;
- check them on the real bundles of the tuning data (factual distortion, and how the x2 and ½ dose differences in u are amplified), compared with the natural spread of u;
- save k_insulin = 5 and k_norepinephrine = 8 as the first calibration point;
- log progress in the notebook.

```python
import numpy as np, torch
def nrm(s):
    mu=s.mean(-1,keepdims=True); sd=np.sqrt(((s-mu)**2).mean(-1,keepdims=True)+1e-5); return (s-mu)/sd, sd
n2,sd2=nrm(s2); nIref,_=nrm(sIref); nNref,_=nrm(sNref)
n2r=n2.reshape(len(rI),len(rN),192); sd2r=sd2.reshape(len(rI),len(rN),1)
DI2=(sd2r*(n2r-nIref[None])).reshape(-1,192); DN2=(sd2r*(n2r-nNref[:,None])).reshape(-1,192)
n1I,sd1I=nrm(s1I); n1Ir,_=nrm(s1Ir); n1N,sd1N=nrm(s1N); n1Nr,_=nrm(s1Nr)
YI2=np.vstack([DI2,sd1I*(n1I-n1Ir),np.zeros_like(s1N),np.zeros((1,192))]); YN2=np.vstack([DN2,np.zeros_like(s1I),sd1N*(n1N-n1Nr),np.zeros((1,192))])
for nm,Y in [('I',YI2),('N',YN2)]:
    B=ridge(X[~te],Y[~te],wts[~te],1e-3); P=X[te]@B
    print(nm,'heldout R2 %.5f'%(1-((P-Y[te])**2).sum()/((Y[te]-Y[te].mean(0))**2).sum()))
BI2=ridge(X,YI2,wts,1e-3); BN2=ridge(X,YN2,wts,1e-3)
def set_amp2(kI,kN,save=True):
    Wn=W2b+(kI-1)*BI2[:768].T+(kN-1)*BN2[:768].T; bn=b2b+(kI-1)*BI2[768]+(kN-1)*BN2[768]
    if save:
        save_edit('encode_action.mlp2.2.weight',torch.tensor(Wn,dtype=torch.float32)); save_edit('encode_action.mlp2.2.bias',torch.tensor(bn,dtype=torch.float32))
    return Wn,bn
cache={}
def h1(code,r):
    key=(code,round(float(r),5))
    if key not in cache: cache[key]=mlp1(np.asarray(datom(code,r),float).ravel()[None])[0]
    return cache[key]
def enc_b(bl,Wl,bl_):
    out=[]
    for ri,rn in bl:
        hs=([h1(8,ri)] if ri is not None else [])+([h1(9,rn)] if rn is not None else [])
        n=len(hs); m=np.sum(hs,0)/np.sqrt(n)+sw*np.log1p(n)+sbz; hid=gelu(m@W2a.T+b2a); out.append(LNv((hid@Wl.T+bl_)[None])[0])
    return np.array(out)
dr=tun[tun.event.isin(['insulin','norepinephrine'])]
pv=dr.pivot_table(index=['stay_id','hour'],columns='event',values='rate',aggfunc='first')
bund=[(None if np.isnan(a) else a, None if np.isnan(b) else b) for a,b in zip(pv.get('insulin',np.nan*np.ones(len(pv))),pv.get('norepinephrine',np.nan*np.ones(len(pv))))]
rng2=np.random.default_rng(1); idx=rng2.choice(len(bund),min(800,len(bund)),replace=False); bs=[bund[i] for i in idx]
print('bundles',len(bund),'insulin-only',sum(1 for a,b in bund if b is None),'ne-only',sum(1 for a,b in bund if a is None),'both',sum(1 for a,b in bund if a is not None and b is not None))
kI,kN=5.0,8.0; Wn,bn=set_amp2(kI,kN)
U0=enc_b(bs,W2b,b2b); U1=enc_b(bs,Wn,bn); print('natural spread of u (rms dev from mean) %.3f'%np.sqrt(((U0-U0.mean(0))**2).sum(1)).mean(), '| factual distortion |u1-u0| mean %.3f p90 %.3f'%(np.linalg.norm(U1-U0,axis=1).mean(),np.quantile(np.linalg.norm(U1-U0,axis=1),.9)))
for drug,mul in [('I',2.0),('I',0.5),('N',2.0),('N',0.5)]:
    sel=[(a,b) for a,b in bs if (a if drug=='I' else b) is not None]
    ed=[((a*mul if drug=='I' else a),(b*mul if drug=='N' and b is not None else b)) for a,b in sel]
    d0=enc_b(ed,W2b,b2b)-enc_b(sel,W2b,b2b); d1=enc_b(ed,Wn,bn)-enc_b(sel,Wn,bn)
    cos=(d0*d1).sum(1)/(np.linalg.norm(d0,axis=1)*np.linalg.norm(d1,axis=1)+1e-12)
    print(drug,mul,'|du| orig %.3f amp %.3f ratio %.2f cos %.3f'%(np.linalg.norm(d0,axis=1).mean(),np.linalg.norm(d1,axis=1).mean(),np.linalg.norm(d1,axis=1).mean()/np.linalg.norm(d0,axis=1).mean(),cos.mean()))
print('edited:',edited())
with open('/work/notes.md','a') as f:
    f.write('\n## Restart 6 (runs 272-279)\n- fwd9 factual both (writer v2): no change; fwd10 states both: z write ~.03, h diff .03 vs natural 10.6 -> z-token route dead (prelude LN). fwd11 dose_edit both tuning: writer v2 ~no effect (ins x2 -2.07 vs -2.15).\n- a_bar .29-.43 all dims: no slow linear memory. DROPPED all writer edits (run 278).\n- NEW APPROACH: amplify the drug-rate-induced deviation of the action encoder output u (model own dose pathway) per drug: encode_action.mlp2.2 += (k-1)*ridge map hid->deviation; deviation in LN-normalised coords (x sd) relative to ref rates (tuning means ins 1.70, ne 0.49); empty bundle pinned 0. Fit R2~1. set_amp2(kI,kN) in kernel. T must stay the tensor dict (helpers use it); TT = converter.\n- tuning.csv: drug rows only when running (no rate 0); ins rate median 1.5 p95 3.5; ne median .45 p95 1.0.\n- run279: saved kI=5,kN=8 -> next: dose_edit both + factual both to calibrate.\n')
```