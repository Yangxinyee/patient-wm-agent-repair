Run 287 saved the insulin edit at f = 1.16 and showed me how the data are laid out:
- **Model states:** there are 4 with_code vectors per reading, which must be the 4 samples of the model's latent z.
- **Reading codes:** each reading's code is in `rd.event`. map_arterial is 25% of all readings.
- **Vocabulary:** the map_arterial bins run from 35 to 98 mmHg.
- **Re-encoding:** `reenc` returns a fresh copy of the dataset, so calling it is safe.

Next is the feasibility test for the norepinephrine plan, plus a trial build if it passes.

The test fits one direction in the state at map_arterial readings. It must maximise the shift caused by doubling norepinephrine relative to how much that direction varies in factual data. I fit it on one set of 300 train stays and score it on the other 300. I leave out dims 50, 79 and 99, which the insulin repair uses.

Then I build the readout. It uses the two value-head units that cost least to repurpose, judged by a second-order estimate of the fit lost. The two units form a gated pair: their outputs cancel except at map_arterial readings, where they tilt the predicted MAP bins. The tilt is scaled so that one unit of signal shifts the predicted mean by about 1 mmHg. I try two strengths, θ = 2 and θ = 4.

For each strength, the run reports:
- the factual NLL change on the held-out 300 stays;
- the offline readouts on sel1, against the truth of +4.26 for ×2 and −2.13 for ×½.

What would confirm or refute the plan:
- **Works:** the out-of-sample D² is at least about 3–5, the ×½/×2 ratio is near −0.5, and some θ gives a ×2 readout of about +4 at a factual cost below about +0.002.
- **Fails:** a low D² or a high cost would mean the state cannot separate off-record norepinephrine doses from normal ones. The ×½ edit failing to mirror ×2 would also count against the plan.

```python
import time, numpy as np, torch, pandas as pd, traceback as tb
import torch.nn.functional as F_
T0=time.time()
EX=[50,79,99]; keep=torch.tensor([j for j in range(192) if j not in EX])
def blk(nm, code='map_arterial'):
    Z=NE[nm]; m=torch.as_tensor((Z['d']['rd'].event==code).values)
    X=Z['wcf'][m].double().mean(1); D2=(Z['wc2']-Z['wcf'])[m].double().mean(1); Dh=(Z['wch']-Z['wcf'])[m].double().mean(1)
    return X[:,keep], D2[:,keep], Dh[:,keep], m
XE,D2E,DhE,mE=blk('E'); XF,D2F,DhF,mF=blk('F')
aE=D2E.norm(dim=1)>1e-3; aF=D2F.norm(dim=1)>1e-3
print('MAPq E',len(XE),'aff',int(aE.sum()),'F',len(XF),int(aF.sum()),'|D2|aff',round(float(D2E[aE].norm(dim=1).mean()),4),'sd/dim',round(float(XE.std(0).mean()),3))
S=torch.cov(XE.T); db=D2E[aE].mean(0); sc=float(S.diagonal().mean()); best=None
for lam in [1e-4,1e-3,1e-2,1e-1,1.0]:
    u=torch.linalg.solve(S+lam*sc*torch.eye(S.shape[0],dtype=S.dtype),db)
    Ein=(D2E[aE]@u).mean(); Vin=(XE@u).var(); E2=(D2F[aF]@u).mean(); Eh=(DhF[aF]@u).mean(); V=(XF@u).var(); d2o=float(E2**2/V)
    print(f'lam {lam}: inD2 {float(Ein**2/Vin):.2f} outD2 {d2o:.2f} Eh/E2 {float(Eh/E2):.3f} Ein/Eout {float(Ein/E2):.2f}')
    if best is None or d2o>best[0]: best=(d2o,lam)
# final u on E+F with best lam, normalised so mean u.D2 over affected = 1
XA=torch.cat([XE,XF]); D2A=torch.cat([D2E[aE],D2F[aF]]); SA=torch.cov(XA.T)
uN=torch.linalg.solve(SA+best[1]*float(SA.diagonal().mean())*torch.eye(SA.shape[0],dtype=SA.dtype),D2A.mean(0)); uN=uN/(D2A@uN).mean()
cN=float((XA@uN).mean()); print('best',best,'c',round(cN,3),'sd(u.wc)',round(float((XA@uN).std()),3),'Eh',round(float((torch.cat([DhE[aE],DhF[aF]])@uN).mean()),3))
# unit ablation proxy
key2='attr_heads.value_bin.2.weight'; W0=Pcur['attr_heads.value_bin.0.weight']; b0=Pcur['attr_heads.value_bin.0.bias']; W2c=Pcur[key2]
wcA=NE['E']['wcf'].reshape(-1,192); pA=torch.as_tensor(NE['E']['rf'][1]).float().repeat_interleave(4,0)
h1=F_.gelu(wcA@W0.T+b0); h1m=h1.mean(0); h1c=h1-h1m
uc=0.5*(h1c**2*(pA@(W2c**2)-(pA@W2c)**2)).mean(0); uc[[107,109]]=1e9
order=torch.argsort(uc)[:10].tolist(); print('cheap units',[(k,round(float(uc[k]),6)) for k in order])
k3,k4=order[0],order[1]
# gate LDA map_arterial vs rest
def allq(nm):
    Z=NE[nm]; ev=Z['d']['rd'].event.values
    return Z['wcf'][:,:,keep].reshape(-1,len(keep)).double(), torch.as_tensor(np.repeat(ev=='map_arterial',4))
XaE,yE=allq('E'); XaF,yF=allq('F')
mu1=XaE[yE].mean(0); mu0=XaE[~yE].mean(0); Sw=torch.cov(torch.cat([XaE[yE]-mu1,XaE[~yE]-mu0]).T)
gM=torch.linalg.solve(Sw+1e-3*float(Sw.diagonal().mean())*torch.eye(len(keep),dtype=Sw.dtype),mu1-mu0)
pE_=XaE@gM; pF_=XaF@gM; lo=min(float(pE_[yE].min()),float(pF_[yF].min())); hi=max(float(pE_[~yE].max()),float(pF_[~yF].max()))
print('gate min map',round(lo,3),'max other',round(hi,3))
mid=(lo+hi)/2; kap=8.0/max((lo-hi)/2,1e-6)
cent=VOCAB['value_bin_centres']['map_arterial']; print('centres',len(cent),cent)
cv=torch.tensor(cent,dtype=torch.float32); cv=cv[-16:]
pM=torch.as_tensor(NE['E']['rf'][1]).float()[mE][:,1:]; pM=pM/pM.sum(1,keepdim=True)
vbar=float(((pM*cv**2).sum(1)-(pM*cv).sum(1)**2).mean()); print('MAP pred var',round(vbar,2))
tilt=torch.zeros(17); tilt[1:]=(cv-74.0)/vbar
def build_ne(P0,theta):
    P={k:v.clone() for k,v in P0.items()}
    W0=P['attr_heads.value_bin.0.weight']; b0=P['attr_heads.value_bin.0.bias']; W2=P[key2]; b2=P['attr_heads.value_bin.2.bias']
    for k in (k3,k4): b2+=W2[:,k]*h1m[k]; W2[:,k]=0; W0[k,:]=0; b0[k]=0
    g=torch.zeros(192); g[keep]=gM.float(); uu=torch.zeros(192); uu[keep]=uN.float()
    W0[k3]=kap*g+theta*uu; b0[k3]=-kap*mid-theta*cN; W0[k4]=kap*g; b0[k4]=-kap*mid
    W2[:,k3]=tilt; W2[:,k4]=-tilt; return P
try:
    nF0=float(torch.as_tensor(NE['F']['rf'][0]).mean()); nE0=float(torch.as_tensor(NE['E']['rf'][0]).mean())
    for th in [2.0,4.0]:
        Pn=build_ne(Pcur,th)
        cF=float(torch.as_tensor(tf_eval(Pn,NE['F']['d'],False)[0]).mean())-nF0; cE=float(torch.as_tensor(tf_eval(Pn,NE['E']['d'],False)[0]).mean())-nE0
        R=eval_shifts(Pn,sel['sel1'],20,0)
        print('theta',th,'cost F',round(cF,5),'E',round(cE,5),{k:(round(float(v[0]),2),round(float(v[1]),2)) for k,v in R.items()},round(time.time()-T0))
except Exception: print('build err',tb.format_exc()[-900:])
print('sec',round(time.time()-T0))
```