The feasibility test gave a mixed answer.

**What worked.** The ×2 edit pushes the state far outside its factual range along the fitted direction: the out-of-sample D² is 97. A readout along that direction at θ = 2 brought norepinephrine ×2 to +4.96 (truth +4.26) at a factual cost of only +0.0002. Insulin stayed roughly where it was.

**What did not work.**
- The ×½ edit has no projection on that direction (×½/×2 ratio −0.007), so ×½ stayed at −0.5 against a truth of −2.13. My explanation is that recorded norepinephrine rates never exceed 1.0. Doubling therefore produces input the model never saw in training, while halving produces ordinary-looking states. The direction detects "doubled", not dose.
- At θ = 4 the factual cost jumped to +0.27 and the insulin readout collapsed to −4.0. The signal must have overwhelmed the gate's margin, either leaking into glucose readings or pushing the GELU out of its linear range.

Changes in this run:
1. **Gate margin:** I size it from the largest signal actually seen across all readings, so the gate cannot be overrun.
2. **Second direction for ×½:** fitted to the ×½ shift and scaled so its ×½ response is −0.5 per unit. If the encoding is linear, its ×2 response should be about +1.
3. **Combined signal:** the two directions go into one gated unit pair, as θ₂·u₂ + θₕ·uₕ.
4. **Calibration:** on all 400 tuning stays, where the truth is +4.47 and −2.24. I measure (θ₂, θₕ) = (2, 0) and (0, 3), solve the linear 2×2 system, and evaluate at the solution.
5. **Save rule:** the edit is saved only if factual cost stays below 0.0015, insulin stays near −13.8/+6.7 offline, and both norepinephrine readouts move most of the way to the truth.

What would confirm or refute this:
- **Confirm:** the ×½ direction has a reasonable out-of-sample D² (at least about 3) and a ×2/×½ ratio near −2, and the solved setting lands near +4.5 and −2.2 cheaply.
- **Refute:** a tiny D² would mean any honest ×½ response costs too much factual fit. Then the ×½ readout stays mostly unrepaired.

```python
import time, numpy as np, torch, traceback as tb
T0=time.time()
def rdir(S_,d_,lam): return torch.linalg.solve(S_+lam*float(S_.diagonal().mean())*torch.eye(S_.shape[0],dtype=S_.dtype),d_)
print('|Dh| aff',round(float(DhE[aE].norm(dim=1).mean()),4))
bestH=None
for lam in [1e-3,1e-2,1e-1,1.0]:
    uh_=rdir(S,DhE[aE].mean(0),lam); Ehi=(DhE[aE]@uh_).mean(); Eho=(DhF[aF]@uh_).mean(); E2o=(D2F[aF]@uh_).mean(); V=(XF@uh_).var()
    d2h=float(Eho**2/V); print(f'lam {lam}: outD2h {d2h:.2f} inD2h {float(Ehi**2/(XE@uh_).var()):.2f} x2/half {float(E2o/Eho):.2f} in/out {float(Ehi/Eho):.2f}')
    if bestH is None or d2h>bestH[0]: bestH=(d2h,lam)
DhA=torch.cat([DhE[aE],DhF[aF]])
u2=rdir(SA,D2A.mean(0),1e-4); u2=u2/(D2A@u2).mean()
uh=rdir(SA,DhA.mean(0),bestH[1]); uh=-0.5*uh/(DhA@uh).mean()
print('u2 half-resp',round(float((DhA@u2).mean()),3),'sd',round(float((XA@u2).std()),3),'| uh x2-resp',round(float((D2A@uh).mean()),3),'sd',round(float((XA@uh).std()),3))
WCall_E=NE['E']['wcf'][:,:,keep].reshape(-1,len(keep)).double()
def build_ne2(P0,th2,thh):
    U=th2*u2+thh*uh; c=float((XA@U).mean()); smax=float((WCall_E@U-c).abs().max()); G=1.5*smax+10; kp=G/((lo-hi)/2)
    P={k:v.clone() for k,v in P0.items()}
    W0=P['attr_heads.value_bin.0.weight']; b0=P['attr_heads.value_bin.0.bias']; W2=P[key2]; b2=P['attr_heads.value_bin.2.bias']
    for k in (k3,k4): b2+=W2[:,k]*h1m[k]; W2[:,k]=0; W0[k,:]=0; b0[k]=0
    g=torch.zeros(192); g[keep]=gM.float(); uu=torch.zeros(192); uu[keep]=U.float()
    W0[k3]=kp*g+uu; b0[k3]=-kp*mid-c; W0[k4]=kp*g; b0[k4]=-kp*mid; W2[:,k3]=tilt; W2[:,k4]=-tilt
    return P,smax,G
ids=sorted(tu.stay_id.unique().tolist())
nF0=float(torch.as_tensor(NE['F']['rf'][0]).mean()); nE0=float(torch.as_tensor(NE['E']['rf'][0]).mean())
def evalcfg(th2,thh):
    P,smax,G=build_ne2(Pcur,th2,thh)
    cF=float(torch.as_tensor(tf_eval(P,NE['F']['d'],False)[0]).mean())-nF0; cE=float(torch.as_tensor(tf_eval(P,NE['E']['d'],False)[0]).mean())-nE0
    R=eval_shifts(P,ids,20,0); rr={k:round(float(v[0]),2) for k,v in R.items()}
    print(f'th2 {th2:.2f} thh {thh:.2f} smax {smax:.2f} G {G:.1f} costF {cF:.5f} costE {cE:.5f}',rr,'truth',{k:round(float(v[1]),2) for k,v in R.items()},round(time.time()-T0)); return P,rr,cF,cE
x0,h0=-0.61,-0.58
try:
    _,rA,_,_=evalcfg(2.0,0.0); _,rB,_,_=evalcfg(0.0,3.0)
    a2=(rA['norepinephrine_x2']-x0)/2; ah=(rA['norepinephrine_half']-h0)/2; b2=(rB['norepinephrine_x2']-x0)/3; bh=(rB['norepinephrine_half']-h0)/3
    M=np.array([[a2,b2],[ah,bh]]); sol=np.linalg.solve(M,np.array([4.47-x0,-2.24-h0])); print('slopes',np.round(M,3),'sol',np.round(sol,3))
    if np.all(np.isfinite(sol)) and np.all(np.abs(sol)<15):
        Pn,rC,cF,cE=evalcfg(float(sol[0]),float(sol[1]))
        ok=(cF<0.0015 and cE<0.0015 and abs(rC['insulin_x2']+13.8)<0.8 and abs(rC['insulin_half']-6.7)<0.6 and rC['norepinephrine_x2']>2.5 and rC['norepinephrine_half']<-1.2)
        print('ok',ok)
        if ok:
            for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias']: save_edit(n,Pn[n])
            NEsol=(float(sol[0]),float(sol[1])); print('SAVED NE readout; edited',len(edited()))
            with open('/work/notes.md','a') as f: f.write(f'- Run288-289 NE plan B: u2=x2 detector (Sigma^-1 D2, lam1e-4, outD2 97, no half response: x2 doses >1.0 off-manifold), uh=half dir (lam {bestH[1]}, outD2h {bestH[0]:.1f}); units k3={k3},k4={k4} gated LDA map_arterial, margin from smax; tilt (centre-74)/vbar. th=(2,0)->{rA}; (0,3)->{rB}; SOLUTION th2={sol[0]:.3f} thh={sol[1]:.3f} -> {rC} costF {cF:.5f} costE {cE:.5f}. SAVED (value_bin 4 tensors).\n')
except Exception: print('err',tb.format_exc()[-900:])
print('sec',round(time.time()-T0))
```