Run 180 collected the internals for all 5201 bundles in 5.7 s; 29% of bundles start insulin and 43% start norepinephrine. Next I'll check two things.

**1. Where head2 attends.** It should look at the u token at action bundles. What it does at no-action bundles (u = 0) decides whether an injection through it is zero there.

**2. Low-variance directions.** I'll compute the variance spectra of the pre-norm slot tokens x2 and of the out_norm input (mean y). This shows whether there are directions with very low natural variation.

The plan is confirmed if head2 sits on u (≥0.95) and the smallest-direction sds are far below σ≈60, for example under 1. It is refuted if the spectra are flat.

```python
patt=P_['patt']; act=(PD.has_ins_s.values+PD.has_ne_s.values)>0
for h_ in range(4):
    print('head',h_,'slot->tokens mean att (action):',np.round(patt[act][:,h_,:4,:].mean((0,1)),3),' (no action):',np.round(patt[~act][:,h_,:4,:].mean((0,1)),3))
def spec(X):
    Xc=X-X.mean(0); C=Xc.T@Xc/len(X); ev,evec=np.linalg.eigh(C); return np.sqrt(np.clip(ev,0,None)),evec
x2=P_['x2'].astype(np.float64)
print('x2 per-slot mean norm', np.linalg.norm(x2.mean(0),axis=1).round(1), 'slot sim', np.round([np.corrcoef(x2[:,0].ravel(),x2[:,j].ravel())[0,1] for j in range(4)],4))
sd_x2,V_x2=spec(x2[:,0]); print('x2 slot0 sd spectrum smallest 10', np.round(sd_x2[:10],3), 'largest 5', np.round(sd_x2[-5:],1))
sd_x2m,V_x2m=spec(x2.mean(1)); print('x2 slot-mean smallest 10', np.round(sd_x2m[:10],3))
ym=P_['ym'].astype(np.float64); sd_ym,V_ym=spec(ym); print('ym smallest 10', np.round(sd_ym[:10],3),'largest 5',np.round(sd_ym[-5:],1),'std within-vec', np.round(ym.std(1).mean(),1))
E_=P_['E'].astype(np.float64); sd_E,V_E=spec(E_.mean(1)); print('E slot-mean smallest 10', np.round(sd_E[:10],4),'largest 5',np.round(sd_E[-5:],2))
o_=P_['out'].astype(np.float64); sd_o,V_o=spec(o_); print('out smallest 10', np.round(sd_o[:10],4),'largest 5', np.round(sd_o[-5:],2))
```