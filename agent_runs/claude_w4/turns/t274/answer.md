Both pieces are now in place:

- **Head replica.** It reproduces the model's predicted means for all four readings with query type 1. RMSE is 1.8 mg/dL finger, 1.3 lab, 0.7 mmHg arterial and 1.1 cuff, against prediction SDs of about 29–31 and 10.
- **Step replica.** `step_cv(..., 'ageKV')` returns only the new slots y (2319×4×192). I can rebuild `out` myself as out_norm of the slot mean.

Next I'll score that replica against the exact states. If it is good, I'll compute the full‑chain injection Jacobian. That means 192 unit injections into the prelude slot residual at all drug bundles, each passed through `step_cv` and then the head replica, giving the change in predicted finger, lab, arterial and cuff values at the next reading.

From the average response vectors I'll compute two things:
- the best "insulin" direction, which moves glucose while MAP stays fixed;
- the best "NE" direction, the reverse;
- for each, the predicted shift per unit injection norm.

That says how large the injection must be. If the gains are decent (≳0.05 per unit norm) with consistent sign across readings, a targeted carrier is feasible.

```python
y0=A(rb).reshape(2319,4,192)
def outf(y): return LNn(y.mean(1),HW['core.out_norm.weight'],HW['core.out_norm.bias'])
o0=outf(y0); print('ageKV R2 y %.5f out %.5f relerr y %.4f out %.4f'%(r2f(y0,yt_),r2f(o0,ot_),np.linalg.norm(y0-yt_)/np.linalg.norm(yt_),np.linalg.norm(o0-ot_)/np.linalg.norm(ot_)))
TOKa=A(TOK); DO=np.zeros((192,2319,192),np.float32); DYm=np.zeros((192,2319,192),np.float32); t0=time.time()
for j in range(192):
    dX=np.zeros(192); dX[j]=1.0
    Ej,_=prelude_np(TOKa,dX=dX)
    yj=A(step_cv(np.arange(2319),torch.tensor(Ej,dtype=torch.float32),Ht,'ageKV')).reshape(2319,4,192)
    DYm[j]=(yj-y0).mean(1); DO[j]=outf(yj)-o0
print('jac time %.0fs'%(time.time()-t0))
S={}
for ev in EVS:
    d=HD[ev]; base=head_at(d['rest'],d['trans'],d['cen'],1,d['OUT'],d['HM'],d['dt']); kk=d['kk']
    S[ev]=np.stack([head_at(d['rest'],d['trans'],d['cen'],1,d['OUT']+DO[j][kk],d['HM']+DYm[j][kk],d['dt'])-base for j in range(192)],1)
sb={ev:S[ev].mean(0) for ev in EVS}
def proj_out(v,others):
    Q,_=np.linalg.qr(np.stack(others,1)); return v-Q@(Q.T@v)
tI_=proj_out(-sb['glucose_finger'],[sb['map_arterial'],sb['map_cuff']]); tI_/=np.linalg.norm(tI_)
tN_=proj_out(sb['map_arterial'],[sb['glucose_finger'],sb['glucose_lab']]); tN_/=np.linalg.norm(tN_)
for nm,t in [('tI',tI_),('tN',tN_)]:
    print(nm,{ev:'%.4f'%(S[ev]@t).mean() for ev in EVS},' frac consistent sign F %.2f A %.2f'%(((S['glucose_finger']@t)<0).mean(),((S['map_arterial']@t)>0).mean()))
print('|sb| per unit norm',{ev:'%.4f'%np.linalg.norm(sb[ev]) for ev in EVS},' cos F-L %.2f A-C %.2f F-A %.2f'%(cs(sb['glucose_finger'],sb['glucose_lab']),cs(sb['map_arterial'],sb['map_cuff']),cs(sb['glucose_finger'],sb['map_arterial'])))
```