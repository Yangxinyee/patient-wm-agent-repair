`step_at` returns `(h[4,192], out[192])` — exactly what `wc_row` needs, and it accepts a delta `du` on the action token. `u_of` is broken, but I have `UX` (9918 u-vectors) with known rates `YXi/YXn`, so I can fit the u-direction per unit rate and use that as `du`.

Next I will measure the quantity that actually matters: how much the value-head input `q` moves at a glucose row when the preceding insulin action is doubled. Then I compute the whitened reader `w ∝ Σ⁻¹Δq`, whose SNR tells me the largest tilt the factual budget allows.

Confirming outcome: a reader with intervention-SNR well above the Hmean reader's 1/23, ideally ≳1 per rate unit.

```python
import numpy as np
from sklearn.linear_model import Ridge
A=np.c_[np.ones(len(UX)), YXi, YXn, (YXi>0).astype(float), (YXn>0).astype(float)]
co,_,_,_=np.linalg.lstsq(A,UX,rcond=None); g_i=co[1]; g_n=co[2]
pred=A@co; print('u fit resid rms %.4f  |g_i| %.3f |g_n| %.3f'%(np.sqrt(((UX-pred)**2).mean()),np.linalg.norm(g_i),np.linalg.norm(g_n)))
pi=Ridge(alpha=1e-3).fit(UX,YXi); pn=Ridge(alpha=1e-3).fit(UX,YXn)
print('probe R2 ins %.4f ne %.4f'%(pi.score(UX,YXi),pn.score(UX,YXn)))
rows_g=GR[GR['after_act']==True] if 'after_act' in GR.columns else GR
print('gf rows after_act:',len(rows_g),'of',len(GR))
recs=[]
for s,sub in rows_g.groupby('stay'):
    try: B,rws,EU=stay_data(int(s))
    except Exception: continue
    for _,r in sub.iterrows():
        k=int(r['prow'])
        if k<0 or k>=len(EU): continue
        try:
            hsl0,o0=step_at(B,rws,EU,k,None,None)
            u=EU[k][3] if isinstance(EU[k],tuple) and len(EU[k])>3 else None
            uv=np.asarray(u.detach().cpu().numpy() if hasattr(u,'detach') else u,dtype=np.float64).reshape(-1)
            if uv.shape[0]!=192: continue
            ri=float(pi.predict(uv.reshape(1,-1))[0])
            if ri<0.2: continue
            du=np.asarray(g_i*ri,dtype=np.float32)
            hsl1,o1=step_at(B,rws,EU,k,du,None)
            q0=np.asarray(wc_row(o0,hsl0,float(r['gap']),float(r['lv']),float(r['age_s'])),dtype=np.float64).reshape(-1)
            q1=np.asarray(wc_row(o1,hsl1,float(r['gap']),float(r['lv']),float(r['age_s'])),dtype=np.float64).reshape(-1)
            recs.append((ri,q0,q1))
        except Exception as e:
            if len(recs)==0: print('ERR',type(e).__name__,e); break
    if len(recs)>=250: break
print('n recs',len(recs))
if recs:
    RI=np.array([x[0] for x in recs]); Q0=np.array([x[1] for x in recs]); Q1=np.array([x[2] for x in recs])
    DQ_=(Q1-Q0)/RI[:,None]
    print('|dq|/unit mean %.4f  cos-consistency %.3f'%(np.linalg.norm(DQ_,axis=1).mean(),
          float(np.mean(DQ_@ (DQ_.mean(0)/np.linalg.norm(DQ_.mean(0))) / np.linalg.norm(DQ_,axis=1)))))
    dq=DQ_.mean(0); C=np.cov(QG.T)+1e-6*np.eye(192)
    w=np.linalg.solve(C,dq); snr=float(np.sqrt(dq@w)); w=w/ (dq@w) * 1.0
    print('whitened reader: SNR per unit rate = %.3f  (Hmean reader was ~0.045)'%snr)
    print('read gain on dq %.4f ; baseline sd on QG %.4f'%(dq@w, (QG@w).std()))
    DQI=dq; WI=w; np.save('/work/dqi.npy',dq); np.save('/work/wi.npy',w)
```