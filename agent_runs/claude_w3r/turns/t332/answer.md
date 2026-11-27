`panel.pkl` doesn't hold the frame; `B` was almost certainly the training record DataFrame (it is indexed by `stay_id`, and `rows` are row indices into it). I'll restore it from `/data/train.csv`.

I also no longer need to locate the action token: Δq from a dose change only enters through `out` and `h`, both returned by `step_at`, so I can compute Δq = Δout + Δ·out_norm(mean h) directly. I'll drive `du = g_i × rate` using rates read straight from the record rows.

Confirming outcome: a consistent Δq direction per unit rate whose whitened SNR is far above the Hmean reader's ~0.045.

```python
import numpy as np, pandas as pd
B=pd.read_csv('/data/train.csv')
print('B restored',B.shape, list(B.columns))
def zz_ln(x):
    x=np.asarray(x,dtype=np.float64); m=x.mean(); s=np.sqrt(((x-m)**2).mean()+1e-5)
    return np.asarray(ONw,dtype=np.float64)*(x-m)/s+np.asarray(ONb,dtype=np.float64)
def zz_dq(ev,gvec,nmax=400):
    zz_out=[]
    for zz_s in [int(x) for x in selS]:
        try: zz_B,zz_rws,zz_EU=stay_data(zz_s)
        except Exception as zz_e:
            print('stay_data fail',type(zz_e).__name__,zz_e); return None
        for zz_k in range(len(zz_EU)):
            zz_ri=int(zz_rws[zz_k])
            if zz_ri<0 or zz_ri>=len(B): continue
            zz_r=B.iloc[zz_ri]
            if zz_r['event']!=ev or not np.isfinite(zz_r.get('rate',np.nan)) or zz_r['rate']<=0: continue
            zz_rt=float(zz_r['rate'])
            try:
                zz_h0,zz_o0=step_at(zz_B,zz_rws,zz_EU,zz_k,None,None)
                zz_h1,zz_o1=step_at(zz_B,zz_rws,zz_EU,zz_k,np.asarray(gvec*zz_rt,dtype=np.float32),None)
            except Exception as zz_e:
                print('step_at fail',type(zz_e).__name__,zz_e); return None
            zz_H0=np.asarray(zz_h0.detach().cpu().numpy(),dtype=np.float64); zz_H1=np.asarray(zz_h1.detach().cpu().numpy(),dtype=np.float64)
            zz_O0=np.asarray(zz_o0.detach().cpu().numpy(),dtype=np.float64); zz_O1=np.asarray(zz_o1.detach().cpu().numpy(),dtype=np.float64)
            zz_out.append(((zz_O1-zz_O0)+(zz_ln(zz_H1.mean(0))-zz_ln(zz_H0.mean(0))))/zz_rt)
            if len(zz_out)>=nmax: return np.array(zz_out)
    return np.array(zz_out)
for zz_ev,zz_g,zz_Q,zz_nm in (('insulin',g_i,QG,'ins'),('norepinephrine',g_n,QM,'ne')):
    zz_D=zz_dq(zz_ev,zz_g)
    if zz_D is None or len(zz_D)==0: print(zz_nm,'NO DATA'); continue
    zz_d=zz_D.mean(0); zz_cos=float(np.mean(zz_D@(zz_d/np.linalg.norm(zz_d))/np.linalg.norm(zz_D,axis=1)))
    zz_C=np.cov(zz_Q.T)+1e-6*np.eye(192); zz_w=np.linalg.solve(zz_C,zz_d); zz_snr=float(np.sqrt(zz_d@zz_w))
    zz_w=zz_w/(zz_d@zz_w)
    print('%s n=%d |dq|/unit=%.4f cos=%.3f  SNR/unit=%.3f  baseline sd=%.4f'%(zz_nm,len(zz_D),np.linalg.norm(zz_d),zz_cos,zz_snr,(zz_Q@zz_w).std()))
    np.save('/work/dq_%s.npy'%zz_nm,zz_d); np.save('/work/w_%s.npy'%zz_nm,zz_w)
```