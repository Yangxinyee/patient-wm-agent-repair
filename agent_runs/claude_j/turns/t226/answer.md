The pre-restart part of run_cf's output is the faulty piece. Bundles before the restart index carry slot states about 3% (≈60 in norm) off the replica's own trajectory, and those break the early-hour heads. After the restart, the recomputed states agree with a full from-bundle-1 run to within 0.014, so a bad starting state is forgotten within a step. The counterfactual rows (hours after the edit start) are therefore fine; only the factual rows at hours ≤ 12 were bad.

Fix: recompute the factual chunk of each stay with run_cf from bundle 1, then rebuild M from these factual chunks plus the existing counterfactual chunks. If the replica's factual means then match the tool within about 2 mg/dL at all hours, with NLL near 1.51, I'll refit the readout on clean data.

```python
FAC=[]; t0=time.time()
for s in sorted(BUN.keys()):
    rt=RT[RT.stay_id==s].sort_values('b').reset_index(drop=True)
    eps=torch.tensor(np.random.default_rng(int(s)%1000003).standard_normal((len(rt),4,32)))
    hr=np.round(rt.minute.values/60).astype(int); SHIFT.clear()
    O,H,_=run_cf(s,{},BUN[s][1][0],Xs); q,wc,P,mn=heads_on_states(O,H,s,rt,eps); f_,lb,cid=last_feats(rt); lg=logits_draws(wc,lb,cid); mn=np.asarray(mn,float)
    md=pd.DataFrame({'stay':s,'h0':12,'drug':'none','m':1.0,'event':rt.event.values,'hour':hr,'S':0.0,'mean_cf':mn,'obs':rt.observed.values,'b':rt.b.values,'mean_f':mn})
    FAC.append((md,wc.mean(1).numpy(),lg.numpy()))
CH=FAC+[c for c in CFD if c[0].drug.iloc[0]!='none']
M=pd.concat([c[0] for c in CH],ignore_index=True); WC=np.concatenate([c[1] for c in CH]); LG=np.concatenate([c[2] for c in CH])
EVV=M.event.values; NONE=(M.drug.values=='none'); isG=M.event.isin(GLU).values; isM=M.event.isin(MAPC).values
BIN=np.array([fb_(c,x) for c,x in zip(EVV,M.obs.values)])
f=np.where(NONE)[0]; P0=pnew(f,np.zeros(len(f)),vG); n0=-np.log(P0[np.arange(len(f)),BIN[f]])
mm=pd.DataFrame({'stay_id':M.stay.values[f],'hour':M.hour.values[f],'event':EVV[f],'n0':n0,'mf':M.mean_f.values[f]}).merge(fc,on=['stay_id','hour','event'])
print('sec %.0f rows %d | nll rep %.4f tool %.4f | pm err h<=12 %.2f h>12 %.2f'%(time.time()-t0,len(M),mm.n0.mean(),mm.nll.mean(),(mm.mf-mm.predicted_mean).abs()[mm.hour<=12].mean(),(mm.mf-mm.predicted_mean).abs()[mm.hour>12].mean()))
for c in GLU+MAPC: x=mm[mm.event==c]; print(c,'n',len(x),'rep %.3f tool %.3f'%(x.n0.mean(),x.nll.mean()))
```