Recomputing from bundle 1 fixes it: the replica now matches the tool within about 2 mg/dL at every hour. The stored tool states for bundles 1–25 are the faulty input, because they don't reproduce the tool's own head outputs. So the factual rows and the counterfactual starting states in my dataset were partly wrong. Starting from bundle 0 fails only because run_cf needs a previous state.

Fix: build replica factual states from bundle 1 for all 100 stays, put them into the global state store that run_cf reads (`S`, saving the original), and regenerate the whole counterfactual dataset. Then I re-check the factual rows against the tool.
- If the replica-vs-tool error is about 1.5–2 at all hours and the replica NLL is close to 1.51, the data are sound and I can refit the readout.
- If not, I'll start every variant from bundle 1.

```python
if 'S_orig' not in globals(): S_orig=S
S=S_orig; xs=[]; ss=[]
for s in sorted(BUN.keys()):
    O,H,k0=run_cf(s,{},BUN[s][1][0],Xs); n=O.shape[0]
    xs.append(np.concatenate([O.numpy(),H.numpy().reshape(n,768),np.zeros((n,32))],1)); ss.append(np.full(n,s))
S={'x':np.concatenate(xs),'stay_id':np.concatenate(ss)}
print('S_rep',S['x'].shape)
CFD=[]; t0=time.time()
for s in sorted(BUN.keys()): CFD+=cf_stay(s)
print('regen sec %.0f chunks %d'%(time.time()-t0,len(CFD)))
M=pd.concat([c[0] for c in CFD],ignore_index=True); WC=np.concatenate([c[1] for c in CFD]); LG=np.concatenate([c[2] for c in CFD])
EVV=M.event.values; NONE=(M.drug.values=='none'); isG=M.event.isin(GLU).values; isM=M.event.isin(MAPC).values
BIN=np.array([fb_(c,x) for c,x in zip(EVV,M.obs.values)])
f=np.where(NONE)[0]; P0=pnew(f,np.zeros(len(f)),vG); n0=-np.log(P0[np.arange(len(f)),BIN[f]])
mm=pd.DataFrame({'stay_id':M.stay.values[f],'hour':M.hour.values[f],'event':EVV[f],'n0':n0,'mf':M.mean_f.values[f]}).merge(fc,on=['stay_id','hour','event'])
print('merged',len(mm),'nll rep %.4f tool %.4f | pm err h<=12 %.2f h>12 %.2f'%(mm.n0.mean(),mm.nll.mean(),(mm.mf-mm.predicted_mean).abs()[mm.hour<=12].mean(),(mm.mf-mm.predicted_mean).abs()[mm.hour>12].mean()))
g=M[(M.drug=='insulin')&(M.h0==12)&(M.event=='glucose_finger')]
print('ins x2 S %.2f model %.2f | x0 S %.2f model %.2f'%(g[g.m==2].S.mean(),(g[g.m==2].mean_cf-g[g.m==2].mean_f).mean(),g[g.m==0].S.mean(),(g[g.m==0].mean_cf-g[g.m==0].mean_f).mean()))
```