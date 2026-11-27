Nothing changed: same NLL and the same early-hour error. So run_cf doesn't read the stored states through the global `S` I replaced; most likely it uses the X argument (Xs, the tool's states) directly.

Next I build the replica-state array in exactly the row order of the original store (S_orig's stay_id order, bundles in order). I pass it as X and test stay …024 with the edit restart at hour 12.
- If the early-hour error drops to about 2, I regenerate the dataset with X = this array.
- If it doesn't, the copied rows come from somewhere else, and I'll restart every variant from bundle 1 using bundle-hour gating.

```python
S=S_orig; sid0=np.asarray(S_orig['stay_id']); Xrep=np.zeros((len(sid0),992))
for s in sorted(BUN.keys()):
    O,H,k0=run_cf(s,{},BUN[s][1][0],Xs); n=O.shape[0]; pos=np.where(sid0==s)[0]
    Xrep[pos]=np.concatenate([O.numpy(),H.numpy().reshape(n,768),np.zeros((n,32))],1)
print('Xrep vs Xs out diff first-stay %.4f'%np.abs(Xrep[:54,:192]-np.asarray(Xs[:54,:192])).max())
s=2020000024; rt=RT[RT.stay_id==s].sort_values('b').reset_index(drop=True); eps=torch.tensor(np.random.default_rng(int(s)%1000003).standard_normal((len(rt),4,32)))
for XX,nm in ((Xs,'Xs'),(torch.tensor(Xrep),'Xrep_t'),(Xrep,'Xrep_np')):
    try:
        O,H,k0=run_cf(s,{},720,XX); q,wc,P,mn=heads_on_states(O,H,s,rt,eps); mn=np.asarray(mn,float)
        e=np.abs(mn-x[pmc].values); hh=x.hour.values; print(nm,'k0',k0,'err h<=12 %.2f h>12 %.2f'%(e[hh<=12].mean(),e[hh>12].mean()))
    except Exception as ex: print(nm,'ERR',str(ex)[:150])
```