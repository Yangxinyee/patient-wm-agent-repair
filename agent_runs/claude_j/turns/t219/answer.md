Most stays (96 of 100) show large replica-vs-tool errors at hours ≤ 12, up to 60 mg/dL. Yet the direct recomputation for the first stay matched within 1.6. So the stored factual chunk in M, or the merge used to compare it, is probably misaligned rather than the replica being wrong.

Next I recompute the worst stay (…024) directly, as in the successful check. I compare it both with the tool and with M's stored factual means for that stay.
- If the direct computation matches the tool but M doesn't, the bug is in how cf_stay stored the rows.
- If neither matches, something about this stay breaks the replica (for example its first bundles).

```python
s=2020000024; rt=RT[RT.stay_id==s].sort_values('b').reset_index(drop=True)
eps=torch.tensor(np.random.default_rng(int(s)%1000003).standard_normal((len(rt),4,32)))
O,H,k0=run_cf(s,{},720,Xs); q,wc,P,mn=heads_on_states(O,H,s,rt,eps); mn=np.asarray(mn,float)
rt['hour']=np.round(rt.minute/60).astype(int); rt['mn']=mn
x=rt.merge(fc[['stay_id','hour','event','predicted_mean']],on=['stay_id','hour','event'],how='left',suffixes=('','_t'))
pmc='predicted_mean_t' if 'predicted_mean_t' in x else 'predicted_mean'
print('direct vs tool h<=12 %.2f h>12 %.2f'%((x.mn-x[pmc]).abs()[x.hour<=12].mean(),(x.mn-x[pmc]).abs()[x.hour>12].mean()))
mf=M[(M.stay==s)&NONE].sort_values('b'); print('M none rows',len(mf),'vs rt',len(rt),'| M mean_f vs direct %.3f'%np.abs(mf.mean_f.values-mn[:len(mf)]).mean() if len(mf)==len(rt) else '')
print('RT pm vs tool %.2f'%(x.predicted_mean-x[pmc]).abs().mean() if pmc!='predicted_mean' else 'no RT pm col')
```