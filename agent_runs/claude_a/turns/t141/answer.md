My call to FF passed the prefix with a trailing dot. The helper adds its own `.0.weight`, so the prefix must be `core.prelude.0.ffn`. I'll fix that and rerun the same check and fit. The rest of the plan is unchanged.

```python
x2_=T2+ao_
x2_=x2_+FF(F.layer_norm(x2_,(192,),P[pre_+'norm3.weight'],P[pre_+'norm3.bias']),pre_+'ffn',P)
Echk=F.layer_norm(x2_[:,:4],(192,),P['core.prelude_norm.weight'],P['core.prelude_norm.bias'])
print('E check maxdiff', float((Echk-ST2['E']).abs().max()), 'E scale', float(ST2['E'].abs().mean()))
sdx=x2_[:,:4].std(-1,unbiased=False)
print('sigma_x pct', np.round(np.percentile(sdx.numpy(),[1,25,50,75,99]),2), 'within-bundle slot cv', float((sdx.std(1)/sdx.mean(1)).mean()))
rate_ins=np.zeros(N2); a_i=at2[(at2.event=='insulin')&st_]; rate_ins[a_i.row.values]=a_i.rate.values
nxt=np.full(N2,-1); pv=bd2['prev'].values; sidb=bd2['stay_id'].values
for b_ in range(N2):
    if pv[b_]>=0 and sidb[pv[b_]]==sidb[b_]: nxt[pv[b_]]=b_
mins=bd2['minute'].values
r_next=np.array([rate_ins[nxt[b_]] if (nxt[b_]>=0 and mins[nxt[b_]]-mins[b_]==5) else 0. for b_ in range(N2)])
y_st=rate_ins - np.where(gl, r_next, 0.)
Xf=(A_[:,:4,2:4,:]/sdx[:,:,None,None]).mean(1).reshape(N2,96).numpy()
wfit,_,_,_=np.linalg.lstsq(Xf, y_st, rcond=None); pred=Xf@wfit
for nm_,m_ in types.items():
    if m_.sum(): print(f"{nm_:9s} n={m_.sum():5d} target mean {y_st[m_].mean():6.3f} sd {y_st[m_].std():.3f} | resid sd {np.std(pred[m_]-y_st[m_]):.3f} bias {np.mean(pred[m_]-y_st[m_]):.3f}")
gi_=gl&(r_next>0); print('gluc_read with insulin next: n',gi_.sum(),'resid sd',np.std(pred[gi_]-y_st[gi_]).round(3), '| without:',(gl&(r_next==0)).sum(), np.std(pred[gl&(r_next==0)]).round(3))
```