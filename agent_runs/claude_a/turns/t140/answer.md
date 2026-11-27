With kind-correct flags the picture is clean:
- At all 2,484 drug-start bundles (insulin only, insulin+NE, NE only), head 3 puts weight 1.00 on the u token, with no exceptions.
- At END-only bundles it ignores u.
- At glucose reading bundles, head 3 attends e in 93% of cases and head 2 in 95%.

So the rate path through head 3 is exact. The compensation (−expected rate at reading bundles) can come from heads 2 and 3 reading e, though about 5% of reading bundles will be missed.

Next I fit the staging decoder:
- Features: the per-bundle attended value vectors of heads 2 and 3. Each is divided by the per-slot LayerNorm scale of the prelude residual and averaged over the 4 slots, because that is exactly how a write via `prelude_norm` → `b_in` reaches the slot-mean memory.
- Target: +insulin rate at insulin-start bundles, −(insulin rate started 5 minutes later) at glucose reading bundles, 0 elsewhere.
- Before fitting, I check that my reconstructed prelude residual reproduces E.

A good outcome is a residual sd ≲0.1 at insulin bundles and ≲0.3 at reading bundles, which would make the compensation affordable. If the reading-bundle residual is large, that part has to move to the gated design.

```python
x2_=T2+ao_
x2_=x2_+FF(F.layer_norm(x2_,(192,),P[pre_+'norm3.weight'],P[pre_+'norm3.bias']),pre_+'ffn.',P)
Echk=F.layer_norm(x2_[:,:4],(192,),P['core.prelude_norm.weight'],P['core.prelude_norm.bias'])
print('E check maxdiff', float((Echk-ST2['E']).abs().max()), 'E scale', float(ST2['E'].abs().mean()))
sdx=x2_[:,:4].std(-1,unbiased=False)  # N,4
print('sigma_x pct', np.round(np.percentile(sdx.numpy(),[1,25,50,75,99]),2), 'within-bundle slot cv', float((sdx.std(1)/sdx.mean(1)).mean()))
# targets
rate_ins=np.zeros(N2); a_i=at2[(at2.event=='insulin')&st_]; rate_ins[a_i.row.values]=a_i.rate.values
nxt=np.full(N2,-1); pv=bd2['prev'].values
for b_ in range(N2):
    if pv[b_]>=0 and bd2['stay_id'].values[pv[b_]]==bd2['stay_id'].values[b_]: nxt[pv[b_]]=b_
mins=bd2['minute'].values
r_next=np.array([rate_ins[nxt[b_]] if (nxt[b_]>=0 and mins[nxt[b_]]-mins[b_]==5) else 0. for b_ in range(N2)])
y_st=rate_ins - np.where(gl, r_next, 0.)
Xf=(A_[:,:4,2:4,:]/sdx[:,:,None,None]).mean(1).reshape(N2,96).numpy()
wfit,_,_,_=np.linalg.lstsq(Xf, y_st, rcond=None); pred=Xf@wfit
for nm_,m_ in types.items():
    if m_.sum(): print(f"{nm_:9s} n={m_.sum():5d} target mean {y_st[m_].mean():6.3f} sd {y_st[m_].std():.3f} | resid sd {np.std(pred[m_]-y_st[m_]):.3f} bias {np.mean(pred[m_]-y_st[m_]):.3f}")
gi_=gl&(r_next>0); print('gluc_read with insulin next: n',gi_.sum(),'resid sd',np.std(pred[gi_]-y_st[gi_]).round(3), '| without:',(gl&(r_next==0)).sum(), np.std(pred[gl&(r_next==0)]).round(3))
```