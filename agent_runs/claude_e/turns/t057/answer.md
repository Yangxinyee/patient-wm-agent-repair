The best setting so far (gap input log(dt)/10, layout B, type 1) gives MAE 3.5 against the tool's predicted means. That is too large to trust for designing the edit.

Several things are now ruled out:
- `last_status[0]` and the rate-bin row 0 are zero vectors.
- All 8 `last_numeric` inputs carry weight, so the layout matters.
- Layout B, which mirrors [value, dose, rate] with has-flags, beats A by a lot.

Next I will break the error down by event and by whether a previous reading exists, and look for a bias or scale pattern. I will also try three variants:
1. the view using gap(log dt) while z comes from the prior with log(dt)/10;
2. the time flow off;
3. a second register layout.

A per-event split or a clear bias should point to the faulty component.

```python
V10=view_of(kv,dtv,'log10'); pm,P=head_mean(F,V10,1,'B',False,code_ids); err=pm-F.predicted_mean.values
hp=F.prev_val.notna().values
for ev in CID:
    mk=(F.event==ev).values
    print(ev, "MAE",np.abs(err[mk]).mean().round(2),"bias",err[mk].mean().round(2),"corr",np.corrcoef(pm[mk],F.predicted_mean.values[mk])[0,1].round(4),
          "| MAE noprev",np.abs(err[mk&~hp]).mean().round(2) if (mk&~hp).any() else None)
print("timescales (s): min %.0f max %.0f"%(np.exp(-Wt['core.flow.a']).min(),np.exp(-Wt['core.flow.a']).max()))
# variant: view uses gap(log dt) but z from prior(ctx with log10)
def view_v2(k,dt,flow=True):
    st=Xs[k]; out=st[:,:192]; hs=st[:,192:960].reshape(-1,4,192)
    hf=hs*np.exp(-dt[:,None,None]*np.exp(Wt['core.flow.a'])[None,None,:]) if flow else hs
    m=lnorm(hf.mean(1),Wt['core.out_norm.weight'],Wt['core.out_norm.bias'])
    g10=L(gelu(L(np.log(dt)[:,None]/10,'gap.0')),'gap.2'); g1=L(gelu(L(np.log(dt)[:,None],'gap.0')),'gap.2')
    z=L(gelu(L(out+m+g10,'z_prior.0')),'z_prior.2')[:,:32]
    return out+m+g1+L(z,'z_head'), out+m+g10+L(z,'z_head')
Va,Vb=view_v2(kv,dtv,True); Vc,_=view_v2(kv,dtv,False)
for nm,VV in [('view gap(log dt)',Va),('noflow log10',view_v2(kv,dtv,False)[1])]:
    pm2,_=head_mean(F,VV,1,'B',False,code_ids); print(nm,"MAE",np.abs(pm2-F.predicted_mean.values).mean().round(3))
```