For bundles without an action, the model uses the empty-set encoding of the action encoder, not a zero vector as ARCHITECTURE.md says. With that change, MAP-only reading bundles match exactly (z error 0.000) and action bundles are close. Bundles containing glucose still show errors: 0.108 for finger glucose and 0.47 for lab glucose.

Their e encodings are identical to my stored ones, and in run 93 I had reached 0.013 on glucose bundles. So the remaining difference is probably in the context, perhaps the gap term, for these bundles. Next I set `u_none='empty'` and compare my context with the kernel's precomputed pieces (out_p, on_hf, gfull), and the stored ztrue with the states, by bundle kind. If the gap embedding differs for glucose bundles, that is the detail to adopt. If the contexts agree, the difference lies in how the stored glucose E was formed.

```python
u_emp=LN(mlp(F.linear(torch.zeros(1,1),WO['encode_action.size.weight'],WO['encode_action.size.bias']),WO,'encode_action.mlp2'),WO,'encode_action.norm')[0]
Uo2=torch.where(torch.as_tensor(hasu0[rowB])[:,None],Uo,u_emp[None])
print('zerr mine u_empty', zerr_with(Eo,Uo2))
hf_=Hs_[pv]*torch.exp(-dts[:,None,None]*torch.exp(WO['core.flow.a']))
g_=mlp(torch.log(dts)[:,None]/10,WO,'gap','gelu'); on_=LN(hf_.mean(1),WO,'core.out_norm')
for nmv,mine in [('out_p',OUTs[pv]),('on_hf',on_),('gfull',g_),('ztrue',Zs[nf])]:
    ref=globals()[nmv].float() if nmv!='on_hf' else globals()[nmv].float()
    d=(ref-mine).abs().mean(1)
    print(nmv, tuple(ref.shape), {k_:round(float(d[torch.as_tensor(kt==k_)].mean()),4) for k_ in np.unique(kt)})
print('dt_ vs mine', float((torch.as_tensor(np.asarray(dt_)).float()-dts).abs().max()), np.asarray(dt_)[:5], dts[:5])
```