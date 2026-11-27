`zl2` is just zl²/10, a nonlinear stand-in my earlier search picked up. It does not correspond to a real input, so the true discrepancy lies elsewhere.

One clue: readings with an empty register are accurate for MAP but not for glucose. Those rows share the same state and q, so whatever differs must be code-specific. The type embedding is a good candidate. type_emb has 15 rows, matching the 15 count types, so the type index may differ per code rather than being 1 for all.

Next, for each code separately, I scan all 15 type indices and measure the mean absolute error and bias of the predicted means against the tool. If one index per code brings every code down to the noise floor (about 1.0) without bias, the type is code-specific. If type 1 stays best everywhere, the problem is elsewhere, probably the register features.

```python
res_t=[]
for ty_ in range(15):
    q=LN(ctx+F.linear(mu,W['z_head.weight'],W['z_head.bias'])+W['type_emb.weight'][ty_],W,'query_norm')
    wc=q+W['embed.code.weight'][codes_]+F.linear(torch.as_tensor(f8m),W['last_numeric.weight'],W['last_numeric.bias'])+W['embed.bins.value.weight'][lb_]
    lg=mlp(wc,W,'attr_heads.value_bin')+W['embed_transition.value_bin.weight'][codes_*17+lb_]
    pm=(torch.softmax(lg,-1)[:,1:].numpy()*cm).sum(1); d=pm-fa.predicted_mean.values
    for ev_ in fa.event.unique():
        m_=(fa.event.values==ev_); res_t.append((ty_,ev_,round(np.abs(d[m_]).mean(),3),round(d[m_].mean(),3)))
rt=pd.DataFrame(res_t,columns=['ty','event','mae','bias'])
print(rt.sort_values('mae').groupby('event').head(3).sort_values(['event','mae']).to_string())
```