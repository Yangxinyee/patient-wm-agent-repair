The register is settled. The documented-style layout [has, log age, z_last, has, has, 0, 0, 0] is best, and last_status row 0 changes nothing, so that row must be zero. The register is not the source of the remaining error of about 2.

Two cheap things remain to rule out. First, the type embedding index: I only compared types 0–2, and there are 15. Second, the time unit inside the view's gap term: seconds, minutes or hours under the /10 scaling. The posterior context is known to use seconds, but the view might not.

I run both deterministically, with z at the prior mean. An index or unit that cuts the error clearly below 2.08 would fix the head. If nothing improves, the residual is most likely the tool's 4-draw sampling noise plus float32 effects. I would then accept the head as is: about 0.1 predictive sd for the mean, and good enough for design work.

```python
def eval_det2(ty, gunit):
    ae=[]; dn=[]
    for d in READ:
        prev, dt = d['prev'], d['dt']
        h = prev[192:960].reshape(4,192)*np.exp(-dt*np.exp(Wo['core.flow.a']))
        base = prev[:192] + LNn(h.mean(0),'core.out_norm')
        ctx = base + mlp2n(np.array([np.log(dt)/10]),'gap')
        pr = mlp2n(ctx,'z_prior')
        V = base + mlp2n(np.array([np.log(dt/gunit)/10]),'gap') + linn(pr[:32],'z_head')
        q = LNn(V + Wo['type_emb.weight'][ty], 'query_norm')
        has, agemin, zl, vb = d['last']
        wc = q + Wo['embed.code.weight'][CID[d['code']]] + linn(np.array([has,np.log1p(agemin)/10,zl,has,has,0,0,0],float),'last_numeric') + Wo['embed.bins.value.weight'][vb] + Wo['embed.bins.rate.weight'][0]
        lg = linn(gelu(linn(wc,'attr_heads.value_bin.0')),'attr_heads.value_bin.2') + Wo['embed_transition.value_bin.weight'][CID[d['code']]*17+vb]
        lg -= lg.max(); P = np.exp(lg); P /= P.sum()
        ae.append(abs(d['pm'] - P[1:]@CENT[d['code']])); dn.append(abs(d['nll'] + np.log(P[d['ob']])))
    return np.mean(ae), np.mean(dn)
for ty in range(15):
    print('type', ty, 'MAE %.3f |dNLL| %.4f' % eval_det2(ty, 1.0))
for gu in [60.0, 3600.0]:
    print('gap unit', gu, 'MAE %.3f |dNLL| %.4f' % eval_det2(1, gu))
```