The way the prior's spread is parameterised barely matters. All four forms give a mean-prediction error of about 2.0, even with the spread set to zero. So the remaining discrepancy is systematic, not sampling noise.

By code, the error is largest for glucose_finger readings that have a previous value (3.1 mg/dL) and smallest for map_arterial (0.9 mmHg). That points to the register term, which matters most when a previous value exists, rather than to the shared state path.

Two register details are undocumented: which of the 8 slots of `last_numeric` hold "has" flags, and whether a `last_status` row is added. In run 85, adding last_status row 0 helped slightly.

I will run a fast deterministic grid (z at the prior mean) over:
- on/off flags at register slots 3–7;
- the last_status options (none, row 0, row 1, or depending on whether a previous value exists).

A combination that drops the glucose error to near zero would make the head exact. If no combination helps, the register term is not the cause. I would then move on and treat the head replica as approximate (error about 2 mg/dL, about 0.1 predictive sd), with final checks by the tool.

```python
import itertools
PRE = []
for d in READ:
    prev, dt = d['prev'], d['dt']
    h = prev[192:960].reshape(4,192)*np.exp(-dt*np.exp(Wo['core.flow.a']))
    ctx = prev[:192] + LNn(h.mean(0),'core.out_norm') + mlp2n(np.array([np.log(dt)/10]),'gap')
    pr = mlp2n(ctx,'z_prior')
    q = LNn(ctx + linn(pr[:32],'z_head') + Wo['type_emb.weight'][1], 'query_norm')
    PRE.append(q)
PRE = np.array(PRE)
def eval_det(flags, ls):
    ae=[]; dn=[]
    for d, q in zip(READ, PRE):
        has, agemin, zl, vb = d['last']
        lv = np.array([has, np.log1p(agemin)/10, zl] + [has*f for f in flags], float)
        wc = q + Wo['embed.code.weight'][CID[d['code']]] + linn(lv,'last_numeric') + Wo['embed.bins.value.weight'][vb] + Wo['embed.bins.rate.weight'][0]
        if ls=='0': wc = wc + Wo['last_status.weight'][0]
        elif ls=='1': wc = wc + Wo['last_status.weight'][1]
        elif ls=='has' and has: wc = wc + Wo['last_status.weight'][0]
        lg = linn(gelu(linn(wc,'attr_heads.value_bin.0')),'attr_heads.value_bin.2') + Wo['embed_transition.value_bin.weight'][CID[d['code']]*17+vb]
        lg -= lg.max(); P = np.exp(lg); P /= P.sum()
        ae.append(abs(d['pm'] - P[1:]@CENT[d['code']])); dn.append(abs(d['nll'] + np.log(P[d['ob']])))
    return np.mean(ae), np.mean(dn)
res_g = []
for flags in itertools.product([0,1], repeat=5):
    for ls in ['none','0','1','has']:
        a, b = eval_det(flags, ls); res_g.append((a, b, flags, ls))
res_g.sort(key=lambda r: r[0])
for r in res_g[:8]: print('MAE %.3f |dNLL| %.4f' % (r[0], r[1]), r[2], r[3])
print('reference (1,1,0,0,0) none:', [r[:2] for r in res_g if r[2]==(1,1,0,0,0) and r[3]=='none'])
```