Run 87 rules out the bin side: rows on an edge have small errors (median 1.24), and side='right' changes nothing. The excess error (median 2.08 off-edge) comes from somewhere else.

My next suspect is the scale of the z prior. If the model's log-sd output is really a log-variance or a softplus, its 4 draws would be noisier than I assumed, and the "floor" I'm comparing against would be wrong. I'll test three readings of the prior sd: exp(ls), exp(ls/2) and softplus(ls). For each I compute the many-draw mean and the 4-draw spread, then the ratio median|error| / median(sd4). I will also check whether the error depends on the preceding bundle type, via the state age.
- **Confirms it:** one of the alternatives gives a ratio near 0.67 with mean error near 0. The remaining gap is then just sampling.
- **Refutes it:** all three stay near 1.4. The cause is then in ctx or the state alignment.

```python
def pred_v(ctx, mu, ls, lv, age, sdfun, E):
    z = mu + sdfun(ls)*E
    view = ctx + HD.L(z,'z_head'); q = HD.LN(view + W['type_emb.weight'][1],'query_norm')
    c_ = CODE['glucose_finger']; num = torch.zeros(8); lb = 0
    if lv is not None:
        lb = fbin('glucose_finger','value',lv); num[0]=1; num[1]=(math.log(1+age/60.0)/10 if age else 0.0)
        num[2]=zsc('glucose_finger','value',lv); num[3]=1
    v = q + W['embed.code.weight'][c_] + HD.L(num,'last_numeric') + W['embed.bins.value.weight'][lb]
    p = torch.softmax(HD.logits(HD.hidden(v), c_*17+lb), -1)[:,1:]
    m = (p/p.sum(-1,keepdim=True)*cen).sum(-1)
    return m
torch.manual_seed(2); E160 = torch.randn(160,32)
for nm, f in [('exp', torch.exp), ('exp_half', lambda s: torch.exp(0.5*s)), ('softplus', Fnn.softplus)]:
    rows = []
    for x in pre2:
        m = pred_v(*x[:5], f, E160)
        p4 = torch.softmax  # placeholder
        rows.append((x[5], float(m.mean()), float(m.reshape(40,4).mean(1).std())))
    Rv = np.array(rows); e = Rv[:,1]-Rv[:,0]
    print(nm, 'median|e|', np.median(np.abs(e)).round(3), 'mean', e.mean().round(3), 'median sd4(mean-of-means approx)', np.median(Rv[:,2]).round(3),
          'ratio', (np.median(np.abs(e))/np.median(Rv[:,2])).round(3))
sa = np.array([np.round(x[4]/60) if x[4] is not None else -1 for x in pre2])
for c in [0,5,65,185]:
    mk = sa==c
    if mk.sum()>3: print('state-age', c, 'n', mk.sum(), 'median|e|', np.median(np.abs(e_st[mk])).round(3), 'mean', e_st[mk].mean().round(3))
```