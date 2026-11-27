Run 83 shows my local head is biased. Its error against the model is 1.46× the 4-draw sampling spread, where pure sampling noise would give about 0.67×. It is also low on average by 1.86 mg/dL. So part of the 2.3 mg/dL mismatch is a real implementation error, not just z sampling.

Likely causes:
- the omitted `last_status` term;
- the units of the register age;
- the `type_emb` row I used;
- how the no-value class 0 is handled in the mean;
- the 4th `last_numeric` input.

Next I'll test each variant on the same 300 readings, using 64 fixed z draws. The correct variant should bring the mean error to about 0 and the median |error| down to about 1.0, the sampling floor. If no variant does, something else is wrong, possibly in ctx or the gap term.

```python
pre = []
for j in idx:
    d = DH.iloc[j]; i0 = int(d.i0)
    out_ = torch.tensor(SX[i0,:192]); h_ = torch.tensor(SX[i0,192:960]).reshape(4,192)
    dt = (d.hour - SHR[i0])*3600.0
    ctx = HD.ctx(out_, h_, dt); mu, ls = HD.zprior(ctx)
    ph = PREV.get((d.stay_id, round(d.hour,4)))
    age = (d.hour-ph)*3600.0 if (ph is not None and ph==ph) else None
    lv = None if d.gl!=d.gl else float(d.gl)
    pre.append((ctx, mu, ls, lv, age, d.pm))
torch.manual_seed(1); EPS = torch.randn(64,32); W = HD.W
def pred(ctx, mu, ls, lv, age, typ=1, status=None, age_unit=60.0, renorm=True, num3=1.0):
    z = mu + torch.exp(ls)*EPS
    view = ctx + HD.L(z,'z_head'); q = HD.LN(view + W['type_emb.weight'][typ],'query_norm')
    c_ = CODE['glucose_finger']; num = torch.zeros(8); lb = 0
    if lv is not None:
        lb = fbin('glucose_finger','value',lv); num[0]=1; num[1]=(math.log(1+age/age_unit)/10 if age else 0.0)
        num[2]=zsc('glucose_finger','value',lv); num[3]=num3
    v = q + W['embed.code.weight'][c_] + HD.L(num,'last_numeric') + W['embed.bins.value.weight'][lb]
    if status is not None: v = v + W['last_status.weight'][status]
    p = torch.softmax(HD.logits(HD.hidden(v), c_*17+lb), -1)
    pn = p[:,1:]/p[:,1:].sum(-1,keepdim=True) if renorm else p[:,1:]
    return float((pn*cen).sum(-1).mean())
def score(**kw):
    e = np.array([pred(*x[:5], **kw) - x[5] for x in pre]); return float(np.median(np.abs(e))), float(e.mean())
for name, kw in [('base',{}),('status0',{'status':0}),('status1',{'status':1}),('age_sec',{'age_unit':1.0}),
                 ('age_hr',{'age_unit':3600.0}),('norenorm',{'renorm':False}),('num3=0',{'num3':0.0})]:
    print(name, np.round(score(**kw),3))
tsc = [(t, score(typ=t)) for t in range(15)]
print('typ scan', [(t, round(a,2), round(b,2)) for t,(a,b) in tsc])
```