No type row matches cleanly; the best errors are still 5–7 mg/dL against a prediction spread of 25. So something besides the type is wrong.

The pattern fits 15 = 3 atom kinds × 5 event kinds with type = akind·5 + ekind. That makes measurements type 1, which gives the best MAP fit (1.6 mmHg) and a fair glucose fit. The remaining error is most likely the 8-column register layout.

I will fix type 1 and scan every assignment of (has_last, age, z_value, has_value) to distinct columns, scoring mean absolute error on finger glucose and arterial MAP. The right layout should cut the error to about the Monte Carlo noise of the tool's 4 z draws. If even the best layout leaves several mg/dL, the problem is elsewhere (gap scaling, status row, or z handling).

```python
import itertools, time
t0=time.time()
gp_ = gapv(torch.log(DT)/10.0)
ctx0 = OUT + lnorm(HF.mean(1), 'core.out_norm') + gp_
pr0 = mlp2(ctx0, 'z_prior'); mu0 = pr0[:,:32]
view0 = ctx0 + mu0 @ T('z_head.weight').T + T('z_head.bias')
q1 = lnorm(view0 + T('type_emb.weight')[1], 'query_norm')
Wln = T('last_numeric.weight'); bln = T('last_numeric.bias')
fixed = q1 + T('embed.code.weight')[cid] + bln + T('embed.bins.value.weight')[lb] + T('embed.bins.rate.weight')[0]
tr0 = T('embed_transition.value_bin.weight')[cid*17 + lb]
W0v, b0v, W2v, b2v = T('attr_heads.value_bin.0.weight'), T('attr_heads.value_bin.0.bias'), T('attr_heads.value_bin.2.weight'), T('attr_heads.value_bin.2.bias')
mgf = torch.as_tensor(CODES=='glucose_finger'); mma = torch.as_tensor(CODES=='map_arterial')
def eval_inp(inp, st=0, fx=fixed):
    wc = fx + inp @ Wln.T + T('last_status.weight')[st]
    P = torch.softmax(gelu(wc @ W0v.T + b0v) @ W2v.T + b2v + tr0, -1)
    m_ = (P[:,1:]*cen_mat).sum(1); e_ = (m_-pm_t).abs()
    return float(e_[mgf].mean()), float(e_[mma].mean())
best=[]
for lay in itertools.permutations(range(8), 4):
    inp = torch.zeros(len(DT),8,dtype=torch.float64)
    inp[:,lay[0]] = hl; inp[:,lay[1]] = lage; inp[:,lay[2]] = zv; inp[:,lay[3]] += hl
    egf, ema = eval_inp(inp)
    best.append((egf+ema, egf, ema, lay))
best.sort()
print([ (round(a,2),round(b,2),round(c,2),d) for a,b,c,d in best[:8]], 'time', round(time.time()-t0,1))
```