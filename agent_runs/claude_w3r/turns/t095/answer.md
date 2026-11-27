Run 88: none of the z-sd variants brings the error down to the pure-sampling level. The best ratio is 1.03 with exp(ls/2), against 0.67 expected. The mean error also moves by about 0.4 between draw sets. So my local head is approximately right (about 2 mg/dL error against a 14 mg/dL sd) but not exact.

That is acceptable for the plan. Any head edit I build will be calibrated end-to-end with forward passes, and one paired "both" run can confirm that my local estimate of an edit's effect matches the model.

Now the key quantities for the design, on the 1865 glucose readings, using the corrected age:
- (a) How well the head-time query q and hidden activations a linearly decode the current insulin rate and its deviation from the band mean (grouped CV R2).
- (b) The model's existing partial response β1 to the dose deviation, from regressing its predicted mean on the deviation with flexible glucose controls.
- (c) Whether "glucose ≥ 155 but no insulin" rows cluster in early hours.

If β1 is near 0 and the decode R2 is low, the state barely carries the dose. That would limit a head-only edit. A high R2 from the hidden activations would favour it.

```python
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import GroupKFold
PREVt = PREV
ok = (DH.i0.values>0) & DH.gl.notna().values
J = np.where(ok)[0]
Q = np.zeros((len(J),192),np.float32); A = np.zeros_like(Q)
for n, j in enumerate(J):
    d = DH.iloc[j]; i0 = int(d.i0)
    out_ = torch.tensor(SX[i0,:192]); h_ = torch.tensor(SX[i0,192:960]).reshape(4,192)
    dt = (d.hour - SHR[i0])*3600.0
    ctx = HD.ctx(out_, h_, dt); mu, ls = HD.zprior(ctx)
    ph = PREVt.get((d.stay_id, round(d.hour,4))); ages = (SHR[i0]-ph)*3600.0 if (ph is not None and ph==ph) else 0.0
    view = ctx + HD.L(mu,'z_head'); q = HD.LN(view + W['type_emb.weight'][1],'query_norm')
    num = torch.zeros(8); lb = fbin('glucose_finger','value',float(d.gl)); num[0]=1; num[1]=math.log(1+ages/60)/10; num[2]=zsc('glucose_finger','value',float(d.gl)); num[3]=1
    x = q + W['embed.code.weight'][CODE['glucose_finger']] + HD.L(num,'last_numeric') + W['embed.bins.value.weight'][lb]
    Q[n] = q.numpy(); A[n] = HD.hidden(x).numpy()
D = DH.iloc[J].reset_index(drop=True); grp = D.stay_id.values
def cvr2(Xm, y):
    yp = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xm, y, grp):
        yp[te] = RidgeCV(alphas=(0.1,1,10,100,1000)).fit(Xm[tr], y[tr]).predict(Xm[te])
    return round(1-np.mean((y-yp)**2)/np.var(y),3)
act = (D.gl.values>=155)
print('P(rs=0 | gl>=155) by hour<=2 / >2:', (D.rs[act & (D.hour<=2)]==0).mean().round(2), (D.rs[act & (D.hour>2)]==0).mean().round(2))
for nm, y in [('rs', D.rs.values), ('dev', D.dev.values)]:
    print(nm, 'R2 q', cvr2(Q, y), ' R2 a', cvr2(A, y), ' (active rows) q', cvr2(Q[act], y[act]) if False else '')
sel = act & (D.hour.values>2)
Xc = np.column_stack([D.dev.values, np.eye(16)[[fbin('glucose_finger','value',g)-1 for g in D.gl.values]], D.gl.values, D.gl.values**2])[sel]
lr = LinearRegression().fit(Xc, D.pm.values[sel]); print('model beta1 on dev (pm ~ dev + gl controls):', lr.coef_[0].round(3), ' n', sel.sum())
lr2 = LinearRegression().fit(Xc, D.obs.values[sel]); print('records beta1 on dev (obs ~ same):', lr2.coef_[0].round(3))
print('sd dev (sel)', D.dev.values[sel].std().round(3), ' sd rs', D.rs.values[sel].std().round(3))
```