The edit is badly conditioned. The three repurposed prelude FFN units have input weight norms of about 260 (originally about 0.6) and biases near −14, so they amplify any difference in n3, the normalised FFN input, about 400-fold. The rank-2 b_in edit has singular values 100 and 35, against an original b_in norm of 11. The original model's own tool-versus-local difference at drug bundles is about 1e-3 relative, from a small unresolved implementation detail. Amplified this much, it can explain the tool's edit being about 2× larger and roughly reversed.

To pin down the mechanism I'll decompose the tool's edit effect at the first drug bundle of stay 2020000014 (j3). I'll start from the tool's own state at j2 and compute local response vectors for one unit of injected belief along the insulin and NE directions (vbar_I, vbar_N). Then I'll regress both the tool's and my local edited-minus-original state on those responses. That gives the implied injections in each, plus the residual left unexplained.
- If the tool's effect is mostly explained with flipped or inflated coefficients, the decoded x values are corrupted by n3 differences.
- If much of it is unexplained, the large weights leak into other directions.

```python
sid=2020000014; idx=np.where(S5==sid)[0]; B=TB[sid]
def prep(j, Xtool):
    i_prev = idx[j-1]
    out = torch.tensor(Xtool[i_prev][:192], dtype=torch.float64); h = torch.tensor(Xtool[i_prev][192:960].reshape(4,192), dtype=torch.float64)
    es=[]; ts=[]
    for k in range(j+1):
        e,u = enc2(B[k][1]); es.append(e); ts.append(B[k][0])
    Mv = torch.stack(es[-128:]); Ma = torch.tensor([(B[j][0]-s)*3600 for s in ts[-128:]], dtype=torch.float64)
    return dict(out0=out, h0=h, dt=(B[j][0]-B[j-1][0])*3600, e=e, u=u, Mv=Mv, Ma=Ma)
d0 = prep(3, Xo5)
globals()['WCUR']=W64; r0 = mystep(d0); s0 = st_vec(r0)
globals()['WCUR']=WE;  r1 = mystep(d0); s1 = st_vec(r1)
globals()['WCUR']=W64
vI = torch.as_tensor(np.asarray(vbar_I), dtype=torch.float64); vN = torch.as_tensor(np.asarray(vbar_N), dtype=torch.float64)
rI = st_vec(mystep(d0, dhn=vI.expand(4,192).clone())) - s0; rN = st_vec(mystep(d0, dhn=vN.expand(4,192).clone())) - s0
A_ = np.stack([rI, rN], 1)
for nm, dv in [('local', s1-s0), ('tool', Xe5[idx[3]]-Xo5[idx[3]])]:
    c, *_ = np.linalg.lstsq(A_, dv, rcond=None); res = dv - A_@c
    print(nm, 'coef (vbar_I, vbar_N units)', c.round(3), '|d|', np.linalg.norm(dv).round(2), 'resid frac', (np.linalg.norm(res)/np.linalg.norm(dv)).round(3))
dE = (r1[3]-r0[3]).double()
print('local q.dE per slot I', (dE @ torch.as_tensor(np.asarray(q_I),dtype=torch.float64)).numpy().round(4), 'N', (dE @ torch.as_tensor(np.asarray(q_N),dtype=torch.float64)).numpy().round(4))
print('orig tool vs local err', np.linalg.norm(Xo5[idx[3]]-s0)/np.linalg.norm(s0))
```