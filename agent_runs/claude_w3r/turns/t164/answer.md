The pooled mean reproduces the summary exactly. Records-based targets on the sel1 tuning stays, from my refined IRFs:
- insulin x2: −13.7
- insulin x1/2: +6.9
- norepinephrine x2: +5.2
- norepinephrine x1/2: −2.6

The original model gives −0.53, +0.09, −1.33 and −0.48.

I'll record this and the factual-cost findings in the notebook. Then I'll test the channel's first link: whether prelude attention head 2 attends to the u token at all bundles, including ones without drugs (otherwise it injects noise), and whether its 48-dim value vector still encodes the rates linearly. R² ≥ 0.98 on head 2's value would confirm the injection can be a rank-1 edit of the prelude output projection.

```python
notes_add = """
## Session 6 (restart 6)
- Kernel fns have no source (exec'd); use as black boxes: stay_data(s)->(B,rows,EU) [B=(t,atoms), atom rate=(code,val)], step_parts(k) with globals B0,rows0,EU0 -> (h?,E,y,out,att[4x7x7]).
- Probe at drug-start bundles (100 train stays, 2406 bundles): E R2 ins .64, ne .49 (noisy: h dominates); u R2 ins .995, ne .984 -> rates clean in u.
- fwd_3 factual residuals (orig) on lags: gf r1..r3 -1.27,-.28,+.42; MAP r1..r3 +4.15,.01,-1.10 -> model MISSES drug effects factually.
  Gaussian dNLL (per reading, all readings): insulin deviation form b=(-2,-.95,-.26) with band comp: -0.0008 (good); insulin levels: +0.026 (bad).
  MAP levels w/o intercept bad (+0.02..0.08); WITH intercept (+mc,r3 free): -0.005 (good); fixed b1=3.45,b2=.72 + free intercept: -0.005. mc coef ~0.
- READOUT = pooled mean over reached stay-hours. Records-IRF targets on sel1 (tuning): ins x2 -13.71, half +6.86, ne x2 +5.17, half -2.58 (n 484/238).
- PLAN: channel u -> prelude head2 (rank-1 on attn.out) -> E -> b_in rank-1 -> h -> out -> value head dead units; comp: insulin band mean via register, norepi intercept.
"""
open('/work/notes.md','a').write(notes_add)
W_kv = load('core.prelude.0.attn.kv.weight').double(); g1 = load('core.prelude.0.norm1.weight').double(); b1n = load('core.prelude.0.norm1.bias').double()
Wv = W_kv[192:]   # assume kv = [k; v]
def LN_(x, g, b): x = torch.as_tensor(x).double(); m = x.mean(-1, keepdim=True); v = x.var(-1, unbiased=False, keepdim=True); return (x-m)/torch.sqrt(v+1e-5)*g + b
V2 = (LN_(UA, g1, b1n) @ Wv[96:144].T).numpy()
for nm, hk in [('ri','has_i'),('rn','has_n')]:
    yv = np.array([r[nm] for r in recE]); msk = np.array([r[hk] for r in recE])
    p = cvp(RidgeCV(alphas=np.logspace(-6,2,17)), V2[msk], yv[msk], groups=gr[msk], cv=GroupKFold(5))
    print(nm, 'head2-value R2', round(1-np.var(yv[msk]-p)/np.var(yv[msk]),4))
# attention of head 2 at non-drug bundles
att_u = []
for s in sorted(recs.stay_id.unique())[:15]:
    B0, rows0, EU0 = stay_data(int(s))
    for k in range(1, len(B0)):
        isdrug = any(a['code'] in (ci,cn) and a['kind']==0 for a in B0[k][1])
        att_u.append((isdrug, float(step_parts(k)[4][2, :4, 5].mean())))
att_u = np.array(att_u); print('head2 attn to u: drug', att_u[att_u[:,0]==1,1].mean().round(4), 'non-drug', att_u[att_u[:,0]==0,1].mean().round(4), att_u[att_u[:,0]==0,1].min().round(4))
```