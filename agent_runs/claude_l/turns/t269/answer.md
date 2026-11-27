The architecture notes settled one question. z_prior reads the same ctx the head view is built from. So the head-side path could only amplify the drug-bundle signal by decoding it from the state, and it would amplify the state's natural variation along with it, adding noise.

The better amplifier sits at the source. In the prelude layer, heads 1 and 2 attend almost only to u/z on drug bundles and to e on reading bundles. Suppose I add a rank-one term a·bᵀ to those two heads' value projection, where b is orthogonal to the normalised u tokens and the normalised reading e tokens. Normal tokens would then contribute nothing new. Only a small component of the z token along b, set in proportion to the dose residual, would be strongly amplified. That keeps the signal exactly zero when there is no unexpected dose.

Before building this, I'll check three things on 40 tuning stays:
1. My explicit attention computation, including how the kv weights split into keys and values, reproduces the prelude layer exactly.
2. Heads 1 and 2 are hard enough on drug and reading bundles that leakage onto other tokens is tiny.
3. The span of the normalised u and reading-e tokens leaves free directions.

**What would confirm it:** reconstruction error near zero, off-target attention below about 10⁻³, and a span clearly short of 192. If the span is near full rank or the attention is soft, this amplifier would leak, and I'll fall back to calibrating the gain on the existing channel.

```python
import math, time
t0 = time.time()
def cap_prelude(W, B_):
    times = [b[0] for b in B_]
    out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1, 4, 192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]; res = []
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T)
        hf = h*torch.exp(-dt*torch.exp(W['core.flow.a'])); g = bmlp(W, torch.tensor([[math.log(dt)/10]], dtype=T), 'gap')
        ctx = out + bln(W, hf.mean(1), 'core.out_norm') + g
        zt = blin(W, bmlp(W, torch.cat([ctx, e[None], u[None]], -1), 'z_post')[:, :32], 'z_in')
        tok = torch.cat([hf+W['core.slots'], e[None][:, None], u[None][:, None], zt[:, None]], 1)
        a = bln(W, tok, 'core.prelude.0.norm1'); q = a @ W['core.prelude.0.attn.q.weight'].T; kv = a @ W['core.prelude.0.attn.kv.weight'].T
        ref = bself(W, tok, 'core.prelude.0'); errs = []; prs = []
        for lay in ['split', 'perhead']:
            if lay == 'split': kh = kv[..., :192].reshape(1, 7, 4, 48); vh = kv[..., 192:].reshape(1, 7, 4, 48)
            else: kv4 = kv.reshape(1, 7, 4, 96); kh = kv4[..., :48]; vh = kv4[..., 48:]
            pr = torch.softmax(torch.einsum('bihd,bjhd->bhij', q.reshape(1, 7, 4, 48), kh)/math.sqrt(48), -1)
            x1 = tok + torch.einsum('bhij,bjhd->bihd', pr, vh).reshape(1, 7, 192) @ W['core.prelude.0.attn.out.weight'].T
            x2 = x1 + bmlp(W, bln(W, x1, 'core.prelude.0.norm3'), 'core.prelude.0.ffn'); errs.append(float((x2[:, :4]-ref[:, :4]).abs().max())); prs.append(pr)
        res.append((bool(ua), any(a_['code'] in MEASC and a_['kind'] == 0 for a_ in ea), errs, prs[0][0, :, :4, :].numpy(), a[0, 4].numpy().copy(), a[0, 5].numpy().copy(), a[0, 6].numpy().copy()))
        out, h, _ = bstep_z(W, out, h, dt, e[None], u[None], Mt, ages)
    return res
PC = []
for s in b1[:40]: PC += cap_prelude(Wt, BUN[s])
er = np.array([r[2] for r in PC]); print('bundles', len(PC), 'recon err max split/perhead', er.max(0), 'time', round(time.time()-t0, 1))
drug = np.array([r[0] for r in PC]); read = np.array([r[1] for r in PC]); P_ = np.stack([r[3] for r in PC])
for hd in range(4):
    wuz = P_[drug][:, hd, :, 5] + P_[drug][:, hd, :, 6]; oth_d = np.max(P_[drug][:, hd, :, :5], axis=-1)
    we = P_[read][:, hd, :, 4]; oth_r = np.max(np.concatenate([P_[read][:, hd, :, :4], P_[read][:, hd, :, 5:]], -1), axis=-1)
    print('head', hd, 'drug: w(u+z) mean/min', round(float(wuz.mean()), 4), round(float(wuz.min()), 4), 'max other', round(float(oth_d.max()), 4), 'z share mean', round(float((P_[drug][:, hd, :, 6]/(wuz+1e-12)).mean()), 3),
          '| read: w(e) mean/min', round(float(we.mean()), 4), round(float(we.min()), 4), 'max other', round(float(oth_r.max()), 4), 'p99 other', round(float(np.quantile(oth_r, .99)), 4))
NE1 = np.stack([r[4] for r in PC if r[1]]); NU1 = np.vstack([np.stack([r[5] for r in PC]), np.stack([r[4] for r in PC if r[0]])])
for nm, Mx in [('norm1(e reading)', NE1), ('norm1(u)+norm1(e drug)', NU1), ('all', np.vstack([NE1, NU1]))]:
    sv = np.linalg.svd(Mx, compute_uv=False); print(nm, Mx.shape, 'sv ratios at 20,40,60,80,100,120,150,180:', [float(sv[i]/sv[0]) for i in [20, 40, 60, 80, 100, 120, 150, 180] if i < len(sv)])
```