At the reading rows, the per-reading `out_`, `ctx_` and flowed slots carry no more action-B residual than Q99 does (R² 0.06–0.10), and action A stays at 0.31. So my q reconstruction is not what loses the information. The state simply holds the rate information only weakly, and a value-head probe cannot recover action B. Whatever I do, a clean, rate-proportional signal has to be written into the state at the action bundle.

Designing that safely needs an exact reconstruction of the action encoder. The one exact check available is the posterior z mean the tool reports after each action bundle: z = z_post([ctx, e, u]). For single-drug bundles holding only the new record, ctx comes from the previous recorded state and e is presumably zero.

I'll write a clean reconstruction of embedding, set encoder, ctx and z_post, then test hypotheses about the drug atom's categorical rows and bin convention. A near-zero z error (under 1e-3) confirms a hypothesis. The error with u = 0 shows the scale of a miss.

```python
P = {n: T2N(load_original(n)) for n in NAMES}
def LN(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + eps) * w + b
def lin_(x, n): return x @ P[n + '.weight'].T + (P[n + '.bias'] if n + '.bias' in P else 0)
def mlp_(x, n): return lin_(gelu_np(lin_(x, n + '.0')), n + '.2')
def setenc(V, pre):
    V = np.atleast_2d(V); n = len(V)
    s = mlp_(V, pre + '.mlp1').sum(0) / np.sqrt(n) + lin_(np.array([np.log(1 + n)]), pre + '.size')
    return LN(mlp_(s, pre + '.mlp2'), P[pre + '.norm.weight'], P[pre + '.norm.bias'])
NUMV = {(e['code'], e['attribute']): e for e in VOCAB['numeric']}
def zsc_(code, att, x): e = NUMV[(code, att)]; return float(np.clip((x - e['median']) / e['iqr'], -10, 10))
def fb_(code, att, x, le=False): ed = np.array(NUMV[(code, att)]['fine_bin_edges']); return int(1 + ((ed <= x) if le else (ed < x)).sum())
def ctx_from(prev, dt):
    out_p = prev[:192]; h = prev[192:960].reshape(4, 192) * np.exp(-dt * np.exp(P['core.flow.a']))
    return out_p + LN(h.mean(0), P['core.out_norm.weight'], P['core.out_norm.bias']) + mlp_(np.array([np.log(dt) / 10]), 'gap')
def zmean(ctx, e, u): return mlp_(np.concatenate([ctx, e, u]), 'z_post')[:32]
# single-atom action-A start bundles (rate on now, off previous hour)
rows = []
for j in np.where(isA)[0]:
    if S2['n_atoms'][j] != 1 or not S2['has_insulin'][j] or j == 0 or sidS[j - 1] != sidS[j]: continue
    t = int(np.floor(hrS[j])); i = sid2i[int(sidS[j])]
    if ins[i, t + 2] > 0 and not (ins[i, t + 1] > 0): rows.append((j, ins[i, t + 2]))
print('n single start bundles', len(rows))
r0 = tr[(tr.event == 'insulin')].iloc[0]; print('rec example', {k: r0[k] for k in ['rate', 'rate_unit', 'dose', 'dose_unit', 'end_hour']})
def test(embf, rows=rows):
    errs = []
    for j, rate in rows[:60]:
        dt = (hrS[j] - hrS[j - 1]) * 3600; ctx = ctx_from(xs[j - 1], dt)
        u = setenc(embf(rate), 'encode_action') if embf is not None else np.zeros(192)
        errs.append(np.abs(zmean(ctx, np.zeros(192), u) - xs[j, 960:992]).mean())
    return np.mean(errs)
print('u=0 err %.4f' % test(None))
print('atom_vec guess err %.4f' % test(lambda r: atom_vec(8, kind=0, ekind=2, zd=zsc_('insulin', 'dose', r), hd=1.0, zr=zsc_('insulin', 'rate', r), hr=1.0, dbin=fb_('insulin', 'dose', r), rbin=fb_('insulin', 'rate', r))))
```