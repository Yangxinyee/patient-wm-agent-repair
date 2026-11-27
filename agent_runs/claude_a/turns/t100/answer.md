The kernel still holds what I need:

- the per-bundle encodings `eALL` and `uALL`;
- the states from the forward tool for the sel2 stays: `HS` (h after each bundle), `OUT` and `ZP` (z posterior mean);
- the bundle table `SB`, which gives each bundle's predecessor row and time gap.

So I can rebuild every core step in parallel from the stored states, without running stays one after another.

**Plan for this run.**
1. Write a fresh, fully explicit version of one core step, so I can look inside the prelude and later apply edits.
2. Check it against the tool's stored h and `out`.
3. Measure how much attention each prelude head gives from the four slot tokens to the u token (and the e and z tokens), separately for bundles with insulin, norepinephrine, or no action.

**How I will read the result.**
- *Feasible write path:* the step matches the stored states (error near 1e-3 or below), and at least one head puts a large, stable share of slot attention on the u token in action bundles.
- *Engineered key needed:* that attention is small or varies widely. Then I would have to build a dedicated key direction instead.

```python
import math, torch, numpy as np, pandas as pd
import torch.nn.functional as F
PO = {n: torch.as_tensor(load_original(n)).float() for n in NAMES}
def LN(x, pre, P): return F.layer_norm(x, (192,), P[pre+'.weight'], P[pre+'.bias'], 1e-5)
def FF(x, pre, P): return F.linear(F.gelu(F.linear(x, P[pre+'.0.weight'], P[pre+'.0.bias'])), P[pre+'.2.weight'], P[pre+'.2.bias'])
def MH(xq, xkv, pre, P, mask=None, retw=False):
    q = xq @ P[pre+'.q.weight'].T; kv = xkv @ P[pre+'.kv.weight'].T
    k, v = kv[..., :192], kv[..., 192:]
    sh = lambda t: t.reshape(*t.shape[:-1], 4, 48).transpose(-2, -3)
    s = sh(q) @ sh(k).transpose(-1, -2) / math.sqrt(48)
    if mask is not None: s = s.masked_fill(~mask, float('-inf'))
    a = torch.softmax(s, -1)
    o = (a @ sh(v)).transpose(-2, -3).reshape(*q.shape[:-1], 192) @ P[pre+'.out.weight'].T
    return (o, a) if retw else o
def AGE(age_s, P):
    return F.linear(F.gelu(F.linear(torch.log1p(age_s)[..., None], P['core.age.0.weight'], P['core.age.0.bias'])), P['core.age.2.weight'], P['core.age.2.bias'])
def step_b(hprev, e, u, z, dt, Mx, Mage, Mmask, P, ret=False):
    hf = hprev * torch.exp(-dt[:, None, None] * torch.exp(P['core.flow.a']))
    zin = F.linear(z, P['z_in.weight'], P['z_in.bias'])
    T = torch.cat([hf + P['core.slots'], e[:, None], u[:, None], zin[:, None]], 1)
    n1 = LN(T, 'core.prelude.0.norm1', P); ao, aw = MH(n1, n1, 'core.prelude.0.attn', P, retw=True)
    T = T + ao; T = T + FF(LN(T, 'core.prelude.0.norm3', P), 'core.prelude.0.ffn', P)
    E = LN(T[:, :4], 'core.prelude_norm', P); x = hf + E
    kvin = Mx + AGE(Mage, P)
    for L in ('core.blocks.0.0', 'core.blocks.0.1'):
        n = LN(x, L+'.norm1', P); x = x + MH(n, n, L+'.attn', P)
        n = LN(x, L+'.norm2', P); x = x + MH(n, kvin, L+'.cross', P, mask=Mmask[:, None, None, :])
        n = LN(x, L+'.norm3', P); x = x + FF(n, L+'.ffn', P)
    sp = F.softplus(P['core.delta_in']); abar = torch.exp(-sp * torch.exp(P['core.a_in']))
    hn = abar * hf + sp * (E @ P['core.b_in.weight'].T) + (x - (hf + E))
    y = hn @ P['core.c_out.weight'].T
    n = LN(y, 'core.coda.0.norm1', P); y = y + MH(n, n, 'core.coda.0.attn', P)
    n = LN(y, 'core.coda.0.norm3', P); y = y + FF(n, 'core.coda.0.ffn', P)
    out = LN(y.mean(1), 'core.out_norm', P)
    return (y, out, dict(E=E, hf=hf, aw=aw, hn=hn)) if ret else (y, out)
tsec = torch.tensor(MB['hour'].values * 3600.0, dtype=torch.float32)
def stay_inputs(rows_):
    rows_ = np.asarray(rows_); n_ = len(rows_)
    prev = SB['prev'].values[rows_]
    hprev = torch.stack([HS[int(p)] if not np.isnan(p) else torch.zeros(4, 192) for p in prev])
    dt = torch.tensor([SB['dts'].values[r] if not np.isnan(SB['prev'].values[r]) else 1.0 for r in rows_], dtype=torch.float32)
    t = tsec[rows_]; Mx = eALL[rows_][None].expand(n_, n_, 192)
    ages = (t[:, None] - t[None, :]).clamp(min=0); mask = torch.arange(n_)[None, :] <= torch.arange(n_)[:, None]
    return hprev, eALL[rows_], uALL[rows_], ZP[rows_], dt, Mx, ages, mask
stays_SB = SB['stay_id'].unique()
errs_h, errs_o, AW, Rw = [], [], [], []
for s_ in stays_SB:
    rows_ = np.where(SB['stay_id'].values == s_)[0]
    inp = stay_inputs(rows_)
    with torch.no_grad():
        y_, o_, aux = step_b(*inp, PO, ret=True)
    errs_h.append((y_ - HS[rows_]).abs().max().item()); errs_o.append((o_ - OUT[rows_]).abs().max().item())
    AW.append(aux['aw'][:, :, :4, :].mean(2)); Rw.append(rows_)
AW = torch.cat(AW); Rw = np.concatenate(Rw)
print('max|dH|', max(errs_h), 'max|dOut|', max(errs_o), 'h scale', HS.abs().mean().item())
# bundle type from atoms (at2 should be sel2 atoms)
print('at2 stays in SB:', np.isin(at2['stay_id'].unique(), stays_SB).mean(), at2.columns.tolist())
key = pd.Series(np.arange(len(SB)), index=pd.MultiIndex.from_arrays([SB['stay_id'].values, SB['minute'].values]))
at2_ = at2[(at2['kind'] == 0)]
ins_b = set(key.reindex(pd.MultiIndex.from_frame(at2_[at2_['event'] == 'insulin'][['stay_id', 'minute']])).dropna().astype(int))
ne_b = set(key.reindex(pd.MultiIndex.from_frame(at2_[at2_['event'] == 'norepinephrine'][['stay_id', 'minute']])).dropna().astype(int))
typ_ = np.array(['ins' if (r in ins_b and r not in ne_b) else 'ne' if (r in ne_b and r not in ins_b) else 'both' if (r in ins_b) else ('act' if uALL[r].abs().sum() > 0 else 'none') for r in Rw])
tok = ['s1', 's2', 's3', 's4', 'e', 'u', 'z']
for ty in ['ins', 'ne', 'both', 'act', 'none']:
    m_ = typ_ == ty
    if m_.sum() == 0: continue
    A_ty = AW[torch.tensor(m_)]
    print(ty, m_.sum(), ' | '.join(f"h{hd}: " + ' '.join(f"{tok[j]}{A_ty[:, hd, j].mean():.2f}" for j in range(7)) + f" (u sd {A_ty[:, hd, 5].std():.2f})" for hd in range(4)))
```