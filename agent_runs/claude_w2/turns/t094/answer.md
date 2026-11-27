Two ctx changes reduce the z error: dropping the gap term (0.104 → 0.074) and not time-flowing h (0.0935). The other changes (input order, the other output half, other gap scalings, skipping out_norm) make it worse.

Action-only bundles fit better (0.070) than reading-only bundles (0.132). On action bundles e is zero in my code. Their remaining error must therefore come from ctx or u. The extra error on reading bundles points at the measurement encoder as well.

So two pieces look off: how ctx is built (gap and/or time flow), and possibly e.

**This run.** I isolate ctx on the action-only bundles by trying gap/flow combinations, including leaving out the out term or the h term. I also compute the correlation and fitted scale between my z and the stored z. One of three outcomes is expected:
- a combination drops the action-bundle error to near zero, which fixes ctx;
- a high correlation with a scale mismatch, which points at a systematic transform;
- no improvement, which suggests the stored states are not the inputs I assume (for example h stored before the coda).

```python
def zpost_from(ctx, Eb, Ub): return mlp2l(torch.cat([ctx, Eb, Ub], 1), 'z_post', 'gelu')[:, :32]
num, vb, db, rb = atom_feats(ATM, 'left'); xa = embed_atoms(ATM, num, vb, db, rb, U_codes, RU1, True)
E_, U_ = bundle_enc(xa, ATM.bidx.values, ATM.ekind.values == 2, nB, 'gelu')
out_p = xS_t[ip_ - 1, :192]; h_p = xS_t[ip_ - 1, 192:960].reshape(-1, 4, 192)
dt_ = torch.from_numpy(((hS_[ip_] - hS_[ip_ - 1]) * 3600).astype(np.float32))
hf = h_p * torch.exp(-dt_[:, None, None] * torch.exp(Wt['core.flow.a']))
gfull = mlp2l((torch.log(dt_) / 10)[:, None], 'gap', 'gelu')
ztrue = xS_t[ip_, 960:992]
act_m = torch.from_numpy(~hasmeas[ip_] & hi_)
variants = {
  'out+on(hf)+gap': out_p + LNt(hf.mean(1), 'core.out_norm') + gfull,
  'out+on(hf)': out_p + LNt(hf.mean(1), 'core.out_norm'),
  'out+on(h)': out_p + LNt(h_p.mean(1), 'core.out_norm'),
  'out+on(h)+gap': out_p + LNt(h_p.mean(1), 'core.out_norm') + gfull,
  'out': out_p, 'on(hf)': LNt(hf.mean(1), 'core.out_norm'), 'on(hf)+gap': LNt(hf.mean(1), 'core.out_norm') + gfull,
  'out+gap': out_p + gfull, '2*out+gap': 2 * out_p + gfull,
}
for nm_, ctx_ in variants.items():
    zm_ = zpost_from(ctx_, E_[ip_], U_[ip_])
    e_all = (zm_ - ztrue).abs().mean(1); e_act = e_all[act_m].mean()
    c_ = np.corrcoef(zm_.flatten().numpy(), ztrue.flatten().numpy())[0, 1]
    sl_ = float((zm_ * ztrue).sum() / (zm_ * zm_).sum())
    print(f'{nm_:16s} all {float(e_all.mean()):.4f}  action-only {float(e_act):.4f}  corr {c_:.3f}  slope {sl_:.3f}')
print('|out| mean', float(out_p.abs().mean()), '|on(hf)|', float(LNt(hf.mean(1), "core.out_norm").abs().mean()), '|gap|', float(gfull.abs().mean()))
```