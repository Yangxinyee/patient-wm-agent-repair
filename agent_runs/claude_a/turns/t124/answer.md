This finds a clean route for the dose write. **Prelude head 3 attends to the u (dose) token with weight exactly 1.0 (sd 0.0) at every insulin bundle, in all four slots.** So head 3's output there is an exact linear function of the normalised dose embedding. That is far cleaner than decoding from E, where even an insulin-only decoder reaches only R² 0.92 (residual sd 0.29). The signal I need is the within-tier deviation, sd 0.59.

My plan is to write the rate into the memory dimension through the row of head 3's output projection. A reference dimension can cancel LayerNorm's common-mode term.

**Next step:** check three things.
1. How precisely the insulin start rate is linearly decodable from LN1(u) itself, and whether bundles with only NE or end atoms decode to ≈0.
2. Whether head 3 also attends to u when there is no action. Then its output is constant there, which is good: no write.
3. Which head attends to the measurement token e at glucose-reading bundles, as a possible clean route for the −rbar(tier) baseline write.

**What would confirm or refute the route:** LN1(u) decoding at R² ≳ 0.99 with near-zero output elsewhere confirms it. Otherwise I need a different write design.

```python
U = dsA['u'].float() if torch.is_tensor(dsA['u']) else torch.tensor(dsA['u']).float()
Ee = dsA['e'].float() if torch.is_tensor(dsA['e']) else torch.tensor(dsA['e']).float()
n1w, n1b = P['core.prelude.0.norm1.weight'].float(), P['core.prelude.0.norm1.bias'].float()
LNu = torch.nn.functional.layer_norm(U, (192,), n1w, n1b).numpy().astype(np.float64)
LNe = torch.nn.functional.layer_norm(Ee, (192,), n1w, n1b).numpy().astype(np.float64)
uzero = (U.abs().sum(1)==0).numpy()
yR = np.zeros(nbA); yR[ia.index.values] = ia.values          # insulin start rate per bundle
ne_at = at_[(at_.event=='norepinephrine')]
has_ne = np.zeros(nbA,bool); has_ne[ne_at.row.unique()] = True
has_iend = np.zeros(nbA,bool); has_iend[at_[(at_.event=='insulin')&(at_.kind==2)].row.unique()] = True
print("u==0 bundles:", int(uzero.sum()), " ins-start:", int(ins_m.sum()), " ne:", int(has_ne.sum()), " ins-end:", int(has_iend.sum()))
X_ = np.c_[LNu, np.ones(nbA)]
rgu = Ridge(alpha=1e-4, fit_intercept=False).fit(X_[tr_m], yR[tr_m]); pu = rgu.predict(X_)
for nm_, msk_ in [('ins',ins_m),('ne_only',has_ne&~ins_m),('iend_only',has_iend&~ins_m&~has_ne),('u0',uzero)]:
    mm_ = te_m&msk_
    if mm_.sum(): print(f"{nm_}: n{mm_.sum()} resid sd {np.std(yR[mm_]-pu[mm_]):.4f} bias {np.mean(yR[mm_]-pu[mm_]):+.4f}")
r_ins = yR[te_m&ins_m]-pu[te_m&ins_m]; print("R2 rate on ins test: %.5f"%(1-r_ins.var()/yR[te_m&ins_m].var()))
gr_m = btype=='gread'
aw_g = AW[torch.tensor(gr_m)]; aw_o = AW[torch.tensor(uzero & ~gr_m)]
print("gread: attn slots->e by head", [round(float(aw_g[:,hh,:4,4].mean()),3) for hh in range(4)], " ->u", [round(float(aw_g[:,hh,:4,5].mean()),3) for hh in range(4)], " sd->e", [round(float(aw_g[:,hh,:4,4].std()),3) for hh in range(4)])
print("no-action non-read: ->u", [round(float(aw_o[:,hh,:4,5].mean()),3) for hh in range(4)], " ->e", [round(float(aw_o[:,hh,:4,4].mean()),3) for hh in range(4)])
```