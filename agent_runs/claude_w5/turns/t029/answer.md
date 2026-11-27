The per-stay-hour file clarifies the readout:
- Rows are the record's own glucose_finger hours (for norepinephrine, map_arterial hours) in 13–24.
- A row is "reached" once an edited dose record from hour 12 on precedes it. Stay 2020000032 gets its first insulin at hour 17, so its hours 14 and 17 are not reached.
- The headline number is the pooled mean over reached rows.

The model's per-hour shifts are tiny and partly have the wrong sign. In stay 016, halving insulin lowers predicted glucose by up to 2.3 mg/dL by hour 24. That is what a confounded "insulin signals high glucose" association produces.

Next I'll compute my g-estimated true shift for exactly these reached rows: ΔG_T = (k − 1) · Σ_{s=12}^{T−1} γ_{T−s} · u_s, using the recorded doses. Doing the same for norepinephrine with its γ gives concrete targets for these 100 stays, to compare with the model's −0.55 / +0.003 / +0.19 / −0.17.

```python
def true_shift_rows(de, Pdf, drug, gam, factor):
    Pi = Pdf.set_index(['stay_id','h'])[drug]
    out = []
    for _, r in de.iterrows():
        T = int(r.hour); sft = 0.0
        for s_ in range(12, T):
            u = Pi.get((r.stay_id, s_), np.nan)
            if not np.isnan(u) and (T - s_) < len(gam):
                sft += gam[T - s_] * u
        out.append((factor-1)*sft)
    return np.array(out)
gam_ins = gam_all.copy()
res = {}
for ed, drug, gm, fac in [('insulin_x2','insulin',gam_ins,2.0), ('insulin_half','insulin',gam_ins,0.5),
                          ('ne_x2','norepinephrine',gam_ne,2.0), ('ne_half','norepinephrine',gam_ne,0.5)]:
    d = de[(de.edit==ed)&de.reached].copy()
    d['true_hat'] = true_shift_rows(d, PT, drug, gm, fac)
    res[ed] = d
    print(ed, "model:", round(d['shift'].mean(),3), " g-est truth:", round(d.true_hat.mean(),3), " n", len(d))
```