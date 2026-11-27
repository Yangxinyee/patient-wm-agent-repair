Run 294 shows that lowering the gate offset is not safe. It leaves drug bundles exactly unchanged, but at a few non-drug bundles it changes the decoded dose signal by 4–6 units, even at Bg = 12. That change shrinks only linearly as Bg grows, so those bundles sit on a partly open gate, and a smaller Bg would change behaviour there.

A safer route to the same robustness is to enlarge the dose signal relative to the gate offset, rather than shrinking the offset. I can multiply the dose part of each stage-2 unit by m (larger for norepinephrine, whose decoded residual is small, about ±0.3) and divide that unit's output column by m. I also adjust the twin's column so the gate term is unchanged. In exact arithmetic this is identical wherever the GELUs stay linear, and the dose signal becomes m times larger than the rounding error of the offset.

This run checks the twin structure and tests two settings, (m_I, m_N) = (3, 10) and (4, 20). For each it checks the change in the decoded signal at all 480 steps and the bfloat16 fragility (rounding `ffn.0` and `ffn.2`). It saves the best setting only if it is exactly equivalent (all changes below 0.05) and markedly less fragile. The next forward pass would then show whether the tool's ×½ readouts respond, which tests the reduced-precision hypothesis directly.

```python
W2 = WE[F2W].clone()
cAI, cAN, cO = W2[:, UA[0]].clone(), W2[:, UA[1]].clone(), W2[:, UO].clone()
print('twin sum |cAI+cAN+cO| max', float((cAI + cAN + cO).abs().max()), 'vs |cO| max', float(cO.abs().max()))
def variant(mI, mN):
    W, b, W2n = W0.clone(), b0.clone(), W2.clone()
    for uA, m in ((UA[0], mI), (UA[1], mN)):
        W[uA] = W0[UO] + m * (W0[uA] - W0[UO]); b[uA] = b0[UO] + m * (b0[uA] - b0[UO])
        W2n[:, uA] = W2[:, uA] / m
    W2n[:, UO] = W2[:, UO] + (cAI + cAN) - (cAI / mI + cAN / mN)
    Wd = dict(WE); Wd[F0W] = W; Wd[F0B] = b; Wd[F2W] = W2n
    return Wd
def frag(Wbase, drug, oc):
    dev = []
    for sid, (d, ats, dt13, lv) in CACHE.items():
        if not any(a['code'] == drug for a in ats): continue
        lx, la = lv[oc]; e2, u2 = enc2(ats); vals = []
        for Wd in (Wbase, {**Wbase, F0W: r16(Wbase[F0W]), F0B: r16(Wbase[F0B]), F2W: r16(Wbase[F2W])}):
            globals()['WCUR'] = Wd
            r2 = stepW(d['out0'], d['h0'], d['dt'], e2, u2, d['Mv'], d['Ma'])
            vals.append(float(pred_mean(r2[0], r2[1], dt13, oc, lx, la, nz=20, seed=0)))
        dev.append(abs(vals[1] - vals[0]))
    globals()['WCUR'] = WE
    return round(float(np.median(dev)), 3)
summ = {'cur': (0, 0, frag(WE, 'insulin', 'glucose_finger'), frag(WE, 'norepinephrine', 'map_arterial'))}
best = None
for mI, mN in ((3, 10), (4, 20)):
    Wd = variant(mI, mN); dxd, dxn = [], []
    for (dd_, isd), Er in zip(ST, E_ref):
        dE = (Ex(Wd, dd_) - Er).reshape(-1, Er.shape[-1])
        dx = max(float((dE @ q).abs().max()) for q in qv) * 10.0
        (dxd if isd else dxn).append(dx)
    globals()['WCUR'] = WE
    fI, fN = frag(Wd, 'insulin', 'glucose_finger'), frag(Wd, 'norepinephrine', 'map_arterial')
    summ[(mI, mN)] = (round(float(np.max(dxd)), 4), round(float(np.max(dxn)), 4), fI, fN)
    if np.max(dxd) < 0.05 and np.max(dxn) < 0.05 and fI < 0.6 * summ['cur'][2]:
        best = ((mI, mN), Wd)
print('variant: (drug max|dx|, nondrug max|dx|, bf16 frag glc, bf16 frag MAP):', summ)
if best:
    (mI, mN), Wd = best
    for nm in (F0W, F0B, F2W):
        try: save_edit(nm, Wd[nm].to(torch.float32))
        except Exception: save_edit(nm, Wd[nm].numpy().astype(np.float32))
        WE[nm] = Wd[nm]
    print('SAVED m', (mI, mN), 'edited:', edited())
with open('/work/notes.md', 'a') as f:
    f.write(f"- Run 294: lowering Bg changes nondrug steps by 4-6 x-units (rare steps with s>0) -> rejected.\n- Run 295: x-gain scaling m (exact in linear regime, ffn2 cols /m, twin adjusted): {summ}; saved {best[0] if best else None}.\n")
```