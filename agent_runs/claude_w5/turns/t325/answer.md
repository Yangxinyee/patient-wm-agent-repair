Run 295 confirmed the twin structure: the three output columns sum to about 1.5e-7 against column entries up to 4.5. The gain rescaling is therefore exact in the linear regime.

The fragility test then crashed inside the head code. I had overwritten kernel globals: I reused the names `W0`, `b0` and `W2`, and the head's `lin_t` apparently reads a global weights dict by one of these names. That is a self-inflicted bug, not a property of the edit.

To avoid the head entirely, this run measures fragility where it matters: the bfloat16-induced change in the decoded dose signal entering the core (the projection of E on the quiet directions), at all 218 drug steps. I keep the same exactness checks against the current edit at all 480 steps. I check `lin_t`'s globals and restore the weights dict if it uses `W0`, and I use prefixed variable names from now on.

A variant is saved only if it is exact (below 0.05 at drug and non-drug steps) and its signal noise is clearly lower. The tool pass that follows tests the reduced-precision hypothesis: if it holds, the ×½ readouts should move toward their targets.

```python
print('lin_t names', lin_t.__code__.co_names[:12])
if 'W0' in lin_t.__code__.co_names: W0 = W64; print('W0 restored to W64')
zW0, zb0, zW2 = WE[F0W].clone(), WE[F0B].clone(), WE[F2W].clone()
zcI, zcN = zW2[:, UA[0]].clone(), zW2[:, UA[1]].clone()
def zvariant(mI, mN):
    W, b, W2n = zW0.clone(), zb0.clone(), zW2.clone()
    for uA, m in ((UA[0], mI), (UA[1], mN)):
        W[uA] = zW0[UO] + m * (zW0[uA] - zW0[UO]); b[uA] = zb0[UO] + m * (zb0[uA] - zb0[UO])
        W2n[:, uA] = zW2[:, uA] / m
    W2n[:, UO] = zW2[:, UO] + (zcI + zcN) - (zcI / mI + zcN / mN)
    return {**WE, F0W: W, F0B: b, F2W: W2n}
def zfrag(Wd):
    Wr = {**Wd, F0W: r16(Wd[F0W]), F0B: r16(Wd[F0B]), F2W: r16(Wd[F2W])}
    dI, dN = [], []
    for dd_, isd in ST:
        if not isd: continue
        E1, E2 = Ex(Wd, dd_), Ex(Wr, dd_)
        dE = (E2 - E1).reshape(-1, E1.shape[-1])
        dI.append(float((dE @ qv[0]).abs().max()) * 10); dN.append(float((dE @ qv[1]).abs().max()) * 10)
    globals()['WCUR'] = WE
    return round(float(np.median(dI)), 4), round(float(np.median(dN)), 4)
zsumm = {'cur': zfrag(WE)}; zbest = None
for mI, mN in ((3, 10), (4, 20)):
    Wd = zvariant(mI, mN); dxd, dxn = [], []
    for (dd_, isd), Er in zip(ST, E_ref):
        dE = (Ex(Wd, dd_) - Er).reshape(-1, Er.shape[-1])
        dx = max(float((dE @ q).abs().max()) for q in qv) * 10.0
        (dxd if isd else dxn).append(dx)
    globals()['WCUR'] = WE
    fr = zfrag(Wd)
    zsumm[(mI, mN)] = (round(float(np.max(dxd)), 4), round(float(np.max(dxn)), 4), fr)
    if np.max(dxd) < 0.05 and np.max(dxn) < 0.05 and fr[0] < 0.6 * zsumm['cur'][0] and fr[1] < 0.6 * zsumm['cur'][1]:
        zbest = ((mI, mN), Wd)
print('cur frag (median |dx| I, N):', zsumm['cur'])
print('variants (drug max|dx|, nondrug max|dx|, frag):', {k: v for k, v in zsumm.items() if k != 'cur'})
if zbest:
    (mI, mN), Wd = zbest
    for nm in (F0W, F0B, F2W):
        try: save_edit(nm, Wd[nm].to(torch.float32))
        except Exception: save_edit(nm, Wd[nm].numpy().astype(np.float32))
        WE[nm] = Wd[nm]
    print('SAVED m', (mI, mN), 'edited:', edited())
with open('/work/notes.md', 'a') as f:
    f.write(f"- Run 295 crashed: I clobbered kernel global W0/W2/b0 (lin_t uses one) -> use z-prefixed names; W0=W64 restored if needed.\n- Run 296: x-gain scaling (exact, twin adjusted). frag=median bf16 |dx| at drug steps: {zsumm}; saved {zbest[0] if zbest else None}.\n")
```