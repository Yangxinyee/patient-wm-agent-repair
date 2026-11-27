Run 293 locates the fragility. Rounding single tensors to bfloat16 moves the h13 prediction by a median:
- 1.06 (glucose) / 1.05 (MAP) for the stage-2 input weights (`ffn.0`: gate offset of 25 plus a twin unit that cancels it);
- 0.74 / 0.75 for the stage-1 value edit (`kv`, entries up to about 1,900);
- under 0.02 for the output weights and `b_in`.

The tool's per-stay scatter beyond my local values is an SD of about 4–6 per shift, roughly 3–4 per run. bfloat16 weight rounding alone gives about 3 per run, so a reduced-precision rollout is a quantitatively plausible cause of the ×½ failure. The cheapest real fix is the gate.

The three stage-2 units compute GELU(Bg·s + 0.5·x) minus a twin GELU(Bg·s). With the gate fully open, the output does not depend on Bg, so I can shrink Bg exactly. I rebuild the rows for several smaller Bg values from the existing gate and dose parts. I check over whole stays that the decoded dose signal entering the core is unchanged at drug bundles and still about zero at non-drug bundles. I then pick the smallest safe Bg, recheck its bfloat16 fragility, and save it if it is clearly lower.

```python
import torch, numpy as np
UA, UO = (502, 131), 140
W0 = WE[F0W].clone(); b0 = WE[F0B].clone()
ws, bs = W0[UO] / 25.0, b0[UO] / 25.0
def ffn0_with(Bg):
    W, b = W0.clone(), b0.clone()
    for uA in UA:
        W[uA] = Bg * ws + (W0[uA] - W0[UO]); b[uA] = Bg * bs + (b0[uA] - b0[UO])
    W[UO] = Bg * ws; b[UO] = Bg * bs
    return W, b
qv = [torch.as_tensor(np.asarray(q, dtype=np.float64)).reshape(-1) for q in (q_I, q_N)]
ST = []
for sid in list(CACHE)[:12]:
    B = make_bundles(tu[tu.stay_id == sid].sort_values('hour'))
    globals()['WCUR'] = WE
    st = run_tf(B[:40], step=stepW, keep=True)
    st = st if isinstance(st, list) else [x for x in st if isinstance(x, list)][0]
    for i, dd_ in enumerate(st): ST.append((dd_, any(a['code'] in DRUGS for a in B[i][1])))
def Ex(Wd, dd_):
    globals()['WCUR'] = Wd
    r = stepW(dd_['out0'], dd_['h0'], dd_['dt'], dd_['e'], dd_['u'], dd_['Mv'], dd_['Ma'])
    return r[3]
E_ref = [Ex(WE, dd_) for dd_, _ in ST]
out = {}
for Bg in (6.0, 8.0, 10.0, 12.0):
    W, b = ffn0_with(Bg); Wd = dict(WE); Wd[F0W] = W; Wd[F0B] = b
    dxd, dxn = [], []
    for (dd_, isd), Er in zip(ST, E_ref):
        dE = (Ex(Wd, dd_) - Er).reshape(-1, Er.shape[-1])
        dx = max(float((dE @ q).abs().max()) for q in qv) * 10.0
        (dxd if isd else dxn).append(dx)
    out[Bg] = (np.max(dxd), np.median(dxd), np.max(dxn), np.median(dxn))
globals()['WCUR'] = WE
print('n steps', len(ST), 'drug', sum(1 for _, i in ST if i))
for Bg, v in out.items(): print(f'Bg={Bg}: drug max|dx|={v[0]:.4f} med={v[1]:.5f} | nondrug max|dx|={v[2]:.4f} med={v[3]:.6f}')
ok = [Bg for Bg, v in out.items() if v[0] < 0.05 and v[2] < 0.05]
Bsel = min(ok) if ok else None
print('selected Bg', Bsel)
if Bsel:
    W, b = ffn0_with(Bsel); Wn = dict(WE); Wn[F0W] = W; Wn[F0B] = b
    fr = {}
    for tag, Wbase in (('cur', WE), ('new', Wn)):
        dev = []
        for sid, (d, ats, dt13, lv) in CACHE.items():
            if not any(a['code'] == 'insulin' for a in ats): continue
            lx, la = lv['glucose_finger']; e2, u2 = enc2(ats)
            vals = []
            for Wd in (Wbase, {**Wbase, F0W: r16(Wbase[F0W]), F0B: r16(Wbase[F0B])}):
                globals()['WCUR'] = Wd
                r2 = stepW(d['out0'], d['h0'], d['dt'], e2, u2, d['Mv'], d['Ma'])
                vals.append(float(pred_mean(r2[0], r2[1], dt13, 'glucose_finger', lx, la, nz=20, seed=0)))
            dev.append(abs(vals[1] - vals[0]))
        fr[tag] = float(np.median(dev))
    globals()['WCUR'] = WE
    print('bf16 ffn0 fragility (median |d| glucose):', fr)
    if fr['new'] < 0.6 * fr['cur']:
        try: save_edit(F0W, W.to(torch.float32)); save_edit(F0B, b.to(torch.float32))
        except Exception: save_edit(F0W, W.numpy().astype(np.float32)); save_edit(F0B, b.numpy().astype(np.float32))
        WE[F0W] = W; WE[F0B] = b
        print('SAVED Bg', Bsel, 'edited:', edited())
with open('/work/notes.md', 'a') as f:
    f.write(f"- Run 293: bf16 fragility by tensor (median |d| h13 glc/MAP): kv .74/.75, ffn0 1.07/1.05, ffn2 .01, b_in .002 -> tool rollout likely low precision.\n- Run 294: gate offset Bg 25->smaller (exact-equivalent rebuild of units 502,131,140). results {out}; selected {Bsel}.\n")
```