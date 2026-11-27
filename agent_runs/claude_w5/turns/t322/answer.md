Run 292 changes how I read the earlier comparisons.

The recorded-dose mismatch against the tool's `factual` column is the same with original weights at the 12:05 step (3.23) as with the edit (3.26). So that "base" comparison never told me anything about the edit. It reflects some convention in the tool's `factual` column that I don't reproduce, and I drop it as evidence.

The run did reveal a real problem: the edit is numerically fragile. Rounding the weights to bfloat16 moves the edited model's h13 prediction by a median 2.0 mg/dL; the original moves by 0.01. Under bfloat16 the insulin means also move toward the tool (×2 −3.79 → −4.44, ×½ +1.83 → +1.25), a common-mode negative shift like the tool's artifact.

A likely source is my gate design. Each stage-2 unit computes GELU(25·s + 0.5·x) and a twin subtracts GELU(25·s), so a small dose signal is recovered by cancelling two numbers near 22. The ×1,900 stage-1 value weights are another candidate.

This run locates the fragility by rounding one edited tensor at a time to bfloat16, measuring the change in the h13 prediction, and caching the per-stay step states for reuse. It also checks that `build_edit` reproduces the current tensors, so I can rebuild with a smaller gate offset at once. If the FFN tensors dominate, lowering Bg (25 → about 6) should cut the fragility about fourfold without changing exact behaviour.

```python
import torch
KV, F0W, F0B, F2W, BIN = 'core.prelude.0.attn.kv.weight', 'core.prelude.0.ffn.0.weight', 'core.prelude.0.ffn.0.bias', 'core.prelude.0.ffn.2.weight', 'core.b_in.weight'
def r16(t): return t.to(torch.bfloat16).to(torch.float64)
groups = {'kv': [KV], 'ffn0': [F0W, F0B], 'ffn2': [F2W], 'bin': [BIN]}
CACHE = {}
res7, err = [], None
for sid in list(p13.index)[:45]:
    try:
        df = tu[tu.stay_id == sid].sort_values('hour'); B = make_bundles(df)
        ks = [i for i, (h, ats) in enumerate(B) if 12 <= h < 13 and any(a['code'] in DRUGS for a in ats)]
        if not ks: continue
        k = ks[0]; globals()['WCUR'] = WE
        st = run_tf(B[:k+1], step=stepW, keep=True)
        st = st if isinstance(st, list) else [x for x in st if isinstance(x, list)][0]
        d = st[k]; ats = B[k][1]; dt13 = (13.0 - B[k][0]) * 3600
        CACHE[sid] = (d, ats, dt13, {oc: lastval(df, oc, 13.0) for oc in ('glucose_finger', 'map_arterial')})
        rec = {'sid': sid}
        for drug, oc in (('insulin', 'glucose_finger'), ('norepinephrine', 'map_arterial')):
            if not any(a['code'] == drug for a in ats): continue
            lx, la = CACHE[sid][3][oc]; e2, u2 = enc2(ats)
            for gn, names in [('exact', [])] + list(groups.items()):
                Wd = dict(WE)
                for nm in names: Wd[nm] = r16(WE[nm])
                globals()['WCUR'] = Wd
                r2 = stepW(d['out0'], d['h0'], d['dt'], e2, u2, d['Mv'], d['Ma'])
                rec[f'{gn}_{drug[:3]}'] = float(pred_mean(r2[0], r2[1], dt13, oc, lx, la, nz=20, seed=0))
        res7.append(rec)
    except Exception as e:
        err = repr(e)[:200]
globals()['WCUR'] = WE
R7 = pd.DataFrame(res7).set_index('sid')
print('n', len(R7), 'err', err)
for dg in ('ins', 'nor'):
    print(dg, {gn: round(float((R7[f'{gn}_{dg}'] - R7[f'exact_{dg}']).abs().median()), 3) for gn in groups})
try:
    BE = build_edit(-4.6, 6.1)
    print('build_edit ->', type(BE).__name__, (list(BE.keys())[:8] if isinstance(BE, dict) else len(BE)))
    if isinstance(BE, dict):
        for nm in (F0W, F0B, F2W, BIN):
            if nm in BE:
                ref = B_IN_c1 if nm == BIN else WE[nm]
                ref = torch.as_tensor(np.asarray(ref, dtype=np.float64)) if not isinstance(ref, torch.Tensor) else ref
                print(nm, 'maxdiff vs current(c=1 for b_in):', float((torch.as_tensor(BE[nm]).to(torch.float64) - ref).abs().max()))
except Exception as e:
    print('build_edit fail', repr(e)[:200])
```