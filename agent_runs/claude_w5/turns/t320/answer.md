Run 290 rules out merged bundles: merging makes the match much worse (recorded-run correlation 0.7–0.85, median error 14–16). The tool keeps 12:00 readings and 12:05 drugs separate.

With separate bundles, my local recorded-dose h13 prediction for the edited model tracks the tool's closely (correlation 0.98). It is still off by a median 3.3 mg/dL for glucose and 1.8 mmHg for MAP, against 0.37 for the original model. So the tool encodes the hour-12 drug bundle differently even with recorded doses, and only my dose-reading edit notices.

The recorded-dose prediction is a direct, noise-free test of that encoding, because it involves no dose scaling. This run tries all four rate/dose bin-side combinations for the hour-12 drug bundle (LL, RR, RL, LR). For each it compares my local recorded-dose prediction with the tool's (median absolute difference) and also the per-stay shifts.

A combination that brings the median difference down to about 0.4, the original model's level, is the tool's encoding. I would then refit the decoders to read correctly under both it and the factual left-bin encoding. I also print `mystep`'s local variable names, to see whether it can record the prelude internals I would need to refit.

```python
print('mystep vars:', mystep.__code__.co_varnames[:40])
combos = {'LL': fb_orig, 'RR': mk_fb2('right','right'), 'RL': mk_fb2('right','left'), 'LR': mk_fb2('left','right')}
res5, err = [], None
for sid in p13.index:
    try:
        df = tu[tu.stay_id == sid].sort_values('hour'); B = make_bundles(df)
        ks = [i for i, (h, ats) in enumerate(B) if 12 <= h < 13 and any(a['code'] in DRUGS for a in ats)]
        if not ks: continue
        k = ks[0]; globals()['fb_'] = fb_orig
        st = run_tf(B[:k+1], step=stepW, keep=True)
        st = st if isinstance(st, list) else [x for x in st if isinstance(x, list)][0]
        d = st[k]; ats = B[k][1]; rec = {'sid': sid}; dt13 = (13.0 - B[k][0]) * 3600
        for drug, oc in (('insulin', 'glucose_finger'), ('norepinephrine', 'map_arterial')):
            if not any(a['code'] == drug for a in ats): continue
            lx, la = lastval(df, oc, 13.0)
            for cn, fbf in combos.items():
                def pm(at):
                    globals()['fb_'] = fbf; e2, u2 = enc2(at); globals()['fb_'] = fb_orig
                    r2 = stepW(d['out0'], d['h0'], d['dt'], e2, u2, d['Mv'], d['Ma'])
                    return float(pred_mean(r2[0], r2[1], dt13, oc, lx, la, nz=20, seed=0))
                b0 = pm(ats); rec[f'{cn}_{drug[:3]}_base'] = b0
                for f, nm in ((2.0, 'x2'), (0.5, 'h')):
                    rec[f'{cn}_{drug[:3]}_{nm}'] = pm(scl(ats, drug, f, 1)) - b0
        res5.append(rec)
    except Exception as e:
        err = repr(e)[:200]
globals()['fb_'] = fb_orig
print('n', len(res5), 'err', err)
R5 = pd.DataFrame(res5).set_index('sid')
lines = []
for cn in combos:
    s = cn + ':'
    for tc, lc, fc in (('insulin_x2','ins_x2','insulin_x2'), ('insulin_half','ins_h',None), ('ne_x2','nor_x2','ne_x2'), ('ne_half','nor_h',None)):
        j = pd.concat([p13[tc], R5[f'{cn}_{lc}']], axis=1).dropna()
        s += f" {tc} c={j.corr().iloc[0,1]:.2f} m={j.iloc[:,1].mean():.2f}"
        if fc:
            jb = pd.concat([tf13[fc], R5[f'{cn}_{lc[:3]}_base']], axis=1).dropna()
            s += f" [base mad={float((jb.iloc[:,0]-jb.iloc[:,1]).abs().median()):.2f}]"
    lines.append(s)
print('tool means: ins_x2 -4.62 ins_h -0.04 ne_x2 1.20 ne_h -0.50')
print('\n'.join(lines))
with open('/work/notes.md', 'a') as f:
    f.write("- Run 290: merged bundles WORSE (base mad 14-16) -> tool uses separate bundles; S base mad 3.26 glc / 1.82 MAP (orig model 0.37).\n- Run 291 bin-combo test at 12:05 (rate_side,dose_side):\n  " + "\n  ".join(lines) + "\n")
```