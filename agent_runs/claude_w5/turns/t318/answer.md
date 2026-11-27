Run 288 changes the picture. The tool's per-stay h13 shifts have a spread 2–5 times larger than mine (insulin ×½ SD 6.3 vs about 2; norepinephrine ×½ 3.3 vs 0.6). Yet for three of the four edits the means agree within about 0.8, and within 0.1 for norepinephrine, under every bin convention. Bin conventions improve the insulin ×½ match only a little (local mean 1.29–1.87 vs tool −0.04).

Extra spread that averages out points to the sampling of z, not the encoding. My notes record that the tool samples z with the second half of the prior read as log-std. My `pred_mean` calls `probs_from_v0` with its default `lv='logvar'`, which gives a much narrower z spread. A wider z spread changes how the nonlinear head turns my state injection into a predicted mean, so it can create the ×2/×½ asymmetry on its own.

This run re-runs the local h13 comparison with z drawn the tool's way (log-std, 64 draws). If local means then match the tool, including insulin ×½ near 0, and the spreads grow toward the tool's, the asymmetry comes from the head's nonlinear response under wide z. I would then fix the injection direction rather than the encoding. If not, the cause lies elsewhere in the rollout.

```python
_pf0 = globals().get('_pf0', probs_from_v0)
def _pf_ls(*a, **k):
    a = list(a)
    if len(a) >= 5: a[4] = 'logstd'
    else: k['lv'] = 'logstd'
    return _pf0(*a, **k)
globals()['probs_from_v0'] = _pf_ls
res3, err = [], None
for sid in p13.index:
    try:
        df = tu[tu.stay_id == sid].sort_values('hour'); B = make_bundles(df)
        ks = [i for i, (h, ats) in enumerate(B) if 12 <= h < 13 and any(a['code'] in DRUGS for a in ats)]
        if not ks: continue
        k = ks[0]; st = run_tf(B[:k+1], step=stepW, keep=True)
        st = st if isinstance(st, list) else [x for x in st if isinstance(x, list)][0]
        d = st[k]; ats = B[k][1]; rec = {'sid': sid}; dt13 = (13.0 - B[k][0]) * 3600
        for drug, oc in (('insulin', 'glucose_finger'), ('norepinephrine', 'map_arterial')):
            if not any(a['code'] == drug for a in ats): continue
            lx, la = lastval(df, oc, 13.0)
            def pm(at):
                e2, u2 = enc2(at)
                r2 = stepW(d['out0'], d['h0'], d['dt'], e2, u2, d['Mv'], d['Ma'])
                return float(pred_mean(r2[0], r2[1], dt13, oc, lx, la, nz=64, seed=0))
            b0 = pm(ats)
            for f, nm in ((2.0, 'x2'), (0.5, 'h')):
                rec[f'{drug[:3]}_{nm}'] = pm(scl(ats, drug, f, 1)) - b0
        res3.append(rec)
    except Exception as e:
        err = repr(e)[:200]
globals()['probs_from_v0'] = _pf0
print('n', len(res3), 'err', err)
Rl3 = pd.DataFrame(res3).set_index('sid')
lines = []
for tc, lc in (('insulin_x2','ins_x2'), ('insulin_half','ins_h'), ('ne_x2','nor_x2'), ('ne_half','nor_h')):
    j = pd.concat([p13[tc], Rl3[lc]], axis=1).dropna()
    lines.append(f"{tc} logstd: corr={j.corr().iloc[0,1]:.3f} tool={j.iloc[:,0].mean():.2f} local={j.iloc[:,1].mean():.2f} sd_t={j.iloc[:,0].std():.2f} sd_l={j.iloc[:,1].std():.2f}")
print('\n'.join(lines))
with open('/work/notes.md', 'a') as f:
    f.write("- Run 288: bin side barely matters (RA ins_h local 1.29 vs tool -0.04); tool per-stay SD 2-5x local -> suspect z sampling (logstd).\n- Run 289 (pred_mean with lv=logstd, nz=64):\n  " + "\n  ".join(lines) + "\n")
```