Run 289 rules out z sampling: log-std draws change nothing. My local single-step reproduction still matches the tool per stay only weakly. The tool's per-stay spread is 2–5 times mine, even for the edits whose means agree.

My notes also record that the tool's recorded-dose h13 prediction for the edited model differs from my local value by a median 2.1 mg/dL, while the original model matches. So in the dose_edit rollout the edited components (the drug-bundle gate and decoders) see something different at hour 12 from what the factual pass sees.

My new hypothesis is about how the rollout groups events. It may group each hour's readings and drug atoms into one bundle from hour 12 on, instead of keeping 12:00 readings and 12:05 drugs as separate bundles. A merged bundle would half-open my drug gate, which is fitted only on drug-only bundles. A half-open GELU gate responds nonlinearly to the decoded dose, which would give the common-mode, wrong-sign ×½ effect and the large per-stay scatter.

This run compares three local layouts against the tool's per-stay h13 values (recorded-run prediction and the four shifts):
- separate bundles (S);
- 12:00 readings and 12:05 drugs merged at 12:00 (M0);
- the same merge at 12:05 (M5).

If a merged layout brings the per-stay correlation to about 0.9 or more, the hypothesis is confirmed and the fix is to refit the gate and decoders on merged bundles too.

```python
tf13 = dd[dd.hour == 13].pivot_table(index='stay_id', columns='edit', values='factual')
res4, err = [], None
def shifts_for(Bx, tt, df):
    st = run_tf(Bx, step=stepW, keep=True)
    st = st if isinstance(st, list) else [x for x in st if isinstance(x, list)][0]
    d = st[-1]; ats = Bx[-1][1]; out = {}; dt13 = (13.0 - tt) * 3600
    for drug, oc in (('insulin', 'glucose_finger'), ('norepinephrine', 'map_arterial')):
        if not any(a['code'] == drug for a in ats): continue
        lx, la = lastval(df, oc, 13.0)
        def pm(at):
            e2, u2 = enc2(at)
            r2 = stepW(d['out0'], d['h0'], d['dt'], e2, u2, d['Mv'], d['Ma'])
            return float(pred_mean(r2[0], r2[1], dt13, oc, lx, la, nz=20, seed=0))
        b0 = pm(ats); out[f'{drug[:3]}_base'] = b0
        for f, nm in ((2.0, 'x2'), (0.5, 'h')):
            out[f'{drug[:3]}_{nm}'] = pm(scl(ats, drug, f, 1)) - b0
    return out
for sid in p13.index:
    try:
        df = tu[tu.stay_id == sid].sort_values('hour'); B = make_bundles(df)
        ks = [i for i, (h, ats) in enumerate(B) if 12 <= h < 13 and any(a['code'] in DRUGS for a in ats)]
        if not ks: continue
        k = ks[0]; rec = {'sid': sid}
        for tag in ('S', 'M0', 'M5'):
            if tag == 'S': Bx, tt = B[:k+1], B[k][0]
            else:
                hp, ap = B[k-1]
                if not (12.0 <= hp < 12.08): Bx, tt = B[:k+1], B[k][0]
                else:
                    tt = 12.0 if tag == 'M0' else B[k][0]
                    Bx = B[:k-1] + [(tt, list(ap) + list(B[k][1]))]
            for kk, v in shifts_for(Bx, tt, df).items(): rec[f'{tag}_{kk}'] = v
        res4.append(rec)
    except Exception as e:
        err = repr(e)[:200]
print('n', len(res4), 'err', err)
R4 = pd.DataFrame(res4).set_index('sid')
lines = []
for tag in ('S', 'M0', 'M5'):
    for tc, lc, fc in (('insulin_x2','ins_x2','insulin_x2'), ('insulin_half','ins_h',None), ('ne_x2','nor_x2','ne_x2'), ('ne_half','nor_h',None)):
        j = pd.concat([p13[tc], R4[f'{tag}_{lc}']], axis=1).dropna()
        s = f"{tag} {tc}: corr={j.corr().iloc[0,1]:.2f} tool={j.iloc[:,0].mean():.2f} loc={j.iloc[:,1].mean():.2f} sdl={j.iloc[:,1].std():.2f}"
        if fc:
            jb = pd.concat([tf13[fc], R4[f'{tag}_{lc[:3]}_base']], axis=1).dropna()
            s += f" | base corr={jb.corr().iloc[0,1]:.3f} medabs={float((jb.iloc[:,0]-jb.iloc[:,1]).abs().median()):.2f}"
        lines.append(s)
print('\n'.join(lines))
with open('/work/notes.md', 'a') as f:
    f.write("- Run 289: logstd z sampling changes nothing. Run 290 bundle-merge test (S separate / M0 merged 12:00 readings+12:05 drugs at 12:00 / M5 at 12:05):\n  " + "\n  ".join(lines) + "\n")
```