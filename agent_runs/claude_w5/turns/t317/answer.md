Run 287 rules out the end-atom idea. Scaling the kind-2 end atoms changes nothing, so the encoder ignores their rate.

The run also shows my local h13 shifts do not match the tool per stay (correlation 0.4–0.6), although the means for norepinephrine and insulin ×2 are close. The tool's per-path noise is small (shift_se about 0.03), so this mismatch is systematic, not sampling noise. Insulin ×½ is the outlier in the means: local +1.87 against tool −0.04.

The rate data from run 285 point at the bin convention. Recorded rates sit exactly on repeated bin edges, so left versus right binning moves a rate 1–4 bins.

This run reuses the same local pipeline with right-edge binning (`mk_fb2('right','right')`, which `avec` reads through the global `fb_`). I compare three variants against the tool per stay:
- left everywhere;
- right for the hour-12 bundle in all runs, including the recorded one;
- right for the edited runs only.

If one right-bin variant brings the per-stay correlation near 0.9+ and reproduces insulin ×½ ≈ 0, the cause is confirmed. The fix would then be to make the decoders read the same rate under either bin convention.

```python
RR = mk_fb2('right', 'right')
FBL = fb_orig
res2, err = [], None
for sid in p13.index:
    try:
        df = tu[tu.stay_id == sid].sort_values('hour'); B = make_bundles(df)
        ks = [i for i, (h, ats) in enumerate(B) if 12 <= h < 13 and any(a['code'] in DRUGS for a in ats)]
        if not ks: continue
        k = ks[0]
        globals()['fb_'] = FBL
        st = run_tf(B[:k+1], step=stepW, keep=True)
        st = st if isinstance(st, list) else [x for x in st if isinstance(x, list)][0]
        d = st[k]; ats = B[k][1]; rec = {'sid': sid}; dt13 = (13.0 - B[k][0]) * 3600
        for drug, oc in (('insulin', 'glucose_finger'), ('norepinephrine', 'map_arterial')):
            if not any(a['code'] == drug for a in ats): continue
            lx, la = lastval(df, oc, 13.0)
            def pm(at, fbf):
                globals()['fb_'] = fbf
                e2, u2 = enc2(at)
                globals()['fb_'] = FBL
                r2 = stepW(d['out0'], d['h0'], d['dt'], e2, u2, d['Mv'], d['Ma'])
                return float(pred_mean(r2[0], r2[1], dt13, oc, lx, la, nz=20, seed=0))
            bL, bR = pm(ats, FBL), pm(ats, RR)
            for f, nm in ((2.0, 'x2'), (0.5, 'h')):
                sL, sR = pm(scl(ats, drug, f, 1), FBL), pm(scl(ats, drug, f, 1), RR)
                rec[f'{drug[:3]}_{nm}_LL'] = sL - bL; rec[f'{drug[:3]}_{nm}_RA'] = sR - bR; rec[f'{drug[:3]}_{nm}_RE'] = sR - bL
        res2.append(rec)
    except Exception as e:
        err = repr(e)[:200]
globals()['fb_'] = FBL
print('n', len(res2), 'err', err)
Rl2 = pd.DataFrame(res2).set_index('sid')
lines = []
for tc, lc in (('insulin_x2','ins_x2'), ('insulin_half','ins_h'), ('ne_x2','nor_x2'), ('ne_half','nor_h')):
    for v in ('LL', 'RA', 'RE'):
        j = pd.concat([p13[tc], Rl2[f'{lc}_{v}']], axis=1).dropna()
        lines.append(f"{tc} {v}: corr={j.corr().iloc[0,1]:.3f} tool={j.iloc[:,0].mean():.2f} local={j.iloc[:,1].mean():.2f} sd_t={j.iloc[:,0].std():.2f} sd_l={j.iloc[:,1].std():.2f}")
print('\n'.join(lines))
with open('/work/notes.md', 'a') as f:
    f.write("- Run 287 result: END-atom scaling has NO effect on encoding (V0==V1); local-vs-tool per-stay h13 corr only .4-.6.\n- Run 288 bin-side test (LL/RA=right all runs at 12:05/RE=right edited only):\n  " + "\n  ".join(lines) + "\n")
```