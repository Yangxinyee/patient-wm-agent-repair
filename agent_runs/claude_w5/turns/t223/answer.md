Both quiet directions stay quiet on dose-scaled bundles too (q·E sd about 2.4e-4, maximum about 2e-3). E has 42 directions with eigenvalue below 1e-6, so the carrier channel is solid.

The weak point is the target direction. Per-bundle belief directions agree only moderately with their average (median cosine 0.74 for glucose, 0.81 for MAP). The average has a much smaller norm (0.77) than a typical per-bundle direction (2.3), so a fixed direction may move the model's beliefs inconsistently or affect other variables.

The real test is behavioral. Next I check my local head against the stored rows: view0 should match H's v0, and predicted means should match the forward tool's factual output. Then I inject α·v̄_I or α·v̄_N into h_new at drug bundles and measure the shift of the next glucose_finger and MAP predicted means. A fixed direction is acceptable if glucose moves consistently under v̄_I while MAP barely moves, and vice versa. Large spread or cross-effects would mean I need a better direction, e.g. one fit to head responses.

```python
import pandas as pd
f3 = pd.read_csv('/work/fwd_3/factual_original.csv')
Hidx = {}
for i, row in enumerate(H): Hidx.setdefault((int(row['sid']), int(row['k'])), []).append(i)
def head_mean_rows(out_k, y_k, dt, rows_i):
    v0n = view0(out_k, y_k, dt)
    V = v0n.reshape(1, -1).repeat(len(rows_i), 1)
    p = probs_from_v0(V, fixed[rows_i], TRr[rows_i], nz=0)
    p = p if isinstance(p, torch.Tensor) else torch.as_tensor(p)
    return (p[:, 1:] * cen[rows_i]).sum(1) / p[:, 1:].sum(1)
# verify on reading rows right after drug bundles
chk, rows_after = [], []
for r in DB[:400]:
    s, k = r['sid'], r['k']; rows_i = Hidx.get((s, k+1), [])
    if not rows_i: continue
    d = steps_all[s][k]; o = d['res']
    dt = (bundles_B[s][k+1][0] - bundles_B[s][k][0]) * 3600.0
    v0n = view0(o[0], o[1], dt)
    chk.append(float((v0n - H[rows_i[0]]['v0']).abs().max()))
    rows_after.append((r, rows_i, dt))
print('view0 vs H.v0 max abs diff', max(chk), 'n', len(chk))
pm_loc, pm_fwd = [], []
for r, rows_i, dt in rows_after[:150]:
    o = steps_all[r['sid']][r['k']]['res']; m = head_mean_rows(o[0], o[1], dt, rows_i)
    for j, ri in enumerate(rows_i):
        hh = H[ri]; sel_ = f3[(f3.stay_id == hh['sid']) & (np.isclose(f3.hour, hh['hour'])) & (f3.event == hh['code'])]
        if len(sel_) == 1: pm_loc.append(float(m[j])); pm_fwd.append(float(sel_.predicted_mean.iloc[0]))
pm_loc, pm_fwd = np.array(pm_loc), np.array(pm_fwd)
print('local(nz=0) vs fwd predicted mean: n', len(pm_loc), 'corr', np.corrcoef(pm_loc, pm_fwd)[0,1].round(5), 'mean abs diff', np.abs(pm_loc-pm_fwd).mean().round(3))
# head response to fixed-direction injections at drug bundles
alpha = 10.0; resp = {('I','g'):[], ('I','m'):[], ('N','g'):[], ('N','m'):[]}
for r, rows_i, dt in rows_after[:300]:
    d = steps_all[r['sid']][r['k']]; o = d['res']; base_m = head_mean_rows(o[0], o[1], dt, rows_i)
    codes = [H[ri]['code'] for ri in rows_i]
    for tg, vb in (('I', vbar_I), ('N', vbar_N)):
        dh = torch.from_numpy(np.tile(alpha*vb, (4,1)))
        o2 = mystep(d, dhn=dh); m2 = head_mean_rows(o2[0], o2[1], dt, rows_i)
        for j, c in enumerate(codes):
            if c == 'glucose_finger': resp[(tg,'g')].append(float(m2[j]-base_m[j]))
            if c == 'map_arterial': resp[(tg,'m')].append(float(m2[j]-base_m[j]))
for kk, vv in resp.items():
    vv = np.array(vv); print('inject', kk[0], '->', 'glucose' if kk[1]=='g' else 'MAP', 'n', len(vv), 'mean', vv.mean().round(3), 'sd', vv.std().round(3), 'p10/p90', np.percentile(vv,[10,90]).round(3))
```