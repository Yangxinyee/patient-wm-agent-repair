The readout is not overfitted. In-sample correlation with the dose deviation is −0.67, against −0.72 out-of-sample on sel1, so more training states would not reduce its noise.

The gain figures also show why the cost is high. At a = −2.6, the predicted mean moves −2.93 per u/h of deviation on top of the model's own response. Its one-step response is then about −4, twice the true −2.0. The model forgets doses after one step (its lag-2 to lag-6 coefficients are near zero, the truth's are not), so the lag-1 correction must overshoot to give the right sustained effect. In factual data that overshoot is about 30% of the cost; readout noise is the other 70%. A head edit cannot fix either, so the remaining choice is the gain.

I'll take a = −2.5, projected at about 0.0039 per reading and insulin ×2 at about 95% of truth. I'll measure it on all four tuning batches to get the cost and its spread before deciding. First, save it and run the factual on f3. The current edit sits about 2% short of the true ×2 change but too close to the 0.005 limit; −2.5 trades about 4% of accuracy for margin.

```python
E_sel = ins_only(-2.5)
for k, n in KMAP.items():
    if k == 'TR': continue
    v = E_sel[k]; v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    save_edit(n, v.astype(np.float32))
print('edited:', edited(), '| check', float(np.abs(np.asarray(load(KMAP['W2']),np.float64) - np.asarray(E_sel['W2'],np.float64)).max()))
o_c = head_run(X, RR, E_sel); R1['dn'] = np.asarray(o_c[1], float)[m1] - nll_orig1
print('sel1 offline cost a=-2.5: %.4f' % R1.dn.mean())
with open('/work/notes.md','a') as fh:
    fh.write("\n- fwd_13: factual both tu0 (2020000100..199) with ins_only(-2.6): +0.00427/reading (gf +.0126/row, gl +.0061, ma/mc ~0). Too close to .005.\n- Runs 200-206: persistence check on random stays: within-tier resid slope on prev gf only +2.2/50mg (se .78) -> dropped. Readout not overfit (in-sample corr .67 vs sel1 .72). Edited one-step slope ~-4 vs truth -2 (model lacks dose memory beyond lag1; lag-1 overshoot needed for sustained effect).\n- Saved ins_only(-2.5). Plan: factual both on all 4 tuning batches at -2.5; dose_edit at -2.5.\n")
```