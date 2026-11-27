Restoring units 131 and 159 cleaned things up. Rows other than glucose after insulin now change by at most ±0.0007, and predicted means on non-glucose rows move by at most 0.08. The earlier side effects on those rows came from repurposing the NE units.

Insulin-only cost on sel1, a drug-heavy set, per reading:

| a | Overall | glucose_finger after insulin |
|---|---|---|
| −2.2 | 0.0025 | +0.0102 |
| −2.4 | 0.0031 | +0.0123 |
| −2.6 | 0.0037 | +0.0146 |
| −2.8 | 0.0044 | +0.0171 |

On the random f3 set the per-row cost was about 15% higher at −1.9, so −2.6 should cost roughly 0.0037–0.0042 per reading. −2.8 is too close to the 0.005 limit.

Linear scaling from fwd_10 predicts insulin ×2 ≈ −12.8 and ×½ ≈ +5.4 at −2.6, against truth −13.38 and +6.69. I'll save `ins_only(−2.6)` and drop the unchanged transition-table edit. Then I'll run the dose_edit on the same f3 stays as fwd_10 for a direct comparison with the −1.9 run. If the ×2 shift comes out near −12.8, the response is linear in a and I can finish calibrating from the factual cost check. A clearly larger shift would mean it compounds and a smaller gain is enough.

```python
E_sel = ins_only(-2.6)
for k, n in KMAP.items():
    if k == 'TR':
        drop_edit(n); continue
    v = E_sel[k]
    v = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    save_edit(n, v.astype(np.float32))
print('edited now:', edited())
for k, n in KMAP.items():
    if n in edited():
        print(n, 'max|saved - ins_only(-2.6)| %.3g' % float(np.abs(np.asarray(load(n), np.float64) - np.asarray(E_sel[k], np.float64)).max()))
print(sorted(map(int, f3_ids))[:3], len(f3_ids))
json.dump(sorted(map(int, f3_ids)), open('/work/sel_f3.json','w'))
with open('/work/notes.md','a') as fh:
    fh.write("\n- Run 194: ins_only(a)=gated3(a,0) with NE units 131,159 RESTORED to original. sel1 offline cost: a=-2.2 .0025, -2.4 .0031, -2.6 .0037, -2.8 .0044 (6D rows +.010..+.017). Non-glucose rows ~0.\n- NE part DROPPED (noisy readout R2 .19, gate leaks, cost ~.024 per NE-row, cannot fix x2 OOD). Saved ins_only(-2.6) (TR edit dropped). Next: dose_edit both on f3 to compare with fwd_10 (-1.9/3 -> -9.85).\n")
```