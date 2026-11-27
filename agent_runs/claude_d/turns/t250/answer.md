The lam = 1 design behaves as intended:
- Rounding noise is now 3–7% of the signal.
- The realized mean shift per unit of decoded surprise matches C (2.06 at C = 2).
- The bf16 bias is negligible, and cost rises steadily with C.

At C = 2 the norepinephrine channel adds about +0.0018 nats in-sample and +0.0012 on held-out stays, on top of E3b. Together with E3b's held-out +0.0016, the total is about +0.003, inside the 0.005 budget.

Hosting the channel on units 124 and 142 slightly improves the no-norepinephrine MAP rows, by about −0.002 to −0.003.

Next I will save E3b plus the norepinephrine channel (C = 2, lam = 1, units 124 and 142, gates and decoder fitted on all 400 stays) as the current edit. I will also store it in /work/E4.npz and update the notebook. After that I will run dose_edit on sel1, the same stays as forward passes 8 and 10, to see how a lag-1 gain of 2 translates into the rollout readouts.

```python
Wf = build_ne2(2.0, wN, bN, kN, wf+wp, bf+bp, lam=1.0)
for nm_, x in zip(['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias'], Wf):
    save_edit(nm_, torch.tensor(np.asarray(x, dtype=np.float32)))
np.savez('/work/E4.npz', *Wf)
chk = [float(np.abs(np.asarray(load(n))-x).max()) for n,x in zip(['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias'], Wf)]
print(edited(), chk)
with open('/work/notes.md','a') as f:
    f.write("""- Runs 216-223 (NE): NE info at MAP_a rows weak (D_ne_1 R2 full .32, wc .18; prev-bundle-NE no better). Decoder must be fit with zero targets on off-gate rows (om=.1) else extrapolates (dAll +.37!). cvR2 .22 k .194.
  Unbiased-decoder noise var = C^2 VarD/R2 -> affordable C~2 only. bf16 rounding noise of gate rows matters: use lam=1 (noise/sig .07).
  NE channel: units (124,142) (cheap: ablation cost <0), AND gate fam(map_arterial)+pres(ne_1>0) ridge a=1 z-aug, kap 6, tilt tA=(cenA-75)/10, Vref 20.7.
  Cost vs E3b: C1.5 +.0003, C2 +.0018 (heldout +.0012), C2.5 +.0032. Slope ~C, bf16 bias ~0.
- E4 = E3b + NE C=2 lam1 saved (/work/E4.npz, current edit). Next: dose_edit both sel1.
""")
```