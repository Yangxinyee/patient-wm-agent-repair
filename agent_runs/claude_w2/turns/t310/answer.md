Candidate 2 makes things worse:
- **Insulin:** barely moves (x2 −2.43).
- **Norepinephrine:** becomes lopsided (x2 +1.56, x1/2 −2.73).
- **MAP fit:** worse.

So the bin path is not the insulin bottleneck. The model ignores the insulin dose downstream, consistent with the head analysis, and I'll revert the bin edits.

What remains, with 8 runs:
1. **Norepinephrine:** tune the numeric factor, which drives norepinephrine cleanly and improved the fit. Extrapolating from k = 1 to 3 (x2 +0.79 per unit k, x1/2 −0.37) suggests k ≈ 5 for the targets +4.7 and −2.4.
2. **Insulin:** add the small, cheap head readout (μ = 0.3, gain 0.5, gated to glucose codes). Locally it costs about +0.002 nats on glucose rows and adds about −0.4 mg/dL per hour of open-loop response, which the model's own dynamics should accumulate over the rollout.

I'll save both and verify the saved tensors. Then the forward tool will check, per reading type, factual NLL (glucose for the head edit, MAP for the numeric) and the four readouts.

```python
for n in ['embed.bins.rate.weight', 'embed.bins.dose.weight']: drop_edit(n)
Wnum = np.asarray(load_original('embed.numeric.weight'), dtype=np.float32).copy(); Wnum[:, [2, 4]] *= 5.0
save_edit('embed.numeric.weight', torch.as_tensor(Wnum))
w_ins = np.linalg.solve(Q_Gd + 0.3*Q_Gf + np.diag(np.r_[1e-3*np.ones(192), 1e-6]), Q_hd + 0.3*Q_hf)
Eh = Q_build(w_ins, 0.5)
names_h = ['attr_heads.value_bin.0.weight', 'attr_heads.value_bin.0.bias', 'attr_heads.value_bin.2.weight', 'attr_heads.value_bin.2.bias', 'embed_transition.value_bin.weight']
for n, arr in zip(names_h, Eh):
    assert tuple(arr.shape) == tuple(SHAPES[n]); save_edit(n, torch.as_tensor(arr.astype(np.float32)))
print("edited:", edited(), "| max reload err:", max(float(np.abs(np.asarray(load(n), dtype=np.float64) - a).max()) for n, a in zip(names_h, Eh)))
np.save('/work/w_ins_head.npy', w_ins)
with open('/work/notes.md', 'a') as f:
    f.write("- cand2 (bins x3) BAD: factual +0.0055 (MAP worse), ins x2 -2.43, NE x2 1.56/x.5 -2.73 -> reverted.\n"
            "- cand3: numeric x5 + head gated readout unit 109 (Q_build, mu=.3 resid-target, lam=.5, kap4 G20, bounded tilt clip60/Vb201), w in /work/w_ins_head.npy.\n")
```