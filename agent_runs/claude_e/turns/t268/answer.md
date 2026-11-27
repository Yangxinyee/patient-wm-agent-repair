On tu0, −2.8 costs +0.00492 per reading against +0.00428 at −2.6. The real cost factor for 0.2 of gain is therefore 1.15, a little below the 1.20 I assumed. That puts the pooled mean cost at about 0.0031 for −2.8 and 0.0029 for −2.7.

But tu0 is a real batch of 100 stays from the same ICU, and at −2.8 it sits right at the 0.005 limit. With an evaluation set of a few hundred stays, the risk at −2.8 would be roughly 1–1.5%, against about 0.4% at −2.7. The readout gain from −2.7 to −2.8 is only about 4 points of x2 ratio. A failed factual check would void the whole edit, so I'll take the safer step.

I will save `ins_only(-2.7)` and record the measurements. The expected pooled readouts are insulin x2 about 0.95 of truth (≈ −12.9 against −13.55) and x1/2 about 0.90 (≈ +6.1 against +6.8). Then I'll confirm its factual cost on tu0; scaling predicts about 0.0046.

```python
E27=ins_only(-2.7)
for k in ['W0','b0']:
    assert float(np.abs(to_np(E27[k])-load(KMAP[k]).numpy().astype(np.float64)).max())<1e-4
assert float(np.abs(to_np(E27['TR'])-to_np(ORIG['TR'])).max())==0.0
save_edit(KMAP['W2'], torch.tensor(to_np(E27['W2']),dtype=torch.float32))
print('W2 saved check',float(np.abs(load(KMAP['W2']).numpy().astype(np.float64)-to_np(E27['W2'])).max()), edited())
with open('/work/notes.md','a') as f:
    f.write("""- Pooled tuning (400 stays) at -2.6: ins x2 -12.36 vs truth -13.55 (ratio .912), half 5.84 vs 6.77 (.862); NE -0.23/-0.56 vs 4.53/-2.26. f3 was a lucky batch.
- fwd_25: factual tu0 at -2.8 (hash c4297930e4): +0.00492 (vs .00428 at -2.6 -> factor 1.15 per 0.2). Scaled bootstrap: -2.7 mean ~.0029, P(>.005) N=400 ~.5%; -2.8 ~1.3%.
- Run 240: chose and saved ins_only(-2.7) (W2 only differs; W0,b0 as before; TR original). Expected pooled: x2 ratio ~.95, half ~.90.
""")
```