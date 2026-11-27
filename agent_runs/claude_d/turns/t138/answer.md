The family register didn't help either (still 7.32), so the register follows the documented per-code rule. Run 116 only compared two candidate type indices, 1 and 3. Type 1 may just be the less wrong of the two, and q depends on the type embedding for every reading. A cheap check is to score all 15 `type_emb` rows. If one of them brings glucose RMS near the 2 mg/dL sampling floor, the type index was the bug. If none does, I'll stop pursuing an exact reproduction and design an edit that tolerates approximate states.

```python
for t_ in range(15):
    r_ = predict(FR,typ=t_,gap_scale=10.0)
    e_ = r_[0]-FR['predicted_mean'].values; gl_ = FR['event'].str.startswith('glucose').values
    print(t_, np.sqrt((e_[gl_]**2).mean()).round(2), np.sqrt((e_[~gl_]**2).mean()).round(2), end=' | ')
```