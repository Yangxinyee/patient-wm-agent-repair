The noise between two 4-sample runs is MAE 1.47, so a 4-sample run should differ from the expectation by only about 1.0. The 64-sample local head still differs from the tool by 2.4 (correlation 0.999), so a small systematic error remains. Next I'll check whether the best type index differs by code and whether the age units should differ by code.

```python
zh64 = zh_samples(64, 3)
best = {}
for c in [5, 6, 7, 10]:
    sc = []
    for t in range(15):
        for au in [60, 3600]:
            m_, _ = head_mean2(type_i=t, au=au, zh_=zh64); sc.append((np.abs(m_-tool)[cv == c].mean().round(3), t, au))
    sc.sort(); best[c] = sc[:3]
print(best)
```