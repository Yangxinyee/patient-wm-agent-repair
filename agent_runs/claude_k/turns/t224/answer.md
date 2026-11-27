Run 207 shows the kernel still holds what I need for the z-direction analysis:
- the original-model states for the 99 train stays (`S2`), plus states for 300 more stays (`XS`, `zS`, with a drug-bundle flag `isdrug`);
- per-reading arrays for the 99 stays: values `vals99`, codes, predicted means `pmall`, and the reading-to-state index frame `R`.

Next I'll confirm how these line up: the columns of `R`, the arrays in `S2`, and what `fw`/`de` hold. Then I can pair each reading's surprise (observed minus predicted) with the z posterior of the bundle that contains it.

The approach depends on two things. The state row after a reading's bundle must be identifiable, as `R.k + 1` or similar. The z posterior minus its prior must then correlate clearly with the glucose and MAP surprise. If z barely tracks the surprise, this route fails and I fall back to the prelude attention path.

```python
print('R', list(R.columns)[:20], len(R))
print('S2', [(f, S2[f].shape) for f in S2.files])
for nmx in ['fw','de','tu']:
    v=globals()[nmx]; print(nmx, v.shape, list(v.columns)[:14])
print(R.head(3).to_string()[:600])
```