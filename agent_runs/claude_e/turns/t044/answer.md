The safety filter stopped my last two replies, so I will keep outputs short and focus on the technical work.

**What I now know.** Run 40 gave me the full tensor list. It also showed that my earlier helper functions have no recoverable source. Two edit routes look possible:
- **Head-level edit.** Correct the glucose (and MAP) value head using a dose readout taken from the model's state. This needs only the head computations, which are small, plus the per-bundle states from the tool. It does not need the full recurrent core.
- **Full local re-implementation.** This would let me test edits offline, but it is expensive and still mismatched (error 0.257).

**What the head-level edit must do.** It should add the missing dose effect only for the part of the dose that deviates from what the protocol would give for the last glucose reading. Adding it for the raw dose would double-count the average effect the model already absorbed and would hurt factual NLL.

**Next step.** Check that this is feasible: does the saved state (fwd_2) linearly encode the insulin dose beyond what the glucose readings explain?
- A high within-tier R² for dose from the state would support the head-level route.
- A low R² would push me back to the re-implementation.

First I will look at the structure of the saved states file, printing shapes only.

```python
print(list(S2.keys()))
for kk in S2.keys():
    a=S2[kk]; print(kk, a.shape, a.dtype, (a[:3] if a.ndim==1 else ''))
print("t2 cols:", list(t2.columns)); print(t2.head(3).to_string()[:600])
print("m:", m[:10], m.dtype, m.sum() if m.dtype==bool else '')
```