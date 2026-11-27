Row 0 is all zeros in every categorical, bin, ekind and text table, so index 0 means "absent". Several rows still have norms near 0.28, which is untrained initial noise: kq rows 1–3, the second row of action, route, status, discharged_to and text, and unit/rate_unit row 1. They are never used in this data. So the only real embedding terms are:
- code, kind and ekind,
- kq[0], which every atom gets,
- unit, from rows 2–5,
- rate_unit, from rows 2–3,
- the bins and the numeric term.

This matches what impl.py already does. The config confirms 4 attention heads, ff_mult 4, 4 slots and a memory of 128.

My z mismatch must therefore come from something else, for example:
- the activation function (GELU or SiLU),
- the dt scaling in gap,
- how END atoms are encoded,
- or a misaligned comparison.

Next I will inspect the saved states file: its keys, shapes and metadata. Then I can line up bundles exactly and test the input-side pieces one at a time. If the metadata shows something like an extra bundle at the start of each stay, a misalignment would explain the errors directly.

```python
S_ = np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
for k in S_.files: print(k, S_[k].shape, S_[k].dtype, S_[k][:3] if S_[k].ndim==1 else '')
print(os.listdir('/work/fwd_2'))
st0 = S_['stay'] if 'stay' in S_.files else None
```