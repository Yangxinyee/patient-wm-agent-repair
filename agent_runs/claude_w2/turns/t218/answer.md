The saved artifacts show where runs 149–206 went. `make_W_chan` wrote a rank-1 update into the out-projection columns of attention head 2 of `core.prelude.0` (cols 96:144). wI_head2.npy is a 48-dim readout of that head, and transfer_T.npy a 192×192 transfer map. So I was building a dose channel inside the core, then switched to the oracle tests in the value head. None of it was ever saved as an edit, so I am starting the edit itself fresh.

The design follows from my findings.
- **Doses are randomized given the last glucose (insulin) or the last rate and MAP band (norepinephrine).** A perfect maximum-likelihood model's one-step conditionals would then also be the interventional ones.
- **The model's specific error:** it under-uses the dose itself and absorbs the typical policy dose effect into its glucose-level dynamics (AR sum 0.87 vs 0.98).
- **The correct factual-neutral fix:** add the missing effect of the dose *deviation* from policy at lags 1–3. This is Σ_k a_k·d_{t−k} + Σ_k c_k·p(g_{t−k}), with c ≈ −a if the absorption story is right.
- **How to implement it:** one repurposed hidden unit per outcome in `attr_heads.value_bin`, gated on code. Its input is a decoder trained on randomly perturbed doses, so it responds causally. Its output is a logit tilt along the bin centres.

Step 1 is to get a, c from the model's own teacher-forced residuals (tf2k_orig.parquet, 2000 train stays) regressed on dose and policy lags. Before that I need the columns of that file and of the prepared grid gg.

```python
tf2=pd.read_parquet('/work/tf2k_orig.parquet')
print(tf2.shape, tf2.columns.tolist()); print(tf2.head(3))
print('gg cols:', gg.columns.tolist())
print('tr cols:', tr.columns.tolist()); print(tr.head(3))
print('CODE', CODE); print('MEAS',MEAS)
print({c:np.round(CENT[CODE[c]][[0,1,7,14,15]],1) for c in MEAS})
```