wc_vec's 'last' argument unpacks into four values, matching its variable names (has, agemin, zl, vb). I'll call it with "no previous reading" (0, 0, 0, 0) and test the same identity: the difference in with_code between two states should equal the difference in my q_of. An error near 0 validates q_of.

```python
lastc=(0,0.0,0.0,0)
a1=wc_vec(St_[k1-1], dt1, 'glucose_finger', lastc); a2=wc_vec(St_[k2-1], dt1, 'glucose_finger', lastc)
d_wc=a1-a2; d_q=q_of(St_[k1-1], dt1)[0]-q_of(St_[k2-1], dt1)[0]
print('identity err', np.abs(d_wc-d_q).max().round(6), 'scale', np.abs(d_wc).max().round(3))
rowk=[r_ for r_ in gen_rows_all(df_,St=St_) if r_[0]=='glucose_finger' and abs(r_[2]-tb_[k1])<1e-6]
print('row at k1:', [ (r_[2], r_[3]) for r_ in rowk])
```