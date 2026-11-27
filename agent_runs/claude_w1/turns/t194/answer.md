q_of is not right yet. The residual spread for first-of-code rows is 1.1–1.7, not 0. Either the "no previous reading" part varies (for example an age term), or my q differs from wc_vec's.

To separate the two, I'll call wc_vec directly with the same code and last-reading argument but two different previous states. The difference of its outputs must then equal q_of(prev1) − q_of(prev2). If it does, q_of is right and the spread came from the last-numeric part; if not, q_of itself is wrong. I'll try a few formats for 'last' until one is accepted.

```python
s_=list(G3.keys())[1]; df_=G3[s_]; B_=bundles_of_stay(df_,RU0); St_=run_stay(B_); tb_=np.array([b_['t'] for b_ in B_])
k1,k2 = 10, 30; dt1=(tb_[k1]-tb_[k1-1])*3600; dt2=(tb_[k2]-tb_[k2-1])*3600
ok_=None
for cand in [None, (np.nan,np.nan), (150.0, tb_[k1]-1), {'glucose_finger':(150.0,tb_[k1]-1)}, (150.0, 60.0)]:
    try:
        a1=wc_vec(St_[k1-1], dt1, 'glucose_finger', cand); ok_=cand; break
    except Exception as ex: print('fail', type(cand).__name__, repr(ex)[:120])
print('ok last format:', ok_)
a1=wc_vec(St_[k1-1], dt1, 'glucose_finger', ok_); a2=wc_vec(St_[k2-1], dt1, 'glucose_finger', ok_)
d_wc = a1-a2; d_q = q_of(St_[k1-1], dt1)[0]-q_of(St_[k2-1], dt1)[0]
print('identity err', np.abs(d_wc-d_q).max().round(6), 'scale', np.abs(d_wc).max().round(3))
```