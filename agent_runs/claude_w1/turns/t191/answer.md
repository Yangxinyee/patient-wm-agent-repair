Bundles carry their time in hours under 't' (−2 admission, −1, 0, 0.083, …), so each row's bundle index and dt can be recovered. Now the test: for first-of-code rows (no previous reading, e.g. the first lab glucose mid-stay), with_code − q_of(prev, dt) should be the same vector across different stays and times for the same code. A spread near 0 confirms q_of; a large spread means my reconstruction of q is off.

```python
import collections
Dby = collections.defaultdict(list)
for s_ in list(G3.keys())[:25]:
    df_ = G3[s_]; B_ = bundles_of_stay(df_, RU0); St_ = run_stay(B_); R_ = gen_rows_all(df_, St=St_)
    tb_ = np.array([b_['t'] for b_ in B_])
    for r_ in R_:
        if np.isnan(r_[3]) and r_[2] > -1:
            k_ = int(np.argmin(np.abs(tb_ - r_[2])))
            q_, _ = q_of(St_[k_-1], (tb_[k_]-tb_[k_-1])*3600.0)
            Dby[r_[0]].append(r_[8]-q_)
for c_,L_ in Dby.items():
    L_=np.array(L_); print(c_, len(L_), 'spread', np.abs(L_-L_[0]).max().round(5))
```