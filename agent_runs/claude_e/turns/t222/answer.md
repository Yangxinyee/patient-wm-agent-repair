`datasets` has 6 entries. The first five are the 100-stay train batches, which `A_` covers (27,963 rows). The sixth is the sel1 states paired with the mixed `RR`, which broke my offsets. I'll repeat the in-sample check on the first five only.

```python
cin = []; off = 0
for i in range(5):
    Xs_i, Rr_i = datasets[i]
    o_i = head_run(Xs_i, Rr_i, Eu)
    dpm_i = np.asarray(o_i[0], float) - np.asarray(base_out[i][0], float)
    hi_i = A_['hi'].values[off:off+len(Rr_i)].astype(bool); off += len(Rr_i)
    msk_i = (Rr_i.code.values==6) & hi_i
    cin.append(pd.DataFrame({'dpm': dpm_i[msk_i], 'dev1': Rr_i.dev1.values[msk_i]}))
CI = pd.concat(cin, ignore_index=True)
print('in-sample 6D rows', len(CI), 'corr(dpm, dev1) %.3f' % np.corrcoef(CI.dpm, CI.dev1)[0,1], '| gain dpm/dev1 %.3f' % np.polyfit(CI.dev1, CI.dpm, 1)[0], '| sd dev1 %.3f' % CI.dev1.std())
s_ = R1[(R1.code==6) & R1.drug]
o_s1 = head_run(X, RR, Eu); dpm_s1 = np.asarray(o_s1[0], float)[m1][((R1.code==6) & R1.drug).values] - s_.pm_o.values
print('sel1 out-of-sample corr(dpm, dev1t) %.3f gain %.3f' % (np.corrcoef(dpm_s1, s_.dev1t)[0,1], np.polyfit(s_.dev1t, dpm_s1, 1)[0]))
```