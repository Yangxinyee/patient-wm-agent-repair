I typed the stay list in that request by hand instead of loading sel1.json, the same slip as fwd_6. So it ran on a different 100 tuning stays: 55 with insulin after hour 12 and 30 with norepinephrine. The results still compare original and edited on identical stays and random numbers:

| Readout | Original | Edited |
|---|---|---|
| Insulin ×2 | −1.87 | −9.85 |
| Insulin ×½ | +0.76 | +4.13 |
| Norepinephrine ×2 | −0.51 | −0.80 |
| Norepinephrine ×½ | −0.70 | −1.91 |

Insulin works in the right direction, and the model's persistence amplifies the one-step correction about fourfold. It may still be short of the truth.

Norepinephrine is asymmetric. Halving now lowers MAP more, which is the right direction. Doubling still lowers MAP, even though the true effect is positive. My guess is that rates above 1.0, never seen in training, make the model, and possibly the readout, behave oddly. The original model already gives a negative shift for ×2.

To judge any of this I need truth estimates on exactly these rows. Next I will:
- save this stay list;
- look at the shifts by hour since the first dose;
- check the signature of my `true_shift` helper, so the next run can apply the impulse responses estimated from the records to these stays' recorded doses.

```python
st10=[2020000000,2020000008,2020000013,2020000014,2020000015,2020000018,2020000020,2020000021,2020000024,2020000026,2020000027,2020000029,2020000031,2020000034,2020000035,2020000037,2020000041,2020000044,2020000048,2020000051,2020000054,2020000057,2020000058,2020000060,2020000062,2020000064,2020000065,2020000068,2020000070,2020000073,2020000077,2020000078,2020000079,2020000082,2020000083,2020000086,2020000088,2020000089,2020000091,2020000094,2020000095,2020000097,2020000099,2020000102,2020000106,2020000108,2020000109,2020000110,2020000112,2020000113,2020000116,2020000117,2020000121,2020000122,2020000126,2020000128,2020000131,2020000134,2020000137,2020000138,2020000140,2020000141,2020000143,2020000145,2020000148,2020000149,2020000150,2020000151,2020000152,2020000155,2020000158,2020000159,2020000163,2020000164,2020000166,2020000169,2020000171,2020000172,2020000174,2020000176,2020000179,2020000181,2020000182,2020000183,2020000185,2020000187,2020000189,2020000191,2020000193,2020000194,2020000196,2020000198,2020000200,2020000201,2020000203,2020000205,2020000207,2020000208,2020000210,2020000211]
json.dump(st10,open('/work/sel10.json','w')); print('overlap with sel1:',len(set(st10)&set(lists['sel1'])))
for fnm in ['true_shift','lp','deconv','drug_after12','lp_ne','lp_ne2','long_table']:
    f=globals()[fnm]; c=f.__code__; print(fnm,c.co_varnames[:c.co_argcount],f.__defaults__)
do=pd.read_csv('/work/fwd_10/dose_edit_original.csv'); de=pd.read_csv('/work/fwd_10/dose_edit_edited.csv')
print(de.head(3).to_string()); print(de.edit.unique())
for nm,d_ in [('orig',do),('edit',de)]:
    r_=d_[d_.reached.astype(bool)]
    print(nm, r_.groupby('edit')['shift'].agg(['mean','count']).round(3).to_dict())
```