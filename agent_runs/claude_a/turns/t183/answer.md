The README is short. Times are hours since ICU admission, infusion records each last one hour, and the policy is not described, as expected.

Now the timing question. For insulin and norepinephrine starts I'll compute the minute within the hour, and the share that fall in the same minute as a glucose reading in the same stay.
- **Same minute:** a single gated write at the insulin bundle can see both the dose and the triggering glucose.
- **Different minutes:** the policy rate must come from the state or from a separate write at the glucose bundle.

```python
ev=tr['event'].values; hr=tr['hour'].values; sid=tr['stay_id'].values
mins=lambda m: np.round((hr[m]%1)*60).astype(int)
for e_ in ['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']:
    m=ev==e_; mm=mins(m); u,c=np.unique(mm,return_counts=True); o=np.argsort(-c)[:4]
    print(e_, m.sum(), 'top minutes', list(zip(u[o].tolist(), np.round(c[o]/m.sum(),3).tolist())))
key=lambda m: pd.Index(list(zip(sid[m], np.round(hr[m]*60).astype(int))))
gk=set(key(ev=='glucose_finger')) | set(key(ev=='glucose_lab'))
ik=key(ev=='insulin'); print('insulin starts sharing minute with a glucose reading: %.3f'%np.mean([k in gk for k in ik[:200000]]))
```