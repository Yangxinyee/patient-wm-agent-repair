The records are on an hourly grid.
- Readings fall at integer hours, from hour −1 up to about hour 25; the stay ends near 25.17.
- Drug records come 5 minutes after the readings (hour + 0.0833) and each runs for one hour.
- 62% of stays have insulin, 44% have norepinephrine and 27% have both.
- The rates move hour to hour, apparently in response to the readings just taken. In the example, glucose 136 at hour 14 was followed by insulin stopping, and MAP rising was followed by norepinephrine being tapered. This is the "doses follow condition" confounding.

Next I want the stay-level structure, because natural sources of dose variation could act as instruments:
- the stay_id pattern (do prefixes encode sites or policies?);
- stay length;
- reading cadence and missingness per hour;
- how many hours of each stay get a drug.

```python
ids = tr.stay_id.unique()
print(len(ids), ids[:5], ids[-5:])
s = pd.Series(ids).astype(str)
print(s.str.len().value_counts())
for k in range(1,7):
    print(k, s.str[:k].value_counts().head(10).to_dict())
tids = tu.stay_id.unique(); print("tuning", len(tids), tids[:5])
st = tr.groupby('stay_id').agg(hmin=('hour','min'), hmax=('hour','max'), n=('hour','size'))
print(st.describe().to_string())
adm = tr[tr.event=='hospital_admission']; print(adm.hour.value_counts().head(), adm.end_hour.describe())
# per integer hour, frequency of each reading
tr['ih'] = np.floor(tr.hour+1e-6).astype(int)
cnt = tr.groupby(['ih','event']).size().unstack(fill_value=0)
print(cnt.to_string())
```