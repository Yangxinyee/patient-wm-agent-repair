The dose_edit files have per stay-hour columns: stay_id, hour, edit, shift, shift_se, factual, reached. I'll compute the insulin x2 shift for the original (fwd_1), N5 (fwd_6), N20 (fwd_9) and N5+H2.5 (fwd_25). I'll split it by each stay's recorded insulin rate from hour 12 on, and set it against my record-based true effect, which is linear in dose. If N5's response flattens at high doses while N20's does not, the main readout is limited by saturation rather than potency. I'll first check the tuning file's columns.

```python
tu=pd.read_csv('/data/tuning.csv'); print(list(tu.columns)[:15], len(tu)); print(tu.head(3).to_string()[:600])
```