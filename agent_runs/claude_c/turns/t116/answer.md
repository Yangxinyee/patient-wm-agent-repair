The tuning records list each drug infusion as a row with its rate, dose, unit and end hour. That lets me tag each of the 100 tuning stays by drug exposure from hour 12 on:
- whether norepinephrine reaches the 1.0 cap;
- the mean insulin rate, split at its median.

I will then recompute the pooled readouts (mean shift over reached stay-hours) for each group, using the original model (pass 1) and the dose-columns-x5 edit (pass 4). The question is whether the wrong-signed norepinephrine response is confined to capped stays. If non-capped stays show x2 positive and x1/2 negative, the existing pathway is right-signed but too weak. If they are wrong-signed too, norepinephrine needs a new pathway.

```python
dr = tu[(tu.hour>=12)&(tu.event.isin(['insulin','norepinephrine']))&(tu.stay_id.isin(stays100))]
ne_max = dr[dr.event=='norepinephrine'].groupby('stay_id').rate.max()
ins_mean = dr[dr.event=='insulin'].groupby('stay_id').rate.mean()
cap_st = set(ne_max[ne_max>=0.999].index); ne_st = set(ne_max.index)
ins_hi = set(ins_mean[ins_mean>=ins_mean.median()].index); ins_st=set(ins_mean.index)
print('NE stays', len(ne_st), 'cap', len(cap_st), '| insulin stays', len(ins_st))
de4 = pd.read_csv('/work/fwd_4/dose_edit_edited.csv')
def grp_read(de, edit, ids):
    x = de[(de.edit==edit)&de.reached&de.stay_id.isin(ids)]
    return '%.2f (n=%d)' % (x['shift'].mean(), len(x)) if len(x) else 'na'
for lab, de in [('orig',de1),('num_x5',de4)]:
    print(lab, 'NE x2 cap', grp_read(de,'ne_x2',cap_st), 'noncap', grp_read(de,'ne_x2',ne_st-cap_st),
          '| NE half cap', grp_read(de,'ne_half',cap_st), 'noncap', grp_read(de,'ne_half',ne_st-cap_st))
    print(lab, 'INS x2 hi', grp_read(de,'insulin_x2',ins_hi), 'lo', grp_read(de,'insulin_x2',ins_st-ins_hi),
          '| INS half hi', grp_read(de,'insulin_half',ins_hi), 'lo', grp_read(de,'insulin_half',ins_st-ins_hi))
```