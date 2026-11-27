On the fresh stays N5 again improves on the original in the same directions. Insulin x2 goes from -1.88 to -2.33 and insulin x1/2 from +0.87 to +1.82. Norepinephrine x2 goes from -0.31 to +0.18 (now the right sign). Norepinephrine x1/2 goes from -0.65 to -0.24, still negative but weaker.

To put these in context I'll compute the record-based true effects on the same reached stay-hours using `true_in` and `true_ne`. I'll record both sets in the notebook. I expect truths around -13, +6.5, +4.6 and -2.3, as on the first 100 stays.

```python
d30=pd.read_csv('/work/fwd_30/dose_edit_original.csv'); tr={}
for ed,fn,f in [('insulin_x2',true_in,2.0),('insulin_half',true_in,0.5),('ne_x2',true_ne,2.0),('ne_half',true_ne,0.5)]:
    dd=d30[(d30.edit==ed)&(d30.reached==1)]; tr[ed]=round(np.mean([fn(s,h,f) for s,h in zip(dd.stay_id,dd.hour)]),2)
print(tr)
open('/work/notes.md','a').write(f"- Pass 29: N5 factual on tuning 100-199: +0.00201 (1.62044->1.62245). Pooled 200 stays ~+0.0018.\n- Pass 30: dose_edit both on 100-199: orig -1.879/+0.868/-0.312/-0.647; N5 -2.328/+1.819/+0.179/-0.239; truth {tr}\n")
```