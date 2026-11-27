The ×3 prelude scale hurts factual fit badly. Value NLL went from 1.6346 to 1.6516 (+0.017, over budget), driven by glucose (finger +0.027, lab +0.063). So readings also run through the prelude attention, which makes this route non-selective.

The insulin response only grew 26% when the attention output tripled. To see why, I'll split the shift by hours since 12. If the lag-1 shift (before the model's drawn readings feed back) roughly tripled, then the feedback dynamics are absorbing it. If it did not, the dose signal is not carried linearly through the attention output.

```python
with open('/work/notes.md','a') as f:
    f.write("- Pass 11/12: prelude.attn.out x3: ins x2 -2.33, half +0.89, ne x2 -0.78, half -0.98; factual NLL 1.6516 (+0.017; gf +0.027, gl +0.063) -> readings also use prelude attn; not selective. Rejected.\n")
de=pd.read_csv('/work/fwd_11/dose_edit_edited.csv'); do=pd.read_csv('/work/fwd_11/dose_edit_original.csv')
print(de.columns.tolist()[:12], len(de))
print(de.head(3).to_string()[:800])
```