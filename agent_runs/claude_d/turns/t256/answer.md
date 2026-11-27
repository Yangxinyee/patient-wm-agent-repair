This explains the norepinephrine x2 shortfall. Recorded norepinephrine rates never exceed 1.0: the rate bin edges stop at 1.0, the last four are all 1.0, and the dose edges stop at 4.8. Doubling any rate above 0.5 therefore produces an input the model never saw in training:
- the top fine bin (16), which is untrained for norepinephrine;
- z_rate up to 2.6, against a training maximum of 0.92.

Halving keeps every rate in range, which is why x1/2 (−2.15) responds almost linearly while x2 (+1.03) gets roughly zero response from the stays with r > 0.5. The hourly profile fits this: x2 builds only to about +1.5, while x1/2 builds to −3.

Because norepinephrine never takes these bins in the records, editing how such rates are embedded could fix x2 at almost no factual cost. That holds only if the bin tables are not shared with insulin rates that do reach the top bin.

To check, I will get:
- the shapes of the bin and numeric tables;
- insulin's rate and dose edges;
- how often insulin and norepinephrine fall in the top bin in the training records;
- the share of norepinephrine rates above 0.5.

```python
print({n: SHAPES[n] for n in NAMES if n.startswith('embed.bins') or n.startswith('embed.numeric') or n.startswith('encode_action')})
ins_ent = [x for x in nu if x['code']=='insulin']; print([(x['attribute'], x['median'], x['iqr'], x['fine_bin_edges'][-5:]) for x in ins_ent])
for drug in ['insulin','norepinephrine']:
    t_ = TRAIN[TRAIN.event==drug]
    for att in ['rate','dose']:
        e_ = [x for x in nu if x['code']==drug and x['attribute']==att][0]['fine_bin_edges']; v_ = t_[att].dropna().values
        b_ = 1 + (np.asarray(e_)[None,:] < v_[:,None]).sum(1)
        print(drug, att, len(v_), 'max', round(float(v_.max()),3), 'bin16 frac', round(float((b_==16).mean()),4), 'x2->bin16', round(float((1+(np.asarray(e_)[None,:] < 2*v_[:,None]).sum(1)==16).mean()),3), 'frac>half-top', round(float((v_>e_[-1]/2).mean()),3))
```