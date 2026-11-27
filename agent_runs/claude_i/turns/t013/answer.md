The next-hour effect is not quite linear in rate. Within the 150–180 band, relative to 0.5 u/h, rates of 1.0, 1.5 and 2.0 give −0.73, −1.81 and −3.07. Within the 180–220 band the slope is about −2 per u/h above 1.5 but flatter from 1.0 to 1.5. That points to either a convex dose-response or the recorded rate being a rounded version of a continuous true rate. I'll need to settle this, because the x0, x1.5 and x3 edits are scored.

The effect also persists: about −1.9 at h+1 and h+2, decaying to −0.36 by h+6. That decay mixes washout, mean reversion and policy feedback.

There is also a selection problem. The next finger-stick is taken 96% of the time after insulin but only 29% otherwise. So outcomes several hours ahead are selected on post-treatment glucose. A proper estimate will need a state-space style model, which I'll build later.

First I want to see what the model currently answers, the format of the readout, and how far off it is. To choose stays for the forward pass, I'll list tuning stays that have insulin and/or norepinephrine at or after hour 12.

```python
PT = panel(tune)
def drug_after(Pp, col, h0=12):
    x = Pp[col].notna() & (Pp.index.get_level_values('h')>=h0)
    return x.groupby(level='stay_id').sum()
ti = drug_after(PT,'ins'); tn = drug_after(PT,'ne')
both_t = ti[(ti>0)&(tn>0)].index.tolist()
ins_only = ti[(ti>0)&(tn==0)].index.tolist()
ne_only = tn[(tn>0)&(ti==0)].index.tolist()
neither = ti[(ti==0)&(tn==0)].index.tolist()
print(len(both_t), len(ins_only), len(ne_only), len(neither))
sel1 = both_t[:60] + ins_only[:20] + ne_only[:20]
print(len(sel1))
json.dump({'sel1': [int(s) for s in sel1], 'both_t':[int(s) for s in both_t], 'ins_only':[int(s) for s in ins_only], 'ne_only':[int(s) for s in ne_only], 'neither':[int(s) for s in neither]}, open('/work/sel.json','w'))
print(json.dumps([int(s) for s in sel1]))
```