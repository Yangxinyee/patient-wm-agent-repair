The architecture confirms the edit is complete. Drug atoms always carry a rate and a dose, so they always use bins 1–16, and bin 0 means "absent", which only non-drug atoms use. Every bin a drug can reach, at any dose including 2.0, is now the same row 8. The z_dose and z_rate inputs are zeroed. The model therefore cannot see dose magnitude.

The only other place rates appear is the "last rate" register used when predicting a drug's own rate. That affects drug-rate predictions, not glucose or MAP values, so it touches neither the readouts nor the judged NLL.

So the zero shifts are structural, and the causal analysis supports zero for all four readouts (insulin +0.31/−0.16, norepinephrine +0.15/−0.07, all within about 1 SE of 0).

The one hard constraint left is factual fit on unseen stays. fwd_2 used the drug-heavy tuning stays. I'll draw 100 other tuning stays for an independent factual check of original vs edited. The difference should stay of order 1e-4, far below 0.005.

```python
with open('/work/notes.md','a') as f: f.write("""
- Run 49: insulin clean residual (exact-integer reading cells, split readings 150/180/220/260/300 excluded; these get half-tier doses):
  exogenous; impulses k1..12 null; implied x2 +0.31 (se .22-.73), half -0.16; lab x2 +0.10. => insulin null confirmed.
- Run 50: ARCHITECTURE: drug atoms always use fine bins 1..16 (0=absent); z_dose,z_rate via embed.numeric. Our edit (rows1-16->row8, z cols->0)
  makes drug magnitude invisible for any edited rate. 'last rate' register only feeds drug-rate heads (not value NLL / readouts).
- Next: one more factual 'both' on 100 other tuning stays (sel2) to confirm NLL margin; then FINAL.
""")
tu_ids=np.unique(tu['stay_id'].values)
sel1_ids=set(json.load(open('/work/sel1.json')))
rest=[int(s) for s in tu_ids if int(s) not in sel1_ids]
rng=np.random.default_rng(7); sel2=sorted(rng.choice(rest,100,replace=False).tolist())
json.dump(sel2,open('/work/sel2.json','w'))
print(len(tu_ids),len(rest)); print(json.dumps(sel2))
```