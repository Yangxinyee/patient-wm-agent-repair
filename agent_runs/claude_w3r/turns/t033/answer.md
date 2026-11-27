The vocab gives the details I need:
- 4 attention heads.
- z = (x − median)/IQR, clipped to ±10.
- Value bins: 16 fine bins per code. glucose_finger centres run 73…271 with widths of about 10–17 mg/dL around 150–250.
- Predicted mean = Σ p(b)·centre.

Under ×2, norepinephrine rates above 0.5 become > 1.0. That puts them in rate bin 16, a bin norepinephrine never occupied in training (its top edges are 1.0), with z_rate beyond the training range. The norepinephrine ×2 inputs are therefore out of distribution, which may explain the model's odd negative answer. For insulin, doubled rates mostly stay within bins 1–16 (bin 16 is ≥ 5.5) and z ≤ 10. Only the value NLL of glucose and MAP readings counts for factual fit, so action heads are free to change.

Next I will check from the states whether the model's state linearly encodes recent doses. I will choose 100 training stays with both drugs and an arterial line, then request their states in the following step.

```python
with open('/work/notes.md','a') as f:
    f.write("""- VOCAB: 4 heads; z=(x-median)/iqr clip ±10; bins 1..16 = 1+#edges below x. gf centres 73..271 (bins ~10-17 wide mid-range).
  insulin rate med 1.5 iqr 1, edges top 5.0 (bin16 = >5). NOREPI rate med .46 iqr .6, edges top 1.0 -> doubled norepi >1.0 lands in bin16 & z>0.9:
  NEVER seen for norepi in training (OOD) -> explains odd ne_x2 answer. Only value NLL of glucose/MAP readings counts for factual fit.
""")
ins_st = set(tr[(tr.event=='insulin')&(tr.hour>=12)].stay_id); ne_st=set(tr[(tr.event=='norepinephrine')&(tr.hour>=12)].stay_id)
art_st = set(tr[tr.event=='map_arterial'].stay_id)
cand = sorted(ins_st & ne_st & art_st); print('cand', len(cand))
rng=np.random.default_rng(1); selS=[int(s) for s in rng.choice(cand,100,replace=False)]
json.dump(selS, open('/work/selS.json','w')); print(json.dumps(selS))
```