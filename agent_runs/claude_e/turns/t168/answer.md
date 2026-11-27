The saved edit is exactly gated3(-1.9, 3), and fwd_3 holds the original model's factual output on 100 tuning stays. A factual "both" pass on those same stays will give the exact NLL change of the current edit on random stays like the evaluation's.

If the cost there is well below the 0.0052 measured on both-drug training stays, I have room for a larger insulin coefficient. If it is near or above 0.005, I have to cut the expensive old NE term first. I'll write out the stay list.

```python
st3 = sorted(f3.stay_id.unique().tolist())
print(len(st3), st3[:3], st3[-3:])
print(json.dumps(st3)[:120])
json.dump(st3, open('/work/sel_f3.json','w'))
```