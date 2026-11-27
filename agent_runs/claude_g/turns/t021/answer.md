Of the 400 tuning stays, 234 receive insulin at or after hour 12 and 86 receive norepinephrine with an arterial line. 50 stays have all three. To measure all four readouts in one forward pass, I will build a set of 100 stays:
- the 50 with both drugs and an arterial line;
- 30 with insulin only;
- 20 with norepinephrine and an arterial line but no insulin.

I will save this list for reuse.

```python
ne_art_noins = info[(info.ins12==0)&(info.ne12>0)&(info.art12>0)].index.tolist()
set100 = [int(s) for s in both[:50] + ins_only[:30] + ne_art_noins[:20]]
print(len(set100), len(set(set100)))
json.dump(set100, open('/work/set100.json','w'))
print(json.dumps(set100))
```